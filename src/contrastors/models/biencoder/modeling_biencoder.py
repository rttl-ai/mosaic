from contextlib import nullcontext
from functools import partial

import os 
import numpy as np
import re
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.ops.rms_norm import RMSNorm
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from safetensors import safe_open

from contrastors.layers.activations import quick_gelu
from contrastors.layers.attention import FlashAttentionPooling
from contrastors.layers.block import Block
from contrastors.layers.mlp import MLP, GatedMLP
from contrastors.models.decoder import DecoderModel
from contrastors.models.decoder.gpt_neox import gpt_neox_config_to_gpt2_config
from contrastors.models.decoder.open_lm import open_lm_config_to_gpt2_config
from contrastors.models.decoder.llama import llama_config_to_gpt2_config
from contrastors.models.encoder import NomicBertModel, bert_config_to_nomic_config
from contrastors.models.encoder.configuration_nomic_bert import NomicBertConfig

from contrastors.models.vit import (
    ViTModel,
    clip_config_to_vit_config,
    dino_config_to_vit_config,
    hf_vit_config_to_vit_config,
    timm_name_to_vit_config,
)


def remove_bert(state_dict):
    def remove_bert_prefix(key):
        return re.sub(r"^bert\.", "", key)

    filtered = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("cls.predictions."):
            continue  # drop MLM head
        filtered[remove_bert_prefix(k)] = v
    return filtered



def is_local_model_path(model_path):
    return os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, "config.json"))

class LogitScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(config.logit_scale), requires_grad=config.trainable_logit_scale
        )

    def forward(self, x):
        return x * self.logit_scale.exp()

    def __repr__(self):
        return f"LogitScale(logit_scale={self.logit_scale.exp().item()}, trainable={self.logit_scale.requires_grad})"


class ClsSelector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, input_ids, attention_mask):
        return hidden_states[:, 0]


class LastTokenPooling(nn.Module):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id

    def forward(self, hidden_states, input_ids, attention_mask):
        # get the embedding corresponding to the first eos token
        # we don't substract 1 because the eos token is already included in the input_ids and attention_mask
        # and we want to get the embedding of the last token
        sequence_lengths = attention_mask.sum(-1) - 1
        selected_tokens = input_ids[torch.arange(input_ids.shape[0]), sequence_lengths]

        if not torch.all(selected_tokens == self.eos_token_id):
            raise ValueError(
                f"The last token of the input_ids is not the eos token: {selected_tokens}\n{input_ids}\n{sequence_lengths}"
            )
        prev_token = input_ids[torch.arange(input_ids.shape[0]), sequence_lengths - 1]
        if torch.any(prev_token == self.eos_token_id):
            raise ValueError(
                f"The second to last token of the input_ids is the eos token: {selected_tokens}\n{input_ids}\n{sequence_lengths}"
            )

        embs = hidden_states[torch.arange(hidden_states.shape[0]), sequence_lengths]

        return embs


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, attention_mask):
        if attention_mask is None:
            return torch.mean(hidden_states, dim=1)
        # Masked mean
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = mask.sum(dim=1)
        return summed / counts.clamp(min=1e-6)


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, config):
        # adapted from https://github.com/google-research/big_vision/blob/474dd2ebde37268db4ea44decef14c7c1f6a0258/big_vision/models/vit.py#L158
        super().__init__()
        self.attn = FlashAttentionPooling(config)
        activation = (
            F.sigmoid
            if config.activation_function == "glu"
            else (
                F.silu
                if config.activation_function == "swiglu"
                else (quick_gelu if config.activation_function == "quick_gelu" else F.gelu)
            )
        )
        if config.activation_function in ["glu", "swiglu"]:
            self.mlp = GatedMLP(
                config.n_embd,
                hidden_features=config.n_inner,
                bias1=config.mlp_fc1_bias,
                bias2=config.mlp_fc2_bias,
                activation=activation,
                fused_bias_fc=config.fused_bias_fc,
            )
        else:
            self.mlp = MLP(
                config.n_embd,
                hidden_features=config.n_inner,
                bias1=config.mlp_fc1_bias,
                bias2=config.mlp_fc2_bias,
                activation=activation,
                fused_bias_fc=config.fused_bias_fc,
            )
        norm_cls = partial(
            nn.LayerNorm if not config.use_rms_norm else RMSNorm,
            eps=config.layer_norm_epsilon,
        )
        self.norm1 = norm_cls(config.n_embd)

    def forward(self, hidden_states, input_ids, attention_mask):
        if attention_mask is not None:
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(hidden_states, attention_mask)
        else:
            indices = None
            cu_seqlens = None
            max_seqlen_in_batch = None

        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            is_padded_inputs=True,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_k=max_seqlen_in_batch,
        )

        normed = self.norm1(attn_outputs)
        hidden_states = hidden_states + self.mlp(normed)
        if attention_mask is not None:
            hidden_states = pad_input(hidden_states, indices, cu_seqlens, max_seqlen_in_batch)

        return hidden_states[:, 0]



class BiEncoder(PreTrainedModel):
    _tied_weights_keys = ["trunk.cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)


        # The rest of the code is preserved, including support for all trunk loading/decoding
        if config.use_fused_kernels:
            print(f"Initializing {config.model_name}, pretrained={config.pretrained}")
            if getattr(config, "nomic_encoder", True) is False:
                model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
                if "dinov2" in config.model_name:
                    model_config = dino_config_to_vit_config(model_config)
                    self.trunk = ViTModel.from_pretrained(config.model_name, config=model_config) if config.pretrained else ViTModel(config=model_config)
                elif "CLIP" in config.model_name or "openai/clip" in config.model_name:
                    model_config = clip_config_to_vit_config(model_config)
                    model_config.patch_dropout = config.patch_dropout
                    self.trunk = ViTModel.from_pretrained(config.model_name, config=model_config) if config.pretrained else ViTModel(config=model_config)
                elif any(x in config.model_name for x in ["google/vit", "facebook/dino-vit", "facebook/vit-mae"]):
                    model_config = hf_vit_config_to_vit_config(model_config)
                    model_config.patch_dropout = config.patch_dropout
                    self.trunk = ViTModel.from_pretrained(config.model_name, config=model_config) if config.pretrained else ViTModel(config=model_config)
                elif config.model_name.startswith("timm/") or "eva02" in config.model_name:
                    model_config = timm_name_to_vit_config(config.model_name)
                    self.trunk = ViTModel.from_pretrained(config.model_name, config=model_config) if config.pretrained else ViTModel(config=model_config)
                else:
                    if "gpt-neox" in config.model_name or "pythia" in config.model_name:
                        model_config = gpt_neox_config_to_gpt2_config(model_config)
                    elif "open_lm" in config.model_name:
                        model_config = open_lm_config_to_gpt2_config(model_config)
                    elif "meta-llama" in config.model_name:
                        model_config = llama_config_to_gpt2_config(model_config)
                    self.trunk = DecoderModel.from_pretrained(config.model_name, config=model_config) if config.pretrained else DecoderModel(config=model_config)
                    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                    self.eos_token_id = tokenizer.eos_token_id
            else:
                if is_local_model_path(config.model_name):
                    model_config = NomicBertConfig.from_pretrained(config.model_name)
                    model_config.domain_token_ids_path = getattr(config, "domain_token_ids_path", None)
                    self.trunk = NomicBertModel(config=model_config, add_pooling_layer=False)
                    if config.pretrained:
                        safetensors_path = os.path.join(config.model_name, "model.safetensors")
                        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                            state_dict = {k: f.get_tensor(k) for k in f.keys()}
                        if any(k.startswith("bert.") for k in state_dict):
                            state_dict = remove_bert(state_dict)
                        #self.trunk.load_state_dict(state_dict)
                        self.trunk.load_state_dict(state_dict, strict=False)  # <<<<<< HIGHLIGHTED PATCH: allow missing/unexpected keys
                        self.trunk.tie_weights()  # Re-tie MLM decoder weights after loading

                else:
                    model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
                    if model_config.model_type != "nomic_bert":
                        model_config = bert_config_to_nomic_config(model_config)
                    model_config.domain_token_ids_path = getattr(config, "domain_token_ids_path", None)
                    if config.pretrained:
                        self.trunk = NomicBertModel.from_pretrained(
                            config.model_name,
                            add_pooling_layer=False,
                            config=model_config,
                            rotary_scaling_factor=None,
                            num_experts=getattr(config, "num_experts", 0),
                            moe_top_k=getattr(config, "moe_top_k", 1),
                            router_aux_loss_coef=getattr(config, "router_aux_loss_coef", None),
                            moe_impl=getattr(config, "moe_impl", "megablocks"),
                            ffn_div=getattr(config, "ffn_div", 1.0),
                            moe_normalize_expert_weights=getattr(config, "moe_normalize_expert_weights", False),
                            expert_choice_router=getattr(config, "expert_choice_router", False),
                            pad_vocab_size_multiple=getattr(config, "pad_vocab_to_multiple_of", 1),
                            num_shared_experts=getattr(config, "num_shared_experts", 0),
                            resid_pdrop=getattr(config, "resid_pdrop", None),
                            moe_every_n_layers=getattr(config, "moe_every_n_layers", 1),
                            moe_resid_pdrop=getattr(config, "moe_resid_pdrop", None),
                        )
                    else:
                        model_config.num_experts = getattr(config, "num_experts", 0)
                        model_config.moe_top_k = getattr(config, "moe_top_k", 1)
                        model_config.router_aux_loss_coef = getattr(config, "router_aux_loss_coef", None)
                        model_config.moe_impl = getattr(config, "moe_impl", "megablocks")
                        model_config.moe_normalize_expert_weights = getattr(config, "moe_normalize_expert_weights", False)
                        model_config.expert_choice_router = getattr(config, "expert_choice_router", False)
                        self.trunk = NomicBertModel(config=model_config, add_pooling_layer=False)
        else:
            self.trunk = AutoModel.from_pretrained(config.model_name, trust_remote_code=True, add_pooling_layer=False)

        if config.freeze:
            self.trunk.eval()
            for param in self.trunk.parameters():
                param.requires_grad = False
            self.frozen_trunk = True
        else:
            self.frozen_trunk = False

        if config.gradient_checkpointing:
            self.trunk.gradient_checkpointing_enable({"use_reentrant": False})

        if config.projection_dim:
            self.proj = nn.Linear(self.trunk.config.hidden_size, config.projection_dim)
        else:
            self.proj = nn.Identity()

        # Force mean pooling only:
        self.pooling = MeanPooling()
        # No selector/last/cls/MAP pooling logic.

        if config.hamming:
            self.hamming = nn.LayerNorm(self.trunk.config.hidden_size, elementwise_affine=False)
        else:
            self.hamming = nn.Identity()

    def forward(self, input_ids, attention_mask=None, labels=None, is_padded_inputs=True, normalize=True, binarize=False, **kwargs):
        context = torch.no_grad if self.frozen_trunk else nullcontext
        with context():
            outputs = self.trunk(input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        # Get last_hidden_state for embedding
        hidden_states = outputs.last_hidden_state
        embedding = self.pooling(hidden_states, attention_mask)
        embedding = self.hamming(embedding)
        embedding = self.proj(embedding)

        if embedding.dtype != hidden_states.dtype:
            embedding = embedding.to(hidden_states.dtype)

        if binarize:
            embedding = (embedding > 0).float()
        elif normalize:
            embedding = F.normalize(embedding, p=2, dim=-1)

        result = {"embedding": embedding}
        # Return MLM loss if available
        if hasattr(outputs, "loss") and outputs.loss is not None:
            result["mlm_loss"] = outputs.loss

        return result