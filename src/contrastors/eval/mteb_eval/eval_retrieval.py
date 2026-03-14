import logging
import math
import multiprocessing as mp
import os
import queue
from argparse import ArgumentParser

import numpy as np
import torch
from mteb import MTEB
from transformers import AutoTokenizer

from contrastors.models.biencoder import BiEncoder, BiEncoderConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# --- Biomedical-focused Retrieval Tasks ---
TASK_LIST_RETRIEVAL = [
    "CUREv1",
    "MedicalQARetrieval",
    "NFCorpus",
    "PublicHealthQA",
    "SciFact",
    "TRECCOVID",
    # "BIOSSES",  # uncomment if STS is desired here
]


class STransformer:
    def __init__(self, model, add_prefix=False):
        self.model = model
        self.gpu_pool = self.model.start_multi_process_pool()
        self.add_prefix = add_prefix

    def encode(self, sentences, **kwargs):
        if self.add_prefix:
            sentences = [f"query: {sent}" for sent in sentences]
        return self.model.encode_multi_process(sentences, self.gpu_pool, **kwargs)

    def encode_queries(self, queries, **kwargs) -> np.ndarray:
        input_texts = [f'query: {q}' if self.add_prefix else q for q in queries]
        return self.model.encode_multi_process(input_texts, self.gpu_pool, **kwargs)

    def encode_corpus(self, corpus, **kwargs) -> np.ndarray:
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        if self.add_prefix:
            input_texts = ['document: {}'.format(t) for t in input_texts]
        return self.model.encode_multi_process(input_texts, self.gpu_pool, **kwargs)


class CausalModel:
    def __init__(self, model_name):
        config = BiEncoderConfig.from_pretrained(model_name)
        self.model = BiEncoder.from_pretrained(model_name, config=config)
        self.model.to(torch.bfloat16)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.model_max_length = 128
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode(self, sentences, batch_size=256, **kwargs):
        embeddings = []
        device = kwargs.get("device", self.device)
        self.model.to(device)

        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i: i + batch_size]
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                if encoded["input_ids"].shape[1] >= 2048 and batch_size > 256:
                    step_size = 128
                    for j in range(0, len(encoded["input_ids"]), step_size):
                        smaller_batch = {k: v[j: j + step_size].to(device) for k, v in encoded.items()}
                        curr_outputs = self.model(**smaller_batch)
                        embeddings.extend(curr_outputs["embedding"].cpu().float().numpy())
                else:
                    outputs = self.model(**encoded.to(device))
                    embeddings.extend(outputs["embedding"].cpu().float().numpy())

        return embeddings

    def start_multi_process_pool(self, target_devices=None):
        if target_devices is None:
            target_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ['cpu'] * 4
        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(
                target=CausalModel._encode_multi_process_worker,
                args=(cuda_id, self, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}

    @staticmethod
    def _encode_multi_process_worker(target_device: str, model, input_queue, results_queue):
        while True:
            try:
                id, batch_size, sentences = input_queue.get()
                embeddings = model.encode(
                    sentences,
                    device=target_device,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=batch_size,
                )
                results_queue.put([id, embeddings])
            except queue.Empty:
                break

    def encode_multi_process(
        self,
        sentences,
        pool,
        batch_size=128,
        chunk_size=None,
        show_progress_bar=False,
        convert_to_numpy=None,
        convert_to_tensor=None,
    ):
        if chunk_size is None:
            chunk_size = min(math.ceil(len(sentences) / len(pool["processes"]) / 10), 5000)

        logger.debug(f"Chunk data into {math.ceil(len(sentences) / chunk_size)} packages of size {chunk_size}")
        input_queue = pool['input']
        last_chunk_id = 0
        chunk = []

        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, batch_size, chunk])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, batch_size, chunk])
            last_chunk_id += 1

        output_queue = pool['output']
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])
        embeddings = np.concatenate([np.array(t[1]) for t in results_list])
        return embeddings


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument("--add_prefix", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    model = CausalModel(model_name)
    model = STransformer(model, add_prefix=args.add_prefix)

    for task in TASK_LIST_RETRIEVAL:
        logger.info(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])
        evaluation.run(model, output_folder=f"results/{args.model_name}", eval_splits=eval_splits)
