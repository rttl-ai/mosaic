import logging
import os
import time
from argparse import ArgumentParser

from mteb import MTEB
from contrastors.eval.encoder import Encoder, HFEncoder, STransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

os.environ['OPENBLAS_NUM_THREADS'] = '16'

# --- Define Task Lists by Type ---
TASK_LIST_RETRIEVAL = [
    "CUREv1",
    "MedicalQARetrieval",
    "NFCorpus",
    "PublicHealthQA",
    "SciFact",
    "TRECCOVID",
]

TASK_LIST_CLUSTERING = [
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "BiorxivClusteringP2P",
]

TASK_LIST_STS = [
    "BIOSSES",
]

TASK_LIST = TASK_LIST_RETRIEVAL + TASK_LIST_CLUSTERING + TASK_LIST_STS


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument("--add_prefix", action="store_true")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--no_normalize_classification", action="store_false")
    parser.add_argument("--binarize", action="store_true")
    parser.add_argument("--matryoshka_dim", type=int)
    parser.add_argument("--hf_model", action="store_true")
    parser.add_argument("--query_prefix", type=str, default="search_query: ")
    parser.add_argument("--document_prefix", type=str, default="search_document: ")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    tokenizer_name = args.tokenizer_name
    seq_length = args.seq_length
    no_normalize_classification = args.no_normalize_classification

    if args.hf_model:
        model = HFEncoder(args.model_name, seq_length=args.seq_length)
    else:
        model = Encoder(
            model_name,
            seq_length=seq_length,
            tokenizer_name=tokenizer_name,
            matryoshka_dim=args.matryoshka_dim,
        )

    print(f"Add prefix: {args.add_prefix}")
    model = STransformer(model, add_prefix=args.add_prefix, binarize=args.binarize)

    task2prefix = {}
    for task in TASK_LIST_RETRIEVAL:
        task2prefix[task] = {"query": args.query_prefix, "document": args.document_prefix}
    for task in TASK_LIST_CLUSTERING:
        task2prefix[task] = {"query": "clustering", "document": "clustering"}
    for task in TASK_LIST_STS:
        task2prefix[task] = {"query": "classification", "document": "classification"}

    start = time.time()
    for task in TASK_LIST:
        logger.info(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])

        output_name = f"results/{model_name}binarize_{args.binarize}"
        if args.matryoshka_dim:
            output_name += f"_matryoshka_{args.matryoshka_dim}"

        evaluation.run(
            model,
            output_folder=output_name,
            eval_splits=eval_splits,
            show_progress_bar=True,
            batch_size=64  # Reduced to avoid GPU OOM
        )

    end = time.time()
    print(f"Time taken (mins): {(end - start) / 60}")
