import json
import logging
import os
import sys
from mteb import MTEB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

results_folder = sys.argv[1].rstrip("/")
model_name = results_folder.split("/")[-1]

all_results = {}

for file_name in os.listdir(results_folder):
    if not file_name.endswith(".json"):
        logger.info(f"Skipping non-json {file_name}")
        continue
    with open(os.path.join(results_folder, file_name), "r", encoding="utf-8") as f:
        results = json.load(f)
        all_results[file_name.replace(".json", "")] = results

# Custom split overrides
TRAIN_SPLIT = ["DanishPoliticalCommentsClassification"]
VALIDATION_SPLIT = ["AFQMC", "Cmnli", "IFlyTek", "TNews", "MSMARCO", "MultilingualSentiment", "Ocnli"]
DEV_SPLIT = [
    "CmedqaRetrieval", "CovidRetrieval", "DuRetrieval", "EcomRetrieval", "MedicalRetrieval",
    "MMarcoReranking", "MMarcoRetrieval", "MSMARCO", "T2Reranking", "T2Retrieval", "VideoRetrieval"
]

# Custom fallback metadata for biomedical datasets
CUSTOM_METADATA = {
    "CUREv1": {
        "type": "retrieval",
        "hf_hub_name": "mteb/cure-v1",
        "eval_langs": ["en"]
    },
    "MedicalQARetrieval": {
        "type": "retrieval",
        "hf_hub_name": "mteb/medical-qa-retrieval",
        "eval_langs": ["en"]
    },
    "PublicHealthQA": {
        "type": "retrieval",
        "hf_hub_name": "mteb/public-health-qa",
        "eval_langs": ["en"]
    },
    "BIOSSES": {
        "type": "sts",
        "hf_hub_name": "tabilab/BIOSSES",
        "eval_langs": ["en"]
    }
}

MARKER = "---"
TAGS = "tags:"
MTEB_TAG = "- mteb"
HEADER = "model-index:"
MODEL = f"- name: {model_name}"
RES = "  results:"

META_STRING = "\n".join([MARKER, TAGS, MTEB_TAG, HEADER, MODEL, RES])

ONE_TASK = "  - task:\n      type: {}\n    dataset:\n      type: {}\n      name: {}\n      config: {}\n      split: {}\n      revision: {}\n    metrics:"
ONE_METRIC = "    - type: {}\n      value: {}"
SKIP_KEYS = ["std", "evaluation_time", "main_score", "threshold"]

for ds_name, res_dict in sorted(all_results.items()):
    try:
        if ds_name in CUSTOM_METADATA:
            mteb_desc = CUSTOM_METADATA[ds_name]
        else:
            mteb_desc = MTEB(tasks=[ds_name.replace("CQADupstackRetrieval", "CQADupstackAndroidRetrieval")]).tasks[0].metadata_dict
    except Exception as e:
        logger.warning(f"Skipping {ds_name}: metadata lookup failed ({e})")
        continue

    hf_hub_name = mteb_desc.get("hf_hub_name", mteb_desc.get("beir_name", "unknown"))
    mteb_type = mteb_desc["type"]
    eval_langs = mteb_desc.get("eval_langs", ["en"])
    revision = res_dict.get("dataset_revision")

    split = "test"
    if (ds_name in TRAIN_SPLIT) and ("train" in res_dict):
        split = "train"
    elif (ds_name in VALIDATION_SPLIT) and ("validation" in res_dict):
        split = "validation"
    elif (ds_name in DEV_SPLIT) and ("dev" in res_dict):
        split = "dev"
    elif "test" not in res_dict:
        logger.info(f"Skipping {ds_name} as split {split} not present.")
        continue

    res_dict = res_dict.get(split)

    for lang in eval_langs:
        mteb_name = f"MTEB {ds_name}"
        mteb_name += f" ({lang})" if len(eval_langs) > 1 else ""

        test_result_lang = res_dict.get(lang) if len(eval_langs) > 1 else res_dict
        if test_result_lang is None:
            continue

        META_STRING += "\n" + ONE_TASK.format(
            mteb_type,
            hf_hub_name,
            mteb_name,
            lang if len(eval_langs) > 1 else "default",
            split,
            revision,
        )

        for metric, score in test_result_lang.items():
            if not isinstance(score, dict):
                score = {metric: score}
            for sub_metric, sub_score in score.items():
                if any(skip_key in sub_metric for skip_key in SKIP_KEYS):
                    continue
                META_STRING += "\n" + ONE_METRIC.format(
                    f"{metric}_{sub_metric}" if metric != sub_metric else metric,
                    round(sub_score * 100, 2)
                )

META_STRING += "\n" + MARKER

output_path = f"./{model_name}/mteb_metadata.md"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    f.write(META_STRING)

print(f"✅ Metadata written to {output_path}")
