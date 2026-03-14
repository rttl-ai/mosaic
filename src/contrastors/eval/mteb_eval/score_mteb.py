import sys
import pandas as pd
from huggingface_hub.repocard import metadata_load

# Supported MTEB task types
TASKS = [
    "Retrieval",
    "STS",
]

# Biomedical-relevant datasets only
TASK_LIST_RETRIEVAL = [
    "CUREv1",
    "MedicalQARetrieval",
    "NFCorpus",
    "PublicHealthQA",
    "SciFact",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
]

TASK_TO_METRIC = {
    "Retrieval": "ndcg_at_10",
    "STS": "cos_sim_spearman",
}

# Consolidated biomedical datasets
BIOMEDICAL_DATASETS = TASK_LIST_RETRIEVAL + TASK_LIST_STS

# Load metadata from mteb_metadata.md file
metadata_path = sys.argv[1]
meta = metadata_load(metadata_path)

# Extract relevant results
task_results = [
    sub_res
    for sub_res in meta["model-index"][0]["results"]
    if sub_res.get("task", {}).get("type", "") in TASKS
    and any(x in sub_res.get("dataset", {}).get("name", "") for x in BIOMEDICAL_DATASETS)
]

# Map dataset name -> score
out = []
for res in task_results:
    name = res["dataset"]["name"].replace("MTEB ", "").strip()
    metric_type = TASK_TO_METRIC[res["task"]["type"]]
    score = [
        round(score["value"], 2)
        for score in res["metrics"]
        if score["type"] == metric_type
    ][0]
    out.append({name: score})

# Format output as table
out = {k: v for d in out for k, v in d.items()}
df = pd.DataFrame([out])

# Insert biomedical retrieval and STS averages
df.insert(1, f"Retrieval Average ({len(TASK_LIST_RETRIEVAL)} datasets)", df[TASK_LIST_RETRIEVAL].mean(axis=1))
df.insert(2, f"STS Average ({len(TASK_LIST_STS)} datasets)", df[TASK_LIST_STS].mean(axis=1))

# Print and save
df = df.T.reset_index()
df.columns = ["Dataset", "Score"]
print(df.to_markdown())

with open("results.md", "w") as f:
    f.write(df.to_markdown())
