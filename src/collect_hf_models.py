import re
import pandas as pd
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()

def extract_params(text):
    text = text.lower()

    patterns = [
        r"(\d+\.?\d*)\s*billion",
        r"(\d+\.?\d*)\s*bn",
        r"(\d+\.?\d*)\s*b\b",
        r"(\d+\.?\d*)\s*million",
        r"(\d+\.?\d*)\s*mn",
        r"(\d+\.?\d*)\s*m\b",
    ]

    for p in patterns:
        match = re.search(p, text)
        if match:
            num = float(match.group(1))
            if "b" in p or "billion" in p or "bn" in p:
                return int(num * 1_000_000_000)
            if "m" in p or "million" in p or "mn" in p:
                return int(num * 1_000_000)

    return None


def detect_gpu(text):
    text = text.lower()

    gpu_keywords = {
        "a100": 400,
        "h100": 700,
        "v100": 300,
        "t4": 70,
        "rtx 3090": 350,
        "rtx 4090": 450,
        "l4": 72,
    }

    for gpu, tdp in gpu_keywords.items():
        if gpu in text:
            return gpu, tdp

    return None, None


def detect_train_type(text):
    text = text.lower()

    if "fine-tuned" in text or "finetuned" in text or "fine tuned" in text:
        return "fine-tune"
    if "pretrained" in text or "pre-trained" in text or "pre training" in text:
        return "pre-train"

    return "unknown"


def detect_task(model_info):
    if model_info.pipeline_tag:
        return model_info.pipeline_tag
    return "unknown"


rows = []

models = list(api.list_models(
    limit=1000,
    sort="downloads",
    
))

for model in tqdm(models):
    model_id = model.modelId

    try:
        readme_path = hf_hub_download(
            repo_id=model_id,
            filename="README.md",
            repo_type="model"
        )

        with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
            card_text = f.read()

        n_params = extract_params(card_text)
        gpu_model, tdp_w = detect_gpu(card_text)
        train_type = detect_train_type(card_text)
        task = detect_task(model)

        rows.append({
            "model_id": model_id,
            "n_params": n_params,
            "gpu_model": gpu_model,
            "tdp_w": tdp_w,
            "train_type": train_type,
            "task": task,
            "downloads": model.downloads,
            "likes": model.likes,
            "source": "huggingface"
        })

    except Exception:
        continue


df = pd.DataFrame(rows)
df.to_csv("data/hf_raw_models.csv", index=False)

print("Saved data/hf_raw_models.csv")
print(df.head())
print(df.info())