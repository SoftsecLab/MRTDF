import os
import json
import torch
import time
import random
import re
import numpy as np
from functools import wraps
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import transformers


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[{func.__name__}] Took {end - start:.2f}s\n")
        return result
    return wrapper


def load_data(task_id, split="train", file_filter=None, human_limit=None, seed=42, mode=None, category_filter=None, model_prefix=None):

    base_dir = "data/classified_data"
    result = {"text": [], "label": []}

    def load_json_list(path, field=None, label=None):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if field == "modify":
                key = next((k for k in item if k.endswith("output")), None)
                if not key: continue
                text = item.get(key, "")
            else:
                text = item.get(field, "")
            if text:
                result["text"].append(text)
                result["label"].append(label)
    if task_id == 1:
        human_path = os.path.join(base_dir, "human_text", f"{split}_data")
        modify_path = os.path.join(base_dir, "modify_text", f"{split}_data")

        random.seed(seed)
        json_files = ["arxiv_neutralized.json", "bbc_neutralized.json", "blog_neutralized.json",
                      "enron_neutralized.json", "steam_neutralized.json", "ted_neutralized.json"]
        sample_n = 40 if split == "train" else 10
        for fname in json_files:
            full_path = os.path.join(human_path, fname)
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                texts = [item["text"] for item in data if "text" in item]
                sampled = random.sample(texts, min(sample_n, len(texts)))
                result["text"].extend(sampled)
                result["label"].extend([0] * len(sampled))

        for f in os.listdir(modify_path):
            if f.endswith(".json") and (file_filter is None or f == file_filter):
                with open(os.path.join(modify_path, f), "r", encoding="utf-8") as f_json:
                    data = json.load(f_json)
                    for item in data:
                        key = next((k for k in item if k.endswith("output")), None)
                        if not key:
                            continue
                        text = item[key]
                        if text:
                            result["text"].append(text)
                            result["label"].append(1)

    elif task_id == 2:

        human_path = os.path.join(base_dir, "human_text", f"{split}_data")
        machine_path = os.path.join(base_dir, "machine_text", f"{split}_data")
        for f in os.listdir(human_path):
            if f.endswith(".json"):
                load_json_list(os.path.join(human_path, f), field="text", label=0)
        for f in os.listdir(machine_path):
            if f.endswith(".json"):
                load_json_list(os.path.join(machine_path, f), field="text", label=1)

    elif task_id == 3:

        human_path = os.path.join(base_dir, "human_text", f"{split}_data")
        modify_path = os.path.join(base_dir, "modify_text", f"{split}_data")
        machine_path = os.path.join(base_dir, "machine_text", f"{split}_data")
        for f in os.listdir(human_path):
            if f.endswith(".json"):
                load_json_list(os.path.join(human_path, f), field="text", label=0)
        for f in os.listdir(modify_path):
            if f.endswith(".json"):
                load_json_list(os.path.join(modify_path, f), field="modify", label=1)
        for f in os.listdir(machine_path):
            if f.endswith(".json"):
                load_json_list(os.path.join(machine_path, f), field="text", label=1)


    else:
        raise ValueError(f"不支持的任务编号: {task_id}")

    return result


def cal_metrics(y_true, y_pred, y_prob, no_auc=False):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = -1.0 if no_auc else roc_auc_score(y_true, y_prob)
    return acc, precision, recall, f1, auc

def get_clf_results(x_train, y_train, x_test, y_test, name, test_only, no_auc, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(ckpt_dir, f"{name}.joblib")
    if test_only:
        clf = load(model_path)
    else:
        clf = LogisticRegression(random_state=0).fit(x_train, y_train)
        dump(clf, model_path)

    if test_only:
        train_res = (0, 0, 0, 0, -1.0)
    else:
        pred_train = clf.predict(x_train)
        prob_train = clf.predict_proba(x_train)[:, 1]
        train_res = cal_metrics(y_train, pred_train, prob_train, no_auc)

    pred_test = clf.predict(x_test)
    prob_test = clf.predict_proba(x_test)[:, 1]
    test_res = cal_metrics(y_test, pred_test, prob_test, no_auc)

    return train_res, test_res


def load_base_model_and_tokenizer(name="gpt2", cache_dir=None):
    local_path = os.path.join("models", name)
    print(f"Loading base model from: {local_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(local_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(local_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_base_model(model, device):
    print("Moving base model to device...", end="")
    model.to(device)
    print("done.")
