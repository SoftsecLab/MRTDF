import os
import json
import torch
import random
from tqdm import tqdm
from transformers import logging
from utils.projection import Reducer
from utils.feature_extractors import get_pure_style_embedding
from config import (
    REDUCER_SAVE_PATH,
    DEVICE,
    HUMAN_TEST_DIR,
    MODIFY_TEST_DIR,
    require_path
)

import argparse

parser = argparse.ArgumentParser(description="Extract main-experiment features; test data only.")
parser.add_argument('--sample-seed', type=int, default=None, help='Optional seed for HWT subsampling; historical membership is not reconstructed')
args = parser.parse_args()
HUMAN_TEST_DIR = require_path(HUMAN_TEST_DIR, 'HUMAN_TEST_DIR')
MODIFY_TEST_DIR = require_path(MODIFY_TEST_DIR, 'MODIFY_TEST_DIR')
if args.sample_seed is not None:
    import random
    random.seed(args.sample_seed)

logging.set_verbosity_error()

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SUBSET_DIR = os.path.join(OUTPUT_DIR, "subsets_task1")
os.makedirs(SUBSET_DIR, exist_ok=True)


MODIFY_TEST_FILES = [
    "2llama_complete_neutralized.json",
    "2llama_polish_sentence_neutralized.json",
    "2llama_polish_token_neutralized.json",
    "2llama_rewrite_neutralized.json",
    "gpt4_complete_neutralized.json",
    "gpt4_polish_sentence_neutralized.json",
    "gpt4_polish_token_neutralized.json",
    "gpt4_rewrite_neutralized.json"
]

HUMAN_TEST_FILES = [
    "arxiv_neutralized.json", "bbc_neutralized.json", "blog_neutralized.json",
    "enron_neutralized.json", "steam_neutralized.json", "ted_neutralized.json"
]

reducer = Reducer()
reducer.load_state_dict(torch.load(REDUCER_SAVE_PATH, map_location=DEVICE))
reducer.to(DEVICE)
reducer.eval()

@torch.no_grad()
def extract_style_384(orig, restored):
    v_orig = reducer(get_pure_style_embedding(orig).unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
    v_restore = reducer(get_pure_style_embedding(restored).unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
    diff = v_orig - v_restore
    final_vec = torch.cat([v_orig, v_restore, diff], dim=0)
    return final_vec.tolist()

def process_machine_file(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    for item in tqdm(items, desc=f"Processing {os.path.basename(path)}"):
        try:
            output_key = next((k for k in item if k.endswith("output")), None)
            if output_key is None:
                continue
            orig = item[output_key]
            restored = item["restore_text"]
            feature = extract_style_384(orig, restored)
            data.append({"feature": feature, "label": 1})
        except Exception as e:
            print(f"⚠️ 跳过异常机器样本: {e}")
    return data

def sample_human_subset():
    result = []
    for hfile in HUMAN_TEST_FILES:
        hpath = os.path.join(HUMAN_TEST_DIR, hfile)
        with open(hpath, "r", encoding="utf-8") as f:
            items = json.load(f)
            sampled = random.sample(items, 10)
            for item in sampled:
                try:
                    orig = item["text"]
                    restored = item["restore_text"]
                    feature = extract_style_384(orig, restored)
                    result.append({"feature": feature, "label": 0})
                except Exception as e:
                    print(f"⚠️ 跳过异常人类样本: {e}")
    return result

if __name__ == "__main__":


    print("\n📦 正在提取测试子集（任务一）...")
    for mod_file in MODIFY_TEST_FILES:
        print(f"\n📂 正在处理子测试文件：{mod_file}")
        test_output_path = os.path.join(SUBSET_DIR, f"style_cls_task1_test_{mod_file.replace('.json', '')}.jsonl")


        human_subset = sample_human_subset()


        mod_path = os.path.join(MODIFY_TEST_DIR, mod_file)
        machine_subset = process_machine_file(mod_path)

        all_subset = human_subset + machine_subset
        print(f"📊 总计样本数: {len(all_subset)}")
        with open(test_output_path, "w", encoding="utf-8") as f_out:
            for item in all_subset:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"✅ 已保存至: {test_output_path}，共计样本数: {len(all_subset)}")
