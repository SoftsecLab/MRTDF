import os
import torch
from transformers import set_seed
from methods.fast_detectgpt import run_fast_detectgpt_experiments
from methods.utils_custom import load_data, load_base_model_and_tokenizer, load_base_model


class Args:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discrepancy_analytic = True
    base_model_name = "gpt-neo-2.7B"
    cache_dir = None
    seed = 0

args = Args()


TASK_ID = int(os.environ.get('MRTDF_TASK_ID', '1'))
CKPT_DIR = "./ckpt"
MODEL_NAME = args.base_model_name

MODIFY_FILE_NAMES = [
    "2llama_complete_neutralized.json",
    "2llama_polish_sentence_neutralized.json",
    "2llama_polish_token_neutralized.json",
    "2llama_rewrite_neutralized.json",
    "gpt4_complete_neutralized.json",
    "gpt4_polish_sentence_neutralized.json",
    "gpt4_polish_token_neutralized.json",
    "gpt4_rewrite_neutralized.json"
]


print(f"\n🤖 加载语言模型: {MODEL_NAME}")
base_model, base_tokenizer = load_base_model_and_tokenizer(MODEL_NAME, cache_dir=args.cache_dir)
load_base_model(base_model, args.DEVICE)
base_model.eval()


if TASK_ID == 1:
    print("\n🧪 当前任务：TASK 1（人类 vs 机器修改文本，逐子文件）")
    for fname in MODIFY_FILE_NAMES:
        print(f"\n📂 正在处理文件：{fname}")
        train_data = load_data(task_id=1, split="train", file_filter=fname, human_limit=40, seed=42, mode="zero-shot")
        test_data = load_data(task_id=1, split="test", file_filter=fname, human_limit=10, seed=42)
        data = {"train": train_data, "test": test_data}
        shortname = fname.replace(".json", "")
        run_fast_detectgpt_experiments(
            args=args,
            data=data,
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            test_only=False,
            no_auc=False,
            ckpt_dir=CKPT_DIR
        )

elif TASK_ID in [2, 3]:
    task_name = {2: "TASK 2（人类 vs 机器生成）", 3: "TASK 3（人类 vs 非人类）"}[TASK_ID]
    print(f"\n🧪 当前任务：{task_name}")

    train_data = load_data(task_id=TASK_ID, split="train", human_limit=40, seed=42, mode="zero-shot")
    test_data = load_data(task_id=TASK_ID, split="test", human_limit=10, seed=42)
    data = {"train": train_data, "test": test_data}

    run_fast_detectgpt_experiments(
        args=args,
        data=data,
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        test_only=False,
        no_auc=False,
        ckpt_dir=CKPT_DIR
    )


else:
    raise ValueError(f"❌ 不支持的 TASK_ID: {TASK_ID}")
