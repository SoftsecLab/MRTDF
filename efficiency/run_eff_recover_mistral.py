import os
import json
import time
import random
from typing import Any, Dict, List, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import psutil
except ImportError:
    psutil = None


SEED = 42
WARMUP_N = 10
RUN_N = 110
TIMEOUT = None
NEUTRAL_FIELD = "neutralized_text"


TARGET_GPU = 0
DEVICE_STR = f"cuda:{TARGET_GPU}" if torch.cuda.is_available() else "cpu"


from config import OUTPUT_MISTRAL_SAVE_DIR
from utils.recovery import load_recovery_model


INSTRUCTION = (
    "### Instruction: Rewrite the following sentence in a natural, human-like style.\n"
    "Input: {}\n"
    "Output:"
)

GEN_CONFIG = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.85,
    "top_p": 0.92,

}


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Not a list json: {path}")
    return data

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def now_ms() -> float:
    return time.perf_counter() * 1000.0

def get_rss_mb(proc) -> float:
    if proc is None:
        return -1.0
    return proc.memory_info().rss / (1024 * 1024)

def decode_one(decoded: str) -> str:
    return decoded.split("Output:")[-1].strip() if "Output:" in decoded else decoded.strip()

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize(TARGET_GPU)

def cuda_reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(TARGET_GPU)

def cuda_peak_mb() -> float:
    if not torch.cuda.is_available():
        return -1.0
    return torch.cuda.max_memory_allocated(TARGET_GPU) / (1024 * 1024)


def load_model_single_gpu():
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_MISTRAL_SAVE_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = load_recovery_model(
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    if torch.cuda.is_available():
        model = model.to(DEVICE_STR)

    model.eval()

    gen_cfg = dict(GEN_CONFIG)
    gen_cfg["pad_token_id"] = tokenizer.eos_token_id
    return tokenizer, model, gen_cfg


@torch.no_grad()
def recover_one(neutral_text: str, tokenizer, model, gen_cfg) -> Tuple[str, Dict[str, float]]:

    prompt = INSTRUCTION.format(neutral_text)


    t_tok0 = now_ms()
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    if torch.cuda.is_available():
        enc = {k: v.to(DEVICE_STR) for k, v in enc.items()}
    t_tok1 = now_ms()


    cuda_reset_peak()
    cuda_sync()
    t_gen0 = now_ms()
    out_ids = model.generate(**enc, **gen_cfg)
    cuda_sync()
    t_gen1 = now_ms()


    t_dec0 = now_ms()
    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    restored = decode_one(decoded)
    t_dec1 = now_ms()

    timings = {
        "tokenize_ms": round(t_tok1 - t_tok0, 3),
        "generate_ms": round(t_gen1 - t_gen0, 3),
        "decode_ms": round(t_dec1 - t_dec0, 3),
    }
    return restored, timings


def run(samples_path: str, out_log_jsonl: str):
    random.seed(SEED)
    samples = load_json_list(samples_path)

    need = WARMUP_N + RUN_N
    if len(samples) < need:
        raise RuntimeError(f"samples too small: {len(samples)} < warmup+run={need}")

    eval_samples = samples[:need]
    ensure_dir(os.path.dirname(out_log_jsonl))

    proc = psutil.Process(os.getpid()) if psutil else None
    peak_rss = get_rss_mb(proc)

    print(f"🔧 Step2 recover (single GPU) | device={DEVICE_STR} | warmup={WARMUP_N} timed={RUN_N}")
    tokenizer, model, gen_cfg = load_model_single_gpu()

    with open(out_log_jsonl, "w", encoding="utf-8") as f:
        for idx, it in enumerate(tqdm(eval_samples, desc="Step2-Recover", ncols=100)):
            phase = "warmup" if idx < WARMUP_N else "timed"

            src = it.get(NEUTRAL_FIELD, "")
            if not isinstance(src, str) or not src.strip():
                row = {
                    "step": "recover",
                    "phase": phase,
                    "idx": idx,
                    "status": "skip_empty_neutral",
                    "id": it.get("id"),
                    "category": it.get("category"),
                    "peak_rss_mb": round(peak_rss, 3),
                    "cuda_peak_mb": -1.0,
                    "device": DEVICE_STR,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            cuda_sync()
            t0 = now_ms()

            err = ""
            ok = True
            restored = ""

            try:
                restored, tparts = recover_one(src, tokenizer, model, gen_cfg)
                ok = bool(restored)
                if not ok:
                    err = "empty_generation"
            except Exception as e:
                ok = False
                err = repr(e)
                tparts = {"tokenize_ms": float("nan"), "generate_ms": float("nan"), "decode_ms": float("nan")}

            cuda_sync()
            t1 = now_ms()

            if psutil:
                peak_rss = max(peak_rss, get_rss_mb(proc))

            row = {
                "step": "recover",
                "phase": phase,
                "idx": idx,
                "status": "ok" if ok else "fail",
                "err": "" if ok else err,

                "id": it.get("id"),
                "category": it.get("category"),

                "e2e_ms": round(t1 - t0, 3),
                "tokenize_ms": tparts["tokenize_ms"],
                "generate_ms": tparts["generate_ms"],
                "decode_ms": tparts["decode_ms"],

                "peak_rss_mb": round(peak_rss, 3),
                "cuda_peak_mb": round(cuda_peak_mb(), 3),
                "device": DEVICE_STR,
            }


            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (idx + 1) % 10 == 0:
                print(f"[{idx+1}/{len(eval_samples)}] phase={phase} status={row['status']} e2e_ms={row['e2e_ms']} cuda_peak_mb={row['cuda_peak_mb']}")

    print(f"✅ done. log saved to {out_log_jsonl}")


if __name__ == "__main__":


    samples_path = "efficiency/sampled/human_samples.json"
    out_log = "efficiency/logs/human_step2_recover.jsonl"
    run(samples_path, out_log)
