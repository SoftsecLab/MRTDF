import os
import json
import time
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import logging

logging.set_verbosity_error()

try:
    import psutil
except ImportError:
    psutil = None

from utils.projection import Reducer
from utils.feature_extractors import get_pure_style_embedding
from config import REDUCER_SAVE_PATH


SEED = 42
WARMUP_N = 10
RUN_N = 110

TARGET_GPU = 0
DEVICE_STR = f"cuda:{TARGET_GPU}" if torch.cuda.is_available() else "cpu"

RESTORED_FIELD = "restore_text"


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

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize(TARGET_GPU)

def cuda_reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(TARGET_GPU)

def cuda_peak_alloc_mb() -> float:
    if not torch.cuda.is_available():
        return -1.0
    return torch.cuda.max_memory_allocated(TARGET_GPU) / (1024 * 1024)

def get_rss_mb(proc) -> float:
    if proc is None:
        return -1.0
    return proc.memory_info().rss / (1024 * 1024)

def get_machine_orig_text(item: Dict[str, Any]) -> str:

    k = next((kk for kk in item.keys() if isinstance(kk, str) and kk.endswith("output")), None)
    if not k:
        return ""
    v = item.get(k, "")
    return v if isinstance(v, str) else ""

def get_human_orig_text(item: Dict[str, Any]) -> str:
    v = item.get("text", "")
    return v if isinstance(v, str) else ""


def load_reducer():
    reducer = Reducer()
    reducer.load_state_dict(torch.load(REDUCER_SAVE_PATH, map_location="cpu"))
    reducer.eval()
    if torch.cuda.is_available():
        reducer = reducer.to(DEVICE_STR)
    return reducer


@torch.no_grad()
def extract_style_384_timed(orig: str, restored: str, reducer: torch.nn.Module) -> Tuple[torch.Tensor, Dict[str, float]]:


    t0 = now_ms()
    v_orig = get_pure_style_embedding(orig)
    t1 = now_ms()

    t2 = now_ms()
    v_rest = get_pure_style_embedding(restored)
    t3 = now_ms()


    t4 = now_ms()
    v_orig = v_orig.unsqueeze(0)
    v_rest = v_rest.unsqueeze(0)
    if torch.cuda.is_available():
        v_orig = v_orig.to(DEVICE_STR)
        v_rest = v_rest.to(DEVICE_STR)

    r_orig = reducer(v_orig).squeeze(0)
    r_rest = reducer(v_rest).squeeze(0)
    cuda_sync()
    t5 = now_ms()


    t6 = now_ms()
    diff = r_orig - r_rest
    feat = torch.cat([r_orig, r_rest, diff], dim=0)
    t7 = now_ms()

    parts = {
        "embed_orig_ms": round(t1 - t0, 3),
        "embed_rest_ms": round(t3 - t2, 3),
        "reducer_ms": round(t5 - t4, 3),
        "concat_ms": round(t7 - t6, 3),
    }
    return feat, parts


def run(samples_path: str, out_log_jsonl: str, dataset_type: str):

    random.seed(SEED)
    samples = load_json_list(samples_path)

    need = WARMUP_N + RUN_N
    if len(samples) < need:
        raise RuntimeError(f"samples too small: {len(samples)} < warmup+run={need}")

    eval_samples = samples[:need]
    ensure_dir(os.path.dirname(out_log_jsonl))

    proc = psutil.Process(os.getpid()) if psutil else None
    peak_rss = get_rss_mb(proc)

    reducer = load_reducer()

    print(f"🔧 Step3 feature-only | device={DEVICE_STR} | warmup={WARMUP_N} timed={RUN_N} | type={dataset_type}")
    print(f"🔧 restored_field={RESTORED_FIELD}")

    with open(out_log_jsonl, "w", encoding="utf-8") as f:
        for idx, it in enumerate(tqdm(eval_samples, desc=f"Step3-Feat({dataset_type})", ncols=100)):
            phase = "warmup" if idx < WARMUP_N else "timed"


            if dataset_type == "human":
                orig = get_human_orig_text(it)
            else:
                orig = get_machine_orig_text(it)

            restored = it.get(RESTORED_FIELD, "")
            if not isinstance(restored, str):
                restored = ""

            if not orig.strip():
                row = {
                    "step": "encode_reducer",
                    "phase": phase,
                    "idx": idx,
                    "status": "skip_empty_orig",
                    "id": it.get("id"),
                    "category": it.get("category"),
                    "device": DEVICE_STR,
                    "peak_rss_mb": round(peak_rss, 3),
                    "cuda_peak_mb": -1.0,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            if not restored.strip():
                row = {
                    "step": "encode_reducer",
                    "phase": phase,
                    "idx": idx,
                    "status": "skip_empty_restored",
                    "id": it.get("id"),
                    "category": it.get("category"),
                    "device": DEVICE_STR,
                    "peak_rss_mb": round(peak_rss, 3),
                    "cuda_peak_mb": -1.0,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            cuda_reset_peak()
            cuda_sync()
            t0 = now_ms()

            ok = True
            err = ""
            parts: Dict[str, float] = {}
            try:

                _feat, parts = extract_style_384_timed(orig, restored, reducer=reducer)
            except Exception as e:
                ok = False
                err = repr(e)

            cuda_sync()
            t1 = now_ms()

            if psutil:
                peak_rss = max(peak_rss, get_rss_mb(proc))

            row = {
                "step": "encode_reducer",
                "phase": phase,
                "idx": idx,
                "status": "ok" if ok else "fail",
                "err": "" if ok else err,

                "id": it.get("id"),
                "category": it.get("category"),

                "e2e_ms": round(t1 - t0, 3),
                "embed_orig_ms": parts.get("embed_orig_ms", float("nan")),
                "embed_rest_ms": parts.get("embed_rest_ms", float("nan")),
                "reducer_ms": parts.get("reducer_ms", float("nan")),
                "concat_ms": parts.get("concat_ms", float("nan")),

                "peak_rss_mb": round(peak_rss, 3),
                "cuda_peak_mb": round(cuda_peak_alloc_mb(), 3),
                "device": DEVICE_STR,
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (idx + 1) % 10 == 0:
                print(f"[{idx+1}/{len(eval_samples)}] phase={phase} status={row['status']} e2e_ms={row['e2e_ms']} cuda_peak_mb={row['cuda_peak_mb']}")

    print(f"✅ done. log saved to {out_log_jsonl}")


if __name__ == "__main__":

    run(
        samples_path="efficiency/sampled/human_samples.json",
        out_log_jsonl="efficiency/logs/human_step3_encode_reducer.jsonl",
        dataset_type="human"
    )
