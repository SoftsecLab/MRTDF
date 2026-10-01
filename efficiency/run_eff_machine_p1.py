import os
import json
import time
import random
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional

import requests


try:
    import psutil
except ImportError:
    psutil = None
import os
import json
import time
import random
from typing import Dict, Any, List, Tuple, Optional

import requests

try:
    import psutil
except ImportError:
    psutil = None


SEED = 42
WARMUP_N = 10
RUN_N = 110
TIMEOUT = 60
RETRIES = 2
BACKOFF = 1.5


API_KEY = os.environ.get("MRTDF_API_KEY", "")
API_URL = os.environ.get("MRTDF_API_URL", "https://api.deepseek.com/v1/chat/completions")
MODEL_NAME = os.environ.get("MRTDF_NEUTRALIZER", "deepseek-chat")


GENERATION_CONFIG = {
    "temperature": 0.3,


}


def build_prompt(text: str) -> str:
    return (
        "Simplify the text to make it clear and concise while preserving its meaning. "
        "Only output the simplified text without any explanations or additional comments.\n\n"
        f"\"{text}\""
    )


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


def get_output_key(item: Dict[str, Any]) -> Optional[str]:
    return next(
        (k for k in item.keys() if isinstance(k, str) and k.endswith("output")),
        None,
    )


def call_api_once(text: str) -> Tuple[str, int, int]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": build_prompt(text)}],
        **GENERATION_CONFIG,
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    j = resp.json()

    gen = j["choices"][0]["message"]["content"].strip()
    usage = j.get("usage", {}) or {}
    in_tok = int(usage.get("prompt_tokens", 0) or 0)
    out_tok = int(usage.get("completion_tokens", 0) or 0)
    return gen, in_tok, out_tok


def call_api_with_retry(text: str):
    retry_count = 0
    last_err = ""

    for attempt in range(RETRIES + 1):
        try:
            gen, in_tok, out_tok = call_api_once(text)
            if gen:
                return gen, in_tok, out_tok, retry_count, ""
            last_err = "empty_generation"
        except Exception as e:
            last_err = repr(e)

        if attempt < RETRIES:
            retry_count += 1
            time.sleep((BACKOFF ** attempt) + random.random() * 0.2)

    return "", 0, 0, retry_count, last_err


def run(samples_path: str, out_log_jsonl: str):
    random.seed(SEED)
    samples = load_json_list(samples_path)

    if len(samples) < (WARMUP_N + RUN_N):
        raise RuntimeError(
            f"samples too small: {len(samples)} < warmup+run={WARMUP_N + RUN_N}"
        )

    eval_samples = samples[: (WARMUP_N + RUN_N)]
    ensure_dir(os.path.dirname(out_log_jsonl))

    proc = psutil.Process(os.getpid()) if psutil else None
    peak_rss = get_rss_mb(proc)

    with open(out_log_jsonl, "w", encoding="utf-8") as f:
        for idx, it in enumerate(tqdm(eval_samples, desc="Efficiency Eval", ncols=100)):
            phase = "warmup" if idx < WARMUP_N else "timed"

            out_key = get_output_key(it)
            if out_key is None:
                row = {
                    "phase": phase,
                    "idx": idx,
                    "status": "skip_no_output_key",
                    "id": it.get("id"),
                    "category": it.get("category"),
                    "peak_rss_mb": peak_rss,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            src_text = it.get(out_key, "")
            if not isinstance(src_text, str) or not src_text.strip():
                row = {
                    "phase": phase,
                    "idx": idx,
                    "status": "skip_empty_src",
                    "id": it.get("id"),
                    "category": it.get("category"),
                    "output_key": out_key,
                    "peak_rss_mb": peak_rss,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            t0 = now_ms()

            t_prompt0 = now_ms()
            _ = build_prompt(src_text)
            t_prompt1 = now_ms()

            t_req0 = now_ms()
            gen, in_tok, out_tok, retry_count, err = call_api_with_retry(src_text)
            t_req1 = now_ms()

            t_parse0 = now_ms()
            ok = bool(gen)
            t_parse1 = now_ms()

            t1 = now_ms()

            if psutil:
                peak_rss = max(peak_rss, get_rss_mb(proc))

            row = {
                "phase": phase,
                "idx": idx,
                "status": "ok" if ok else "fail",
                "err": err if not ok else "",

                "id": it.get("id"),
                "category": it.get("category"),
                "output_key": out_key,

                "in_tokens": in_tok,
                "out_tokens": out_tok,

                "e2e_ms": round(t1 - t0, 3),
                "build_prompt_ms": round(t_prompt1 - t_prompt0, 3),
                "request_ms": round(t_req1 - t_req0, 3),
                "parse_ms": round(t_parse1 - t_parse0, 3),
                "retry_count": retry_count,

                "peak_rss_mb": round(peak_rss, 3),
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (idx + 1) % 10 == 0:
                print(
                    f"[{idx+1}/{len(eval_samples)}] "
                    f"phase={phase} status={row['status']} "
                    f"e2e_ms={row.get('e2e_ms')}"
                )

    print(f"✅ done. log saved to {out_log_jsonl}")


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit('Set MRTDF_API_KEY before starting paid API measurements.')
    samples_path = "efficiency/sampled/machine_samples.json"
    out_log = "efficiency/logs/machine_p1_log.jsonl"
    run(samples_path, out_log)
