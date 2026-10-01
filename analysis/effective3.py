import os
import json
import numpy as np
from tqdm import tqdm

from config import MODIFY_TEST_DIR, BERTSCORE_MODEL, require_path


N_BOOT = 1000
BOOT_SEED = 42
ALPHA = 0.05
OUTPUT_PATH = "results/semantic_preservation_bertscore_bootstrap.json"


BERTSCORE_NUM_LAYERS = 24
LANG = "en"
BATCH_SIZE = 32

try:
    from bert_score import score as bert_score
except Exception as e:
    raise RuntimeError(
        "bert-score is not installed. Run:\n"
        "  pip install bert-score torch\n"
        f"Original error: {e}"
    )


def get_output_key(item: dict):
    return next((k for k in item.keys() if k.endswith("output")), None)

def collect_pairs_full():
    require_path(MODIFY_TEST_DIR, 'MODIFY_TEST_DIR')
    pairs = []
    files = [fn for fn in os.listdir(MODIFY_TEST_DIR) if fn.endswith(".json")]
    files.sort()

    for fname in files:
        fp = os.path.join(MODIFY_TEST_DIR, fname)
        with open(fp, "r", encoding="utf-8") as f:
            items = json.load(f)

        for item in items:
            out_k = get_output_key(item)
            if out_k is None:
                continue

            hwt = item.get("HWT_sentence", "")
            mrt = item.get(out_k, "")
            snt_hwt = item.get("neutralized_hwt", "")
            snt_mrt = item.get("neutralized_text", "")

            if not (isinstance(hwt, str) and hwt.strip()):
                continue
            if not (isinstance(mrt, str) and mrt.strip()):
                continue
            if not (isinstance(snt_hwt, str) and snt_hwt.strip()):
                continue
            if not (isinstance(snt_mrt, str) and snt_mrt.strip()):
                continue

            pairs.append({
                "hwt": hwt.strip(),
                "mrt": mrt.strip(),
                "snt_hwt": snt_hwt.strip(),
                "snt_mrt": snt_mrt.strip(),
            })

    return pairs


def nanmean(x):
    x = np.array(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    return float(np.mean(x)) if len(x) else float("nan")

def nanstd(x):
    x = np.array(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    return float(np.std(x)) if len(x) else float("nan")

def bootstrap_ci_mean(values, n_boot=N_BOOT, seed=BOOT_SEED, alpha=ALPHA):
    v = np.array(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = len(v)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = np.mean(v[idx])

    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi

def bootstrap_ci_mean_diff(a, b, n_boot=N_BOOT, seed=BOOT_SEED, alpha=ALPHA):

    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    m = (~np.isnan(a)) & (~np.isnan(b))
    a = a[m]; b = b[m]
    if len(a) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = len(a)
    stats = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[k] = np.mean(b[idx] - a[idx])

    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi


def compute_bertscore_f1(
    refs,
    hyps,
    model_type=BERTSCORE_MODEL,
    num_layers=BERTSCORE_NUM_LAYERS,
    lang=LANG,
    batch_size=BATCH_SIZE,
):

    model_type = require_path(model_type, 'BERTSCORE_MODEL')
    P, R, F1 = bert_score(
        cands=hyps,
        refs=refs,
        lang=lang,
        model_type=model_type,
        num_layers=num_layers,
        batch_size=batch_size,
        verbose=False,
        rescale_with_baseline=True,
    )
    return F1.detach().cpu().numpy().astype(np.float64)


def main():
    pairs = collect_pairs_full()
    print(f"✅ Collected {len(pairs)} paired samples from MODIFY_TEST_DIR (full set).")


    hwt_refs = [p["hwt"] for p in pairs]
    hwt_hyps = [p["snt_hwt"] for p in pairs]

    mrt_refs = [p["mrt"] for p in pairs]
    mrt_hyps = [p["snt_mrt"] for p in pairs]

    print(
        f"\n🔍 Computing BERTScore-F1 "
        f"(model={BERTSCORE_MODEL}, num_layers={BERTSCORE_NUM_LAYERS}, "
        f"rescale_with_baseline=True) ..."
    )

    hwt_f1 = compute_bertscore_f1(hwt_refs, hwt_hyps)
    mrt_f1 = compute_bertscore_f1(mrt_refs, mrt_hyps)


    hwt_mu, hwt_sd = nanmean(hwt_f1), nanstd(hwt_f1)
    mrt_mu, mrt_sd = nanmean(mrt_f1), nanstd(mrt_f1)

    hwt_ci = bootstrap_ci_mean(hwt_f1)
    mrt_ci = bootstrap_ci_mean(mrt_f1)


    diff = mrt_f1 - hwt_f1
    diff_mu, diff_sd = nanmean(diff), nanstd(diff)
    diff_ci = bootstrap_ci_mean_diff(hwt_f1, mrt_f1)

    summary = {
        "n_pairs_full": len(pairs),
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "alpha": ALPHA,
        "bertscore": {
            "model_type": BERTSCORE_MODEL,
            "num_layers": BERTSCORE_NUM_LAYERS,
            "lang": LANG,
            "batch_size": BATCH_SIZE,
            "rescale_with_baseline": True
        },
        "HWT_preservation": {
            "metric": "BERTScore-F1",
            "mean": hwt_mu,
            "std": hwt_sd,
            "ci95": hwt_ci
        },
        "MRT_preservation": {
            "metric": "BERTScore-F1",
            "mean": mrt_mu,
            "std": mrt_sd,
            "ci95": mrt_ci
        },
        "Diff_MRT_minus_HWT": {
            "metric": "BERTScore-F1",
            "mean": diff_mu,
            "std": diff_sd,
            "ci95": diff_ci
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


    print(f"\n✅ Saved summary to: {OUTPUT_PATH}\n")
    print("--------------------------------------------------")
    print("Semantic Preservation (BERTScore-F1)")
    print(f"HWT: orig vs snt_hwt  : {hwt_mu:.4f} ± {hwt_sd:.4f}   95%CI [{hwt_ci[0]:.4f}, {hwt_ci[1]:.4f}]")
    print(f"MRT: orig vs snt_mrt  : {mrt_mu:.4f} ± {mrt_sd:.4f}   95%CI [{mrt_ci[0]:.4f}, {mrt_ci[1]:.4f}]")
    print(f"Δ (MRT - HWT)         : {diff_mu:.4f} ± {diff_sd:.4f}   95%CI [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
