import os, json, math, argparse, glob
from typing import Any, Dict, List, Optional


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def read_many(patterns: List[str]) -> List[Dict[str, Any]]:
    files = []
    for p in patterns:
        matched = glob.glob(p)
        if matched:
            files.extend(matched)
        else:
            if os.path.exists(p):
                files.append(p)
    files = sorted(list(dict.fromkeys(files)))
    rows_all = []
    for fp in files:
        rows_all.extend(read_jsonl(fp))
    return rows_all


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def percentile(sorted_xs: List[float], p: float) -> float:
    if not sorted_xs:
        return float("nan")
    if p <= 0:
        return sorted_xs[0]
    if p >= 100:
        return sorted_xs[-1]
    k = (len(sorted_xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_xs[int(k)]
    return sorted_xs[f] * (c - k) + sorted_xs[c] * (k - f)


def summarize_numeric(xs: List[float]) -> Dict[str, float]:
    xs = [x for x in xs if x is not None]
    xs = [float(x) for x in xs]
    xs.sort()
    if not xs:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    n = len(xs)
    mean = sum(xs) / n
    return {"n": n, "mean": mean, "p50": percentile(xs, 50), "p95": percentile(xs, 95), "max": xs[-1]}


def summarize_key(rows: List[Dict[str, Any]], key: str, nonneg_only: bool = False) -> Dict[str, float]:
    xs = []
    for r in rows:
        v = _to_float(r.get(key))
        if v is None:
            continue
        if nonneg_only and v < 0:
            continue
        xs.append(v)
    return summarize_numeric(xs)


def add_stats(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    if not a or not b or int(a.get("n", 0)) == 0 or int(b.get("n", 0)) == 0:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": int(min(a["n"], b["n"])),
        "mean": a["mean"] + b["mean"],
        "p50":  a["p50"] + b["p50"],
        "p95":  a["p95"] + b["p95"],
        "max":  a["max"] + b["max"],
    }


def fmt_ms(v: float) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "NA"
    return f"{v:.3f}"


def filter_timed_ok(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("type") == "meta":
            continue
        if r.get("phase") != "timed":
            continue
        if r.get("status") != "ok":
            continue
        out.append(r)
    return out


def group_by_step(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        step = r.get("step")
        if not step:
            step = "neutralize"
        g.setdefault(step, []).append(r)
    return g


def print_latency_block(title: str, rows_ok: List[Dict[str, Any]]):
    e2e = summarize_key(rows_ok, "e2e_ms")


    req = summarize_key(rows_ok, "request_ms")
    bp  = summarize_key(rows_ok, "build_prompt_ms")
    ps  = summarize_key(rows_ok, "parse_ms")


    tok = summarize_key(rows_ok, "tokenize_ms")
    gen = summarize_key(rows_ok, "generate_ms")
    dec = summarize_key(rows_ok, "decode_ms")


    eo  = summarize_key(rows_ok, "embed_orig_ms")
    er  = summarize_key(rows_ok, "embed_rest_ms")
    red = summarize_key(rows_ok, "reducer_ms")
    con = summarize_key(rows_ok, "concat_ms")


    cuda = summarize_key(rows_ok, "cuda_peak_mb", nonneg_only=True)
    rss  = summarize_key(rows_ok, "peak_rss_mb", nonneg_only=True)

    print("\n" + "-" * 90)
    print(f"{title} | timed&ok n={len(rows_ok)}")
    print(f"{'metric':<18} {'n':>6} {'mean':>12} {'p50':>12} {'p95':>12} {'max':>12}")


    print(f"{'e2e_ms':<18} {int(e2e['n']):>6} {fmt_ms(e2e['mean']):>12} {fmt_ms(e2e['p50']):>12} {fmt_ms(e2e['p95']):>12} {fmt_ms(e2e['max']):>12}")

    def maybe_row(name: str, s: Dict[str, float]):
        if int(s.get("n", 0)) > 0:
            print(f"{name:<18} {int(s['n']):>6} {fmt_ms(s['mean']):>12} {fmt_ms(s['p50']):>12} {fmt_ms(s['p95']):>12} {fmt_ms(s['max']):>12}")


    maybe_row("request_ms", req)
    maybe_row("build_prompt_ms", bp)
    maybe_row("parse_ms", ps)

    maybe_row("tokenize_ms", tok)
    maybe_row("generate_ms", gen)
    maybe_row("decode_ms", dec)

    maybe_row("embed_orig_ms", eo)
    maybe_row("embed_rest_ms", er)
    maybe_row("reducer_ms", red)
    maybe_row("concat_ms", con)


    if int(cuda["n"]) > 0:
        print(f"\nCUDA peak (MB): mean={fmt_ms(cuda['mean'])} p95={fmt_ms(cuda['p95'])} max={fmt_ms(cuda['max'])}")
    else:
        print("\nCUDA peak (MB): NA (non-local step or not recorded)")

    if int(rss["n"]) > 0:
        print(f"RSS peak (MB):  mean={fmt_ms(rss['mean'])} p95={fmt_ms(rss['p95'])} max={fmt_ms(rss['max'])}")
    else:
        print("RSS peak (MB):  NA")


def print_pipeline_summary(grouped: Dict[str, List[Dict[str, Any]]]):
    need_steps = ["neutralize", "recover", "encode_reducer"]
    missing = [s for s in need_steps if s not in grouped]
    if missing:
        print("\n" + "-" * 90)
        print(f"PIPELINE SUMMARY: missing steps {missing} -> skip pipeline summary")
        return


    s_neu = summarize_key(grouped["neutralize"], "e2e_ms")
    s_rec = summarize_key(grouped["recover"], "e2e_ms")
    s_enc = summarize_key(grouped["encode_reducer"], "e2e_ms")
    pipeline_lat = add_stats(add_stats(s_neu, s_rec), s_enc)


    cuda_rec = summarize_key(grouped["recover"], "cuda_peak_mb", nonneg_only=True)
    cuda_enc = summarize_key(grouped["encode_reducer"], "cuda_peak_mb", nonneg_only=True)
    cuda_stats = [s for s in [cuda_rec, cuda_enc] if int(s.get("n", 0)) > 0]
    if not cuda_stats:
        pipeline_cuda = {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    else:
        pipeline_cuda = {
            "n": int(min(s["n"] for s in cuda_stats)),
            "mean": max(s["mean"] for s in cuda_stats),
            "p50":  max(s["p50"]  for s in cuda_stats),
            "p95":  max(s["p95"]  for s in cuda_stats),
            "max":  max(s["max"]  for s in cuda_stats),
        }


    rss_neu = summarize_key(grouped["neutralize"], "peak_rss_mb", nonneg_only=True)
    rss_rec = summarize_key(grouped["recover"], "peak_rss_mb", nonneg_only=True)
    rss_enc = summarize_key(grouped["encode_reducer"], "peak_rss_mb", nonneg_only=True)
    rss_stats = [s for s in [rss_neu, rss_rec, rss_enc] if int(s.get("n", 0)) > 0]
    if not rss_stats:
        pipeline_rss = {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    else:
        pipeline_rss = {
            "n": int(min(s["n"] for s in rss_stats)),
            "mean": max(s["mean"] for s in rss_stats),
            "p50":  max(s["p50"]  for s in rss_stats),
            "p95":  max(s["p95"]  for s in rss_stats),
            "max":  max(s["max"]  for s in rss_stats),
        }

    print("\n" + "-" * 90)
    print("PIPELINE SUMMARY (approximate): neutralize + recover + encode_reducer")
    print(f"{'metric':<22} {'n':>6} {'mean':>12} {'p50':>12} {'p95':>12} {'max':>12}")

    print(f"{'pipeline_latency_ms':<22} {int(pipeline_lat['n']):>6} "
          f"{fmt_ms(pipeline_lat['mean']):>12} {fmt_ms(pipeline_lat['p50']):>12} "
          f"{fmt_ms(pipeline_lat['p95']):>12} {fmt_ms(pipeline_lat['max']):>12}")

    if int(pipeline_cuda["n"]) > 0:
        print(f"{'pipeline_cuda_peak_mb':<22} {int(pipeline_cuda['n']):>6} "
              f"{fmt_ms(pipeline_cuda['mean']):>12} {fmt_ms(pipeline_cuda['p50']):>12} "
              f"{fmt_ms(pipeline_cuda['p95']):>12} {fmt_ms(pipeline_cuda['max']):>12}")
    else:
        print(f"{'pipeline_cuda_peak_mb':<22} {0:>6} {'NA':>12} {'NA':>12} {'NA':>12} {'NA':>12}")

    if int(pipeline_rss["n"]) > 0:
        print(f"{'pipeline_rss_peak_mb':<22} {int(pipeline_rss['n']):>6} "
              f"{fmt_ms(pipeline_rss['mean']):>12} {fmt_ms(pipeline_rss['p50']):>12} "
              f"{fmt_ms(pipeline_rss['p95']):>12} {fmt_ms(pipeline_rss['max']):>12}")
    else:
        print(f"{'pipeline_rss_peak_mb':<22} {0:>6} {'NA':>12} {'NA':>12} {'NA':>12} {'NA':>12}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human_logs", nargs="+", default=["efficiency/logs/human_*.jsonl"])
    ap.add_argument("--machine_logs", nargs="+", default=["efficiency/logs/machine_*.jsonl"])
    ap.add_argument("--separate", action="store_true")
    args = ap.parse_args()

    rows_h = read_many(args.human_logs)
    rows_m = read_many(args.machine_logs)
    rows_all = rows_h + rows_m

    def run_one(tag: str, rows: List[Dict[str, Any]]):
        rows_ok = filter_timed_ok(rows)
        grouped = group_by_step(rows_ok)

        print("\n" + "=" * 90)
        print(f"DATASET: {tag} | total lines={len(rows)} | timed&ok={len(rows_ok)} | steps={sorted(grouped.keys())}")
        for step_name in sorted(grouped.keys()):
            print_latency_block(f"STEP={step_name}", grouped[step_name])

        print_pipeline_summary(grouped)

    run_one("ALL", rows_all)
    if args.separate:
        run_one("HUMAN", rows_h)
        run_one("MACHINE", rows_m)


if __name__ == "__main__":
    main()
