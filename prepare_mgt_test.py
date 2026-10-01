import argparse
import hashlib
import json
from pathlib import Path

from prepare_mrt_test import content_hash, write_outputs


ROOT = Path(__file__).resolve().parent


def collect_outputs(source_dir, manifest):
    if manifest["sample_columns"] != ["source_index", "row_index", "token_length"] or manifest["index_base"] != 0:
        raise ValueError("Unsupported sample index format.")
    sources = []
    for source in manifest["sources"]:
        name = source["filename"]
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("Invalid source filename in the split manifest.")
        raw = (source_dir / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"{name}: source checksum mismatch; use the DetectRL revision "
                             "recorded in mgt_test_split.json.")
        rows = json.loads(raw)
        if not isinstance(rows, list) or len(rows) != source["rows"]:
            raise ValueError(f"{name}: unexpected source format or row count.")
        sources.append(rows)
    outputs = {}
    used_samples = set()
    fields = ("text", "label", "data_type", "llm_type")
    for entry in manifest["files"]:
        name = entry["output_file"]
        if Path(name).name != name or not name.endswith(".json") or name in outputs:
            raise ValueError("Invalid or duplicate output filename.")
        if len(entry["samples"]) != manifest["test_rows_per_file"]:
            raise ValueError(f"{name}: unexpected test sample count.")
        selected = []
        for sample in entry["samples"]:
            if not isinstance(sample, list) or len(sample) != 3 or any(type(i) is not int for i in sample):
                raise ValueError(f"{name}: invalid sample index.")
            source_index, row_index, token_length = sample
            if (not 0 <= source_index < len(sources) or
                    not 0 <= row_index < len(sources[source_index]) or token_length <= 0):
                raise ValueError(f"{name}: sample index or token length out of range.")
            key = (source_index, row_index)
            if key in used_samples:
                raise ValueError(f"{name}: duplicate sample index {key}.")
            used_samples.add(key)
            row = sources[source_index][row_index]
            if not isinstance(row, dict) or any(field not in row for field in fields):
                raise ValueError(f"{name}: missing required source fields.")
            if row["label"] != "llm" or row["data_type"] != "direct_prompt" or row["llm_type"] != entry["llm_type"]:
                raise ValueError(f"{name}: sample label, generation type or model mismatch.")
            record = {field: row[field] for field in fields}
            record["token_length"] = token_length
            selected.append(record)
        if content_hash(selected) != entry["test_sha256"]:
            raise ValueError(f"{name}: test content differs from the recorded split.")
        outputs[name] = selected
    if not outputs:
        raise ValueError("No output files specified in the split manifest.")
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Restore the fixed MGT test split from official DetectRL files.")
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Directory containing direct_prompt_train.json and direct_prompt_test.json.")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/classified_data/machine_text/test_data")
    args = parser.parse_args()
    try:
        manifest = json.loads((ROOT / "mgt_test_split.json").read_text(encoding="utf-8"))
        outputs = collect_outputs(args.source_dir, manifest)
        write_outputs(outputs, args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Prepared {sum(len(rows) for rows in outputs.values())} MGT test samples "
          f"in {len(outputs)} files: {args.output_dir}")


if __name__ == "__main__":
    main()
