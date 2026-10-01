import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def content_hash(rows):
    content = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def collect_outputs(source_dir, manifest):
    test_ids = manifest["test_ids"]
    if not test_ids or any(type(i) is not int for i in test_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("Invalid or duplicate test IDs in the split manifest.")
    outputs = {}
    for entry in manifest["files"]:
        name = entry["source_file"]
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("Invalid source filename in the split manifest.")
        rows = json.loads((source_dir / name).read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != manifest["source_rows_per_file"]:
            raise ValueError(f"{name}: unexpected source format or row count.")
        by_id = {}
        for row in rows:
            if not isinstance(row, dict) or type(row.get("id")) is not int:
                raise ValueError(f"{name}: expected records with integer IDs.")
            if row["id"] in by_id:
                raise ValueError(f"{name}: duplicate ID {row['id']}.")
            by_id[row["id"]] = row
        missing = [i for i in test_ids if i not in by_id]
        if missing:
            raise ValueError(f"{name}: missing test IDs {missing}.")
        fields = ("category", "id", "HWT_sentence", entry["revision_field"])
        selected = []
        for sample_id in test_ids:
            row = by_id[sample_id]
            if any(field not in row for field in fields):
                raise ValueError(f"{name}: missing required fields for ID {sample_id}.")
            selected.append({field: row[field] for field in fields})
        if content_hash(selected) != entry["test_sha256"]:
            raise ValueError(f"{name}: test content differs from the recorded split; "
                             "use the MixSet revision recorded in mrt_test_split.json.")
        output_name = Path(name).stem + "_neutralized.json"
        if output_name in outputs:
            raise ValueError(f"Duplicate output filename: {output_name}.")
        outputs[output_name] = selected
    if not outputs:
        raise ValueError("No source files specified in the split manifest.")
    return outputs


def write_outputs(outputs, output_dir):
    for name in outputs:
        path = output_dir / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Existing output will not be overwritten: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        with (output_dir / name).open("x", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Restore the fixed MRT test split from official MixSet files.")
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Directory containing the eight official separated JSON files.")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/classified_data/modify_text/test_data")
    args = parser.parse_args()
    try:
        manifest = json.loads((ROOT / "mrt_test_split.json").read_text(encoding="utf-8"))
        outputs = collect_outputs(args.source_dir, manifest)
        write_outputs(outputs, args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Prepared {sum(len(rows) for rows in outputs.values())} MRT test samples "
          f"in {len(outputs)} files: {args.output_dir}")


if __name__ == "__main__":
    main()
