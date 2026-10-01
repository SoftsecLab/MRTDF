import os, json, random
from typing import List, Dict, Any
from copy import deepcopy
from config import MODIFY_TEST_DIR, HUMAN_TEST_DIR, require_path

SEED = 42
PER_MACHINE_FILE = 15
HUMAN_N = 120


ADD_META = False

def read_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Not a list json: {path}")
    return data

def get_output_key(item: dict):

    return next((k for k in item.keys() if isinstance(k, str) and k.endswith("output")), None)

def write_json_list(rows: List[Dict[str, Any]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def is_valid_machine_item(it: Dict[str, Any]) -> bool:
    out_k = get_output_key(it)
    if out_k is None:
        return False
    if not isinstance(it.get("id", None), (int, str)):
        return False
    if not (isinstance(it.get("category", ""), str) and it.get("category", "").strip()):
        return False
    if not (isinstance(it.get("HWT_sentence", ""), str) and it.get("HWT_sentence", "").strip()):
        return False
    mrt = it.get(out_k, "")
    if not (isinstance(mrt, str) and mrt.strip()):
        return False
    return True

def is_valid_human_item(it: Dict[str, Any]) -> bool:
    if not isinstance(it.get("id", None), (int, str)):
        return False
    if not (isinstance(it.get("category", ""), str) and it.get("category", "").strip()):
        return False
    if not (isinstance(it.get("text", ""), str) and it.get("text", "").strip()):
        return False
    if not (isinstance(it.get("source", ""), str) and it.get("source", "").strip()):
        return False
    return True

def attach_meta(it: Dict[str, Any], source_file: str, index_in_file: int) -> Dict[str, Any]:
    if not ADD_META:
        return it
    it2 = deepcopy(it)
    it2["_meta"] = {"source_file": source_file, "index_in_file": index_in_file}
    return it2

def main():
    require_path(MODIFY_TEST_DIR, 'MODIFY_TEST_DIR')
    require_path(HUMAN_TEST_DIR, 'HUMAN_TEST_DIR')
    random.seed(SEED)


    m_files = sorted([fn for fn in os.listdir(MODIFY_TEST_DIR) if fn.endswith(".json")])[:8]
    machine_samples: List[Dict[str, Any]] = []

    for fn in m_files:
        fp = os.path.join(MODIFY_TEST_DIR, fn)
        items = read_json_list(fp)

        valid_indices = [i for i, it in enumerate(items) if isinstance(it, dict) and is_valid_machine_item(it)]
        if len(valid_indices) < PER_MACHINE_FILE:
            raise RuntimeError(f"{fn} valid={len(valid_indices)} < {PER_MACHINE_FILE}")

        pick = random.sample(valid_indices, k=PER_MACHINE_FILE)
        for idx in pick:

            machine_samples.append(attach_meta(items[idx], fn, idx))


    h_files = sorted([fn for fn in os.listdir(HUMAN_TEST_DIR) if fn.endswith(".json")])
    human_pool: List[Dict[str, Any]] = []

    for fn in h_files:
        fp = os.path.join(HUMAN_TEST_DIR, fn)
        items = read_json_list(fp)

        for i, it in enumerate(items):
            if isinstance(it, dict) and is_valid_human_item(it):
                human_pool.append(attach_meta(it, fn, i))

    if len(human_pool) < HUMAN_N:
        raise RuntimeError(f"Human pool too small: {len(human_pool)} < {HUMAN_N}")

    human_samples = random.sample(human_pool, k=HUMAN_N)


    write_json_list(machine_samples, "efficiency/sampled/machine_samples.json")
    write_json_list(human_samples, "efficiency/sampled/human_samples.json")

    print(f"✅ Machine samples: {len(machine_samples)} (expected 120)")
    print(f"✅ Human samples:   {len(human_samples)} (expected 120)")
    print("✅ Saved to efficiency/sampled/")

if __name__ == "__main__":
    main()
