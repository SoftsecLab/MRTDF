import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def _path(name, default):
    raw = os.environ.get(f"MRTDF_{name}", default)
    if raw is None or not str(raw).strip():
        return None
    value = Path(raw).expanduser()
    return str(value if value.is_absolute() else ROOT / value)


def require_path(value, name):
    if value is None or not str(value).strip():
        raise ValueError(f'Configure {name} in config.py or provide the corresponding command-line path.')
    return str(value)


DEVICE = os.environ.get("MRTDF_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

Mistral_PATH = _path("MISTRAL_PATH", "")
SBERT_PATH = _path("SBERT_PATH", "")
DEBERTA_PATH = _path("DEBERTA_PATH", "")
BERT_PATH = _path("BERT_PATH", "")
BERTSCORE_MODEL = _path("BERTSCORE_MODEL", "")

NEUTRALIZATION_API_KEY = ""
NEUTRALIZATION_API_URL = ""
NEUTRALIZATION_MODEL = ""
NEUTRALIZATION_INPUT = _path("NEUTRALIZATION_INPUT", "")
NEUTRALIZATION_OUTPUT = _path("NEUTRALIZATION_OUTPUT", "results/neutralized/modify_text/gpt4_rewrite_neutralized.json")
RECOVERY_INPUT = _path("RECOVERY_INPUT", NEUTRALIZATION_OUTPUT)
RECOVERY_OUTPUT = _path("RECOVERY_OUTPUT", "results/restored/modify_text/gpt4_rewrite_neutralized.json")

HUMAN_TRAIN_DIR = _path("HUMAN_TRAIN_DIR", "")
HUMAN_TEST_DIR = _path("HUMAN_TEST_DIR", "")
MACHINE_TRAIN_DIR = _path("MACHINE_TRAIN_DIR", "")
MACHINE_TEST_DIR = _path("MACHINE_TEST_DIR", "")
MODIFY_TRAIN_DIR = _path("MODIFY_TRAIN_DIR", "")
MODIFY_TEST_DIR = _path("MODIFY_TEST_DIR", "")

STYLE_CLASSIFIER_FULL_PATH = _path("STYLE_CLASSIFIER_FULL_PATH", "")
REDUCER_SAVE_PATH = _path("REDUCER_SAVE_PATH", "checkpoints/reducer.pt")
MAIN_CLASSIFIER_DIR = _path("MAIN_CLASSIFIER_DIR", "checkpoints/main")
OUTPUT_PROJECTION_PATH = _path("OUTPUT_PROJECTION_PATH", "checkpoints/projection.pt")
OUTPUT_MISTRAL_SAVE_DIR = _path("OUTPUT_MISTRAL_SAVE_DIR", "checkpoints/Mistral-7B-Instruct_finetune")

OUTPUT_FEATURE_FILE = _path("OUTPUT_FEATURE_FILE", "results/feature_training_data.pt")
REDUCED_FEATURE_MACHINE_PATH = _path("REDUCED_FEATURE_MACHINE_PATH", "results/feature_reduced_machine.pt")
OUTPUT_TEST_METRIC_MACHINE = _path("OUTPUT_TEST_METRIC_MACHINE", "results/metric_restore_machine.json")
