import os
import json
import random
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from transformers import DebertaV2Tokenizer, DebertaV2Model

from config import DEVICE, MODIFY_TRAIN_DIR, DEBERTA_PATH, REDUCER_SAVE_PATH, require_path
from utils.projection import Reducer


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


TSNE_DATA_DIR = os.environ.get('MRTDF_TSNE_DATA_DIR', MODIFY_TRAIN_DIR)
TSNE_DATA_DIR = require_path(TSNE_DATA_DIR, 'MODIFY_TRAIN_DIR (or MRTDF_TSNE_DATA_DIR)')
if not os.path.isdir(TSNE_DATA_DIR):
    raise FileNotFoundError('Set MRTDF_TSNE_DATA_DIR to paired text data with neutralized_hwt and neutralized_text.')
require_path(DEBERTA_PATH, 'DEBERTA_PATH')
tokenizer = DebertaV2Tokenizer.from_pretrained(DEBERTA_PATH)
deberta = DebertaV2Model.from_pretrained(DEBERTA_PATH).to(DEVICE).eval()

reducer = Reducer()
reducer.load_state_dict(torch.load(REDUCER_SAVE_PATH, map_location=DEVICE))
reducer.to(DEVICE).eval()

os.makedirs("image", exist_ok=True)


@torch.no_grad()
def extract_features(text: str) -> np.ndarray:

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    outputs = deberta(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden = outputs.last_hidden_state

    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    feat = (summed / denom).squeeze(0).cpu().numpy()
    return feat


@torch.no_grad()
def map_to_style(feat_768: np.ndarray) -> np.ndarray:
    x = torch.tensor(feat_768, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    z = reducer(x).squeeze(0).detach().cpu().numpy()
    return z


def get_output_key(item: dict):
    return next((k for k in item.keys() if k.endswith("output")), None)

def collect_pairs(max_pairs: int = 1000, seed: int = 42):

    random.seed(seed)
    pairs = []

    files = [fn for fn in os.listdir(TSNE_DATA_DIR) if fn.endswith(".json")]
    files.sort()

    for fname in files:
        fp = os.path.join(TSNE_DATA_DIR, fname)
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
                "hwt": hwt,
                "mrt": mrt,
                "snt_hwt": snt_hwt,
                "snt_mrt": snt_mrt
            })

    random.shuffle(pairs)
    return pairs[:max_pairs]


def build_style_vectors_from_pairs(pairs, mode: str):

    texts = []
    labels = []

    for p in pairs:
        if mode == "original":
            h_text, m_text = p["hwt"], p["mrt"]
        elif mode == "neutralized":
            h_text, m_text = p["snt_hwt"], p["snt_mrt"]
        else:
            raise ValueError(mode)

        texts.append(h_text); labels.append(0)
        texts.append(m_text); labels.append(1)

    labels = np.array(labels, dtype=np.int64)

    Z = []
    for t in tqdm(texts, desc=f"Extract+Map({mode})"):
        feat = extract_features(t)
        z = map_to_style(feat)
        Z.append(z)
    Z = np.stack(Z, axis=0)
    return Z, labels


def run_tsne(Z: np.ndarray, seed: int = 42):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        n_iter=2000
    )
    return tsne.fit_transform(Z)

def plot_tsne_side_by_side(
    Y_left: np.ndarray,  labels_left: np.ndarray,  title_left: str,
    Y_right: np.ndarray, labels_right: np.ndarray, title_right: str,
    save_path_pdf: str,
    left_names=("HWT_original", "MRT_original"),
    right_names=("HWT_neutralized", "MRT_neutralized"),
):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    def _scatter(ax, Y, labels, title, names):

        for lb, name in [(0, names[0]), (1, names[1])]:
            idx = (labels == lb)
            ax.scatter(Y[idx, 0], Y[idx, 1], s=8, alpha=0.7, label=name)

        ax.grid(True)
        ax.legend(frameon=True)

    _scatter(axes[0], Y_left,  labels_left,  title_left,  left_names)
    _scatter(axes[1], Y_right, labels_right, title_right, right_names)

    fig.savefig(save_path_pdf, format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main():

    pairs = collect_pairs(max_pairs=1000, seed=42)
    print(f"✅ Collected {len(pairs)} pairs for t-SNE.")


    Z0, y0 = build_style_vectors_from_pairs(pairs, mode="original")
    Y0 = run_tsne(Z0, seed=42)


    Z1, y1 = build_style_vectors_from_pairs(pairs, mode="neutralized")
    Y1 = run_tsne(Z1, seed=42)


    plot_tsne_side_by_side(
        Y0, y0, "Original",
        Y1, y1, "Neutralized",
        save_path_pdf="image/tsne_style_side_by_side.pdf",
        left_names=("HWT_original", "MRT_original"),
        right_names=("SNT(HWT)", "SNT(MRT)"),
    )


if __name__ == "__main__":
    main()
