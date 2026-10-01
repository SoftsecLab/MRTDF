import argparse
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / 'results' / 'training'


def parser_for(stage, description, *, epochs, batch_size, lr):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument('--batch-size', type=int, default=batch_size)
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--output', type=Path, default=TRAINING_ROOT / stage)
    return parser


def training_device(args):
    from config import DEVICE
    for key in ('epochs', 'batch_size', 'lr'):
        value = getattr(args, key)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite.')
    return args.device or DEVICE


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def new_output(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}; choose a new --output path.')
    path.mkdir(parents=True)
    return path


def load_features(path, reduced=False):
    rows = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty list of paired training features.')
    required = {'text', 'target', 'target_id', 'style_src', 'style_tgt'}
    if reduced:
        required.add('style_tgt_reduced')
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f'Each feature record must contain {sorted(required)}.')
        for name, dim in [('style_src', 768), ('style_tgt', 768)]:
            if row[name].shape != (dim,) or not torch.isfinite(row[name]).all():
                raise ValueError(f'{name} must be a finite {dim}-dimensional tensor.')
        if reduced and (row['style_tgt_reduced'].shape != (128,) or
                        not torch.isfinite(row['style_tgt_reduced']).all()):
            raise ValueError('style_tgt_reduced must be a finite 128-dimensional tensor.')
        if any(not isinstance(row[name], str) or not row[name].strip()
               for name in ('text', 'target', 'target_id')):
            raise ValueError('Training texts and target IDs must be nonempty strings.')
    return rows


def masked_mean(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


def sfsp_weight(step, total_steps):
    if total_steps <= 0 or not 0 <= step <= total_steps:
        raise ValueError('Expected 0 <= step <= total_steps, with total_steps > 0.')
    return step / total_steps


def recovery_example(tokenizer, source, target, max_len):
    prompt = ('### Instruction: Rewrite the following sentence in a natural, human-like style.\n'
              f'Input: {source}\nOutput:')
    prefix = tokenizer(prompt, add_special_tokens=True)['input_ids']
    answer = tokenizer(target, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
    if len(prefix) >= max_len:
        raise ValueError('Recovery prompt exceeds max_len; increase max_len or shorten the input.')
    answer = answer[:max_len - len(prefix)]
    ids = prefix + answer
    labels = [-100] * len(prefix) + answer
    padding = max_len - len(ids)
    return {'input_ids': torch.tensor(ids + [tokenizer.pad_token_id] * padding),
            'attention_mask': torch.tensor([1] * len(ids) + [0] * padding),
            'labels': torch.tensor(labels + [-100] * padding)}


def model_dtype(device):
    if str(device).startswith('cuda'):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32
