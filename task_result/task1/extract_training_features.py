import argparse
import json
from pathlib import Path

import torch
from utils.projection import Reducer
from utils.training import TRAINING_ROOT


def main():
    from config import DEVICE, HUMAN_TRAIN_DIR, MODIFY_TRAIN_DIR, REDUCER_SAVE_PATH, require_path
    parser = argparse.ArgumentParser(description='Extract task1 features from restored training texts only.')
    parser.add_argument('--human-dir', type=Path, default=HUMAN_TRAIN_DIR)
    parser.add_argument('--mrt-dir', type=Path, default=MODIFY_TRAIN_DIR)
    parser.add_argument('--reducer', type=Path, default=REDUCER_SAVE_PATH)
    parser.add_argument('--output', type=Path, default=TRAINING_ROOT / 'task1_train.jsonl')
    args = parser.parse_args()
    args.human_dir = Path(require_path(args.human_dir, 'HUMAN_TRAIN_DIR'))
    args.mrt_dir = Path(require_path(args.mrt_dir, 'MODIFY_TRAIN_DIR'))
    args.reducer = Path(require_path(args.reducer, 'REDUCER_SAVE_PATH'))
    if args.output.exists():
        parser.error('Output already exists; choose another --output.')
    examples = []
    for directory, label in ((args.human_dir, 0), (args.mrt_dir, 1)):
        paths = sorted(directory.glob('*.json'))
        if not paths:
            parser.error(f'No training JSON files found in {directory}.')
        count = 0
        for path in paths:
            rows = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(rows, list):
                raise ValueError(f'{path}: expected a JSON array.')
            for row in rows:
                if label == 0:
                    original = row.get('text')
                else:
                    keys = [k for k in row if k.endswith('output')]
                    if len(keys) != 1:
                        raise ValueError(f'{path}: expected exactly one MRT output field.')
                    original = row[keys[0]]
                restored = row.get('restore_text')
                if any(not isinstance(t, str) or not t.strip() for t in (original, restored)):
                    raise ValueError(f'{path}: missing original text or restore_text.')
                examples.append((original, restored, label))
                count += 1
        if count == 0:
            parser.error(f'No training samples in {directory}.')
    from utils.feature_extractors import get_pure_style_embedding
    reducer = Reducer().to(DEVICE)
    reducer.load_state_dict(torch.load(args.reducer, map_location=DEVICE, weights_only=True))
    reducer.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream, torch.no_grad():
        for original, restored, label in examples:
            a = reducer(get_pure_style_embedding(original).to(DEVICE))
            b = reducer(get_pure_style_embedding(restored).to(DEVICE))
            feature = torch.cat([a, b, a - b]).cpu().tolist()
            stream.write(json.dumps({'feature': feature, 'label': label}) + '\n')
    print(f'Saved {len(examples)} task1 training features to {args.output}')


if __name__ == '__main__':
    main()
