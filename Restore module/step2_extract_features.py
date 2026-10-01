import argparse
import hashlib
import json
from pathlib import Path

import torch
from utils.training import ROOT, TRAINING_ROOT


def main():
    parser = argparse.ArgumentParser(description='Extract frozen style features for paired training texts.')
    parser.add_argument('--input', type=Path, default=ROOT / 'data/training/neutralized_pairs.json')
    parser.add_argument('--output', type=Path, default=TRAINING_ROOT / 'paired_features.pt')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose another --output.')
    paths = sorted(args.input.glob('*.json')) if args.input.is_dir() else [args.input]
    pairs = []
    for path in paths:
        rows = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(rows, list):
            raise ValueError(f'{path}: expected a JSON array.')
        for row in rows:
            source, target = row.get('neutralized_text'), row.get('HWT_sentence')
            if any(not isinstance(t, str) or not t.strip() for t in (source, target)):
                raise ValueError(f'{path}: each record needs neutralized_text and HWT_sentence.')
            pairs.append((source, target))
    if not pairs:
        parser.error('No training pairs found.')
    from utils.feature_extractors import get_pure_style_embedding
    features = []
    with torch.no_grad():
        for source, target in pairs:
            features.append({'text': source, 'target': target,
                             'target_id': hashlib.sha1(target.encode('utf-8')).hexdigest(),
                             'style_src': get_pure_style_embedding(source).detach().cpu().float(),
                             'style_tgt': get_pure_style_embedding(target).detach().cpu().float()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, args.output)
    print(f'Saved {len(features)} paired features to {args.output}')


if __name__ == '__main__':
    main()
