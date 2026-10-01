from pathlib import Path

import torch
from torch.utils.data import DataLoader
from utils.classifier import FeatureDataset, ScoringNetwork
from utils.training import (TRAINING_ROOT, parser_for, training_device, seed_all, new_output)


BATCH_SIZE = 64
EPOCHS = 60
LR = 1e-4
SEEDS = [41, 42, 43, 44, 45]


def main():
    parser = parser_for('main', 'Train task1 classifiers on HWT and MRT training features.',
                        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
    parser.add_argument('--features', type=Path, default=TRAINING_ROOT / 'task1_train.jsonl')
    parser.add_argument('--seeds', type=int, nargs='+')

    parser.set_defaults(seed=None)
    args = parser.parse_args()
    device = training_device(args)
    if args.seed is not None and args.seeds is not None:
        parser.error('Choose --seed or --seeds, not both.')
    seeds = ([args.seed] if args.seed is not None else args.seeds)
    if seeds is None:
        seeds = SEEDS
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error('Supply a nonempty list of unique seeds.')
    args.seeds = seeds
    dataset = FeatureDataset(str(args.features))
    labels = set()
    for feature, label in dataset:
        if feature.shape != (384,) or not torch.isfinite(feature).all() or label.item() not in (0, 1):
            parser.error('Expected finite 384-dimensional features with binary labels.')
        labels.add(label.item())
    if labels != {0., 1.}:
        parser.error('Task1 training requires both HWT (0) and MRT (1) samples.')
    output = new_output(args.output)
    for seed in seeds:
        seed_all(seed)
        model = ScoringNetwork().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        for epoch in range(args.epochs):
            total = 0.
            model.train()
            for features, targets in loader:
                loss = criterion(model(features.to(device)), targets.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite classifier loss.')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(features)
            print(f'seed={seed} epoch={epoch + 1} loss={total / len(dataset):.6f}')
        torch.save(model.cpu().state_dict(), output / f'task1_seed{seed}.pt')


if __name__ == '__main__':
    main()
