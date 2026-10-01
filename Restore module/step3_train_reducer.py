from pathlib import Path

import torch
from torch.utils.data import DataLoader
from utils.projection import Reducer
from utils.training import (TRAINING_ROOT, parser_for, training_device, seed_all,
                            new_output, load_features)
from utils.training_losses import (style_triplet_consistency_loss,
                                   variance_regularization_loss, pullaway_loss)


BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
MARGIN_NEUTRAL = 0.1
MARGIN_TARGET = 0.4
MARGIN_HUMAN = 0.3
VAR_LOSS_WEIGHT = 0.05
PT_LOSS_WEIGHT = 0.05
POOL_SIZE = 10


def main():
    parser = parser_for('sdm', 'Train the SDM using STC-Loss.',
                        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
    parser.add_argument('--features', type=Path, default=TRAINING_ROOT / 'paired_features.pt')
    args = parser.parse_args()
    device = training_device(args)
    rows = load_features(args.features)
    if len(rows) < 2 or args.batch_size < 2:
        parser.error('SDM training requires at least two samples and batch_size >= 2.')
    output = new_output(args.output)
    seed_all(args.seed)
    model = Reducer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    singleton = len(rows) % args.batch_size == 1
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, drop_last=singleton)
    if singleton:
        print('The final singleton batch is omitted during SDM optimization.')
    for epoch in range(args.epochs):
        total, count = 0., 0
        model.train()
        for batch in loader:
            src = model(batch['style_src'].float().to(device))
            tgt = model(batch['style_tgt'].float().to(device))
            loss = style_triplet_consistency_loss(
                src, tgt, batch['target_id'], MARGIN_NEUTRAL,
                MARGIN_TARGET, MARGIN_HUMAN, POOL_SIZE)
            loss = loss + VAR_LOSS_WEIGHT * variance_regularization_loss(src)
            loss = loss + PT_LOSS_WEIGHT * pullaway_loss(src)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite SDM loss.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(src)
            count += len(src)
        print(f'epoch={epoch + 1} loss={total / count:.6f}')
    model.eval()
    with torch.no_grad():
        for row in rows:
            row['style_src_reduced'] = model(row['style_src'].float().to(device)).cpu()
            row['style_tgt_reduced'] = model(row['style_tgt'].float().to(device)).cpu()
    torch.save(model.cpu().state_dict(), output / 'reducer.pt')
    torch.save(rows, output / 'reduced_features.pt')


if __name__ == '__main__':
    main()
