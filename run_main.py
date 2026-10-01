import argparse
import json
from pathlib import Path
import statistics


def evaluate(model, dataset, batch_size, device):
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score, f1_score

    labels, probabilities = [], []
    model.eval()
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            probabilities.extend(torch.sigmoid(model(x.to(device))).cpu().tolist())
            labels.extend(y.tolist())
    if len(set(labels)) != 2:
        raise ValueError('AUROC evaluation requires both classes.')
    return {'auc': float(roc_auc_score(labels, probabilities)),
            'f1': float(f1_score(labels, [p >= 0.5 for p in probabilities])), 'n': len(labels)}


def main():
    parser = argparse.ArgumentParser(description='Evaluate main-experiment classifier checkpoints.')
    parser.add_argument('--task', type=int, choices=(1,), default=1)
    parser.add_argument('--checkpoint', type=Path, nargs='+')
    parser.add_argument('--test-path', type=Path)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    import torch
    from config import DEVICE, ROOT, MAIN_CLASSIFIER_DIR
    from utils.classifier import FeatureDataset, ScoringNetwork

    if args.checkpoint is None:
        args.checkpoint = [Path(MAIN_CLASSIFIER_DIR) / 'task1_seed41.pt']
    if any(not p.is_file() for p in args.checkpoint):
        parser.error('Classifier weights are missing. Run task_result/task1/train_diff_classifier.py first, or supply --checkpoint.')
    if len({p.resolve() for p in args.checkpoint}) != len(args.checkpoint):
        parser.error('Checkpoint paths must be unique.')

    device = args.device or DEVICE
    batch_size = 64
    path = args.test_path or ROOT / 'results/subsets_task1'
    names = [f'{model}_{operation}' for model in ('2llama', 'gpt4')
             for operation in ('rewrite', 'complete', 'polish_token', 'polish_sentence')]
    paths = [path / f'style_cls_task1_test_{name}_neutralized.jsonl' for name in names]
    if any(not p.is_file() for p in paths):
        parser.error('Test features are missing. Run the corresponding extract_style_diff_features.py first.')
    tests = {p.stem: FeatureDataset(str(p)) for p in paths}
    records = []
    for checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model = ScoringNetwork().to(device)
        model.load_state_dict(state)
        metrics = {name: evaluate(model, ds, batch_size, device) for name, ds in tests.items()}
        record = {'checkpoint': str(checkpoint_path), 'subsets': metrics,
                  'mean_auc': statistics.mean(m['auc'] for m in metrics.values()),
                  'mean_f1': statistics.mean(m['f1'] for m in metrics.values())}
        records.append(record)
        print(json.dumps(record))
    if len(records) > 1:
        print(json.dumps({metric: {'mean': statistics.mean(r[metric] for r in records),
                                   'sample_std': statistics.stdev(r[metric] for r in records)}
                          for metric in ('mean_auc', 'mean_f1')}))


if __name__ == '__main__':
    main()
