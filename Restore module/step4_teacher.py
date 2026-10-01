from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from utils.projection import ProjectionNet
from utils.training import (TRAINING_ROOT, parser_for, training_device, seed_all, new_output,
                            load_features, masked_mean, model_dtype)
from utils.training_losses import variance_regularization_loss


BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-4
MAX_LEN = 512
VARIANCE_LOSS_WEIGHT = 0.1


def main():
    from config import Mistral_PATH, require_path
    parser = parser_for('fam', 'Train FAM with a frozen Mistral backbone and fixed SDM targets.',
                        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
    parser.add_argument('--features', type=Path, default=TRAINING_ROOT / 'sdm/reduced_features.pt')
    parser.add_argument('--model', default=Mistral_PATH)
    parser.add_argument('--max-len', type=int, default=MAX_LEN)
    args = parser.parse_args()
    args.model = require_path(args.model, 'Mistral_PATH')
    device = training_device(args)
    rows = load_features(args.features, reduced=True)
    output = new_output(args.output)
    seed_all(args.seed)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    backbone = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=model_dtype(device)).to(device)
    backbone.requires_grad_(False).eval()

    cache = {}
    with torch.no_grad():
        for row in rows:
            target = row['target']
            if target not in cache:
                inputs = tokenizer(target, return_tensors='pt', truncation=True,
                                   max_length=args.max_len).to(device)
                inputs.pop('token_type_ids', None)
                outputs = backbone(**inputs, output_hidden_states=True, use_cache=False)
                cache[target] = masked_mean(outputs.hidden_states[-1], inputs['attention_mask'])[0].float().cpu()
    hidden_dim = backbone.config.hidden_size
    del outputs, backbone
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    dataset = TensorDataset(torch.stack([cache[r['target']] for r in rows]),
                            torch.stack([r['style_tgt_reduced'].float() for r in rows]))
    model = ProjectionNet(input_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        total = 0.
        model.train()
        for hidden, target in loader:
            pred = model(hidden.to(device))
            loss = 1 - F.cosine_similarity(pred, target.to(device)).mean()
            loss = loss + VARIANCE_LOSS_WEIGHT * variance_regularization_loss(pred)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite FAM loss.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(hidden)
        print(f'epoch={epoch + 1} loss={total / len(dataset):.6f}')
    torch.save(model.cpu().state_dict(), output / 'projection.pt')


if __name__ == '__main__':
    main()
