from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.projection import ProjectionNet
from utils.training import (TRAINING_ROOT, parser_for, training_device, seed_all, new_output,
                            load_features, masked_mean, sfsp_weight,
                            recovery_example, model_dtype)


BATCH_SIZE = 4
EPOCHS = 3
LR = 1e-4
MAX_LEN = 512
PROJECTION_DIM = 128
LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']


def main():
    from config import Mistral_PATH, require_path
    parser = parser_for('recovery', 'Fine-tune the style recovery model using LoRA and SFSP.',
                        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
    parser.add_argument('--features', type=Path, default=TRAINING_ROOT / 'sdm/reduced_features.pt')
    parser.add_argument('--projection', type=Path, default=TRAINING_ROOT / 'fam/projection.pt')
    parser.add_argument('--model', default=Mistral_PATH)
    args = parser.parse_args()
    args.model = require_path(args.model, 'Mistral_PATH')
    device = training_device(args)
    rows = load_features(args.features, reduced=True)
    projection_state = torch.load(args.projection, map_location='cpu', weights_only=True)
    if args.output.exists():
        parser.error('Output already exists; choose another --output.')
    seed_all(args.seed)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    examples = []
    for row in rows:
        item = recovery_example(tokenizer, row['text'], row['target'], MAX_LEN)
        item['style_target'] = row['style_tgt_reduced'].float()
        examples.append(item)
    output = new_output(args.output)
    dtype = model_dtype(device)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    projection = ProjectionNet(input_dim=base.config.hidden_size, output_dim=PROJECTION_DIM).to(device)
    projection.load_state_dict(projection_state)
    projection.requires_grad_(False).eval()
    model = get_peft_model(base, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias='none', task_type='CAUSAL_LM'))

    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True)
    total_steps = args.epochs * len(loader)
    step = 0
    for epoch in range(args.epochs):
        model.train()
        total = 0.
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            target = batch.pop('style_target')
            optimizer.zero_grad(set_to_none=True)
            result = model(**batch, output_hidden_states=True, use_cache=False)
            hidden = masked_mean(result.hidden_states[-1], batch['attention_mask']).float()
            style_loss = 1 - F.cosine_similarity(projection(hidden), target).mean()
            loss = result.loss + sfsp_weight(step, total_steps) * style_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite recovery loss.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            total += loss.item() * len(target)
        print(f'epoch={epoch + 1} loss={total / len(examples):.6f}')
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)


if __name__ == '__main__':
    main()
