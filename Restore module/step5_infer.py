import argparse
import json
from pathlib import Path

INSTRUCTION = '### Instruction: Rewrite the following sentence in a natural, human-like style.\nInput: {}\nOutput:'

def main():
    from config import RECOVERY_INPUT, RECOVERY_OUTPUT, require_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=RECOVERY_INPUT)
    parser.add_argument('--output', type=Path, default=RECOVERY_OUTPUT)
    args = parser.parse_args()
    args.input = Path(require_path(args.input, 'RECOVERY_INPUT'))
    args.output = Path(require_path(args.output, 'RECOVERY_OUTPUT'))
    if args.output.exists() or args.input.resolve() == args.output.resolve():
        parser.error('Choose a new output file.')
    rows = json.loads(args.input.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or any(not isinstance(r.get('neutralized_text'), str) for r in rows):
        parser.error('Expected JSON records with neutralized_text.')
    import torch
    from transformers import AutoTokenizer
    from config import DEVICE, OUTPUT_MISTRAL_SAVE_DIR
    from utils.recovery import load_recovery_model
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_MISTRAL_SAVE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if str(DEVICE).startswith('cuda') else torch.float32
    model = load_recovery_model(torch_dtype=dtype).to(DEVICE).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        ids = tokenizer(INSTRUCTION.format(row['neutralized_text']), return_tensors='pt').input_ids.to(DEVICE)
        with torch.no_grad():
            output = model.generate(input_ids=ids, max_new_tokens=256, do_sample=True,
                temperature=0.85, top_p=0.92, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        row['restore_text'] = decoded.split('Output:')[-1].strip() if 'Output:' in decoded else decoded.strip()
        args.output.write_text(json.dumps(rows[:i + 1], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'{i + 1}/{len(rows)}')

if __name__ == '__main__':
    main()
