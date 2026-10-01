import argparse
import json
import os
from pathlib import Path

def build_prompt(text):
    return ('Simplify the text to make it clear and concise while preserving its meaning. '
            'Only output the simplified text without any explanations or additional comments.\n\n'
            f'"{text}"')

def input_text(row, field):
    if field == 'auto':
        field = next((k for k in row if k.endswith('output')), 'text')
    value = row[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Missing nonempty text in field {field}')
    return value

def main():
    from config import (NEUTRALIZATION_API_KEY, NEUTRALIZATION_API_URL,
                        NEUTRALIZATION_MODEL, NEUTRALIZATION_INPUT, NEUTRALIZATION_OUTPUT, require_path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=NEUTRALIZATION_INPUT)
    parser.add_argument('--output', type=Path, default=NEUTRALIZATION_OUTPUT)
    parser.add_argument('--field', default='auto', help='auto, text, HWT_sentence, or a revision field')
    parser.add_argument('--output-field', default='neutralized_text')
    parser.add_argument('--model', default=os.environ.get('MRTDF_NEUTRALIZER') or NEUTRALIZATION_MODEL)
    parser.add_argument('--api-url', default=os.environ.get('MRTDF_API_URL') or NEUTRALIZATION_API_URL)
    args = parser.parse_args()
    args.input = Path(require_path(args.input, 'NEUTRALIZATION_INPUT'))
    args.output = Path(require_path(args.output, 'NEUTRALIZATION_OUTPUT'))
    key = os.environ.get('MRTDF_API_KEY') or NEUTRALIZATION_API_KEY
    if not key or not args.model or not args.api_url:
        parser.error('Configure NEUTRALIZATION_API_KEY, NEUTRALIZATION_API_URL and NEUTRALIZATION_MODEL in config.py.')
    if args.output.exists() or args.input.resolve() == args.output.resolve():
        parser.error('Choose a new output file; existing files are not overwritten.')
    rows = json.loads(args.input.read_text(encoding='utf-8'))
    if not isinstance(rows, list):
        parser.error('Input must be a JSON list.')
    texts = [input_text(row, args.field) for row in rows]
    import requests
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i, (row, text) in enumerate(zip(rows, texts)):
        response = requests.post(args.api_url, headers={'Authorization': f'Bearer {key}'},
            json={'model': args.model, 'messages': [{'role': 'user', 'content': build_prompt(text)}],
                  'temperature': 0.3}, timeout=60)
        if not response.ok:
            raise RuntimeError(f'Request {i} failed: HTTP {response.status_code}; no automatic retries.')
        content = response.json()['choices'][0]['message']['content']
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f'Request {i} returned empty content.')
        row[args.output_field] = content.strip()
        args.output.write_text(json.dumps(rows[:i + 1], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'{i + 1}/{len(rows)}')

if __name__ == '__main__':
    main()
