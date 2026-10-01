import argparse
import os
from pathlib import Path
import runpy
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('script', help='Python script relative to this directory')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    script = (root / args.script).resolve()
    if not script.is_relative_to(root) or not script.is_file() or script.suffix != '.py':
        parser.error('Choose an existing Python script inside this release directory.')
    os.chdir(root)
    sys.path.insert(0, str(root))
    if '--help' not in args.args and '-h' not in args.args:
        for directory in ('results', 'checkpoints', 'image', 'ckpt'):
            (root / directory).mkdir(exist_ok=True)
    sys.argv = [str(script), *args.args]
    runpy.run_path(str(script), run_name='__main__')

if __name__ == '__main__':
    main()
