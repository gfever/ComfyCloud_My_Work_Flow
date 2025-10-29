"""Run a notebook safely with papermill: ensure kernelspec exists and call papermill with kernel override.

Usage:
    python run_notebook_safe.py input.ipynb output.ipynb --kernel python3

The script will:
- load the input notebook JSON,
- if metadata.kernelspec is missing, inject one with the requested kernel name and display name,
- write a temporary notebook copy (not overwriting original),
- call papermill with -k/--kernel to execute the notebook, forwarding additional args.

This addresses the papermill error: "No kernel name found in notebook and no override provided." by ensuring both metadata and command-line override.
"""
import json
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
import argparse


def ensure_kernelspec(nb_path: Path, kernel_name: str = 'python3') -> Path:
    p = Path(nb_path)
    with p.open('r', encoding='utf-8') as f:
        nb = json.load(f)
    meta = nb.get('metadata', {})
    ks = meta.get('kernelspec')
    changed = False
    if not ks or not ks.get('name'):
        meta['kernelspec'] = {'name': kernel_name, 'display_name': 'Python 3'}
        nb['metadata'] = meta
        changed = True
    # also ensure language_info exists minimally
    if 'language_info' not in meta:
        meta['language_info'] = {'name': 'python', 'version': sys.version.split()[0]}
        nb['metadata'] = meta
        changed = True
    if changed:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix='.ipynb', prefix='nb_safe_')
        tf.close()
        with open(tf.name, 'w', encoding='utf-8') as out:
            json.dump(nb, out, ensure_ascii=False)
        return Path(tf.name)
    else:
        return p


def run_papermill(input_nb: Path, output_nb: Path, kernel_name: str, papermill_args: list):
    cmd = ['papermill', str(input_nb), str(output_nb), '-k', kernel_name] + papermill_args
    print('Running:', ' '.join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print('returncode:', proc.returncode)
    if proc.stdout:
        print('\n--- STDOUT ---\n', proc.stdout)
    if proc.stderr:
        print('\n--- STDERR ---\n', proc.stderr)
    return proc.returncode


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('input', help='Input notebook path')
    p.add_argument('output', help='Output notebook path')
    p.add_argument('--kernel', default='python3', help='Kernel name to ensure/use')
    p.add_argument('--', dest='papermill_args', nargs=argparse.REMAINDER, help='Additional papermill args to forward', default=None)
    args = p.parse_args(argv)

    input_nb = Path(args.input)
    output_nb = Path(args.output)
    kernel = args.kernel
    extra = args.papermill_args or []

    if not input_nb.exists():
        print('Input notebook not found:', input_nb)
        sys.exit(2)

    safe_nb = ensure_kernelspec(input_nb, kernel_name=kernel)
    try:
        rc = run_papermill(safe_nb, output_nb, kernel, extra)
    finally:
        # cleanup temp notebook if created
        if safe_nb != input_nb and safe_nb.exists():
            try:
                safe_nb.unlink()
            except Exception:
                pass
    sys.exit(rc)


if __name__ == '__main__':
    main()

