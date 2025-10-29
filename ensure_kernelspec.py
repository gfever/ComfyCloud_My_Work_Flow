"""Ensure a notebook has metadata.kernelspec.name set (e.g., 'python3').
Usage:
    python ensure_kernelspec.py path/to/notebook.ipynb
This updates the notebook in-place (a backup is created with .bak).
"""
import sys
import json
from pathlib import Path

def ensure_nb_kernelspec(path: Path, kernel_name: str = 'python3', display_name: str = 'Python 3') -> bool:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding='utf-8')
    nb = json.loads(text)
    meta = nb.get('metadata', {})
    ks = meta.get('kernelspec')
    changed = False
    if not ks or not ks.get('name'):
        meta['kernelspec'] = {'name': kernel_name, 'display_name': display_name}
        nb['metadata'] = meta
        changed = True
    if 'language_info' not in meta:
        meta['language_info'] = {'name': 'python', 'version': sys.version.split()[0]}
        nb['metadata'] = meta
        changed = True
    if changed:
        backup = path.with_suffix(path.suffix + '.bak')
        backup.write_text(text, encoding='utf-8')
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    return changed

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python ensure_kernelspec.py notebook.ipynb')
        sys.exit(2)
    p = Path(sys.argv[1])
    try:
        changed = ensure_nb_kernelspec(p)
        if changed:
            print('Updated kernelspec in', p)
            print('Backup of original saved as', str(p) + '.bak')
        else:
            print('No changes (kernelspec already present) for', p)
    except Exception as e:
        print('Error:', e)
        sys.exit(1)

