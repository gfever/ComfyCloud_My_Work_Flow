"""
Упаковочный скрипт для Kaggle dataset.
Собирает указанные файлы в папку и вызывает `kaggle datasets create -p <folder>`.
Требуется установленный kaggle CLI и авторизация (kaggle.json).

Пример использования:
    python create_kaggle_dataset.py --folder-name comfy-gojo-dataset

Этот скрипт НЕ перезатирает существующий датасет и запросит подтверждение.
"""
import os
import shutil
import argparse
import subprocess

DEFAULT_FILES = [
    'kaggle_simple_cells_v4_clean.ipynb',
    'prompts.json'
]

parser = argparse.ArgumentParser()
parser.add_argument('--folder-name', default='comfy-gojo-dataset', help='Target folder name to create for the dataset')
parser.add_argument('--files', nargs='*', default=DEFAULT_FILES, help='Files to include in the dataset folder')
parser.add_argument('--no-upload', action='store_true', help='Only prepare folder locally, do not call kaggle CLI')
args = parser.parse_args()

cwd = os.path.dirname(__file__)
folder = os.path.join(cwd, args.folder_name)
if os.path.exists(folder):
    print(f'Folder {folder} already exists. Remove or choose a different --folder-name.')
    exit(1)
os.makedirs(folder, exist_ok=True)

copied = []
for f in args.files:
    src = os.path.join(cwd, f)
    if os.path.exists(src):
        shutil.copy(src, folder)
        copied.append(f)
        print(f'Copied {f} -> {folder}')
    else:
        print(f'Warning: {f} not found in repository root; skipping')

print(f"\nPrepared folder: {folder}")
if copied:
    print('Files copied:', ', '.join(copied))
else:
    print('No files were copied.')

if args.no_upload:
    print('\n--no-upload specified, skipping kaggle CLI upload.')
    print('You can zip the folder and upload via Kaggle web UI or run the kaggle CLI locally:')
    print(f'  kaggle datasets create -p "{folder}"')
    exit(0)

# Create dataset via kaggle CLI
print('\nNow calling `kaggle datasets create` (requires kaggle CLI and authentication)')
try:
    subprocess.check_call(['kaggle', 'datasets', 'create', '-p', folder])
except Exception as e:
    print('Failed to create dataset via kaggle CLI:', e)
    print('You can manually zip the folder and upload via Kaggle web UI.')

print('\nDone.')
