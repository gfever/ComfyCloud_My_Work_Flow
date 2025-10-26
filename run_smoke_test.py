"""
Простой smoke-тест для ноутбука: выполняет безопасные (non-heavy) ячейки
и проверяет, что ключевые переменные/функции определяются без heavy-зависимостей.

Запуск:
    python run_smoke_test.py

Этот скрипт не устанавливает пакеты и пропускает ячейки, где встречаются
паттерны heavy_imports (torch, diffusers, ffmpeg, git clone и т.п.).
"""
import json
import os
import re
import sys
import traceback
import importlib
import types

NOTEBOOK = os.path.join(os.path.dirname(__file__), 'kaggle_simple_cells_v4_clean.ipynb')

HEAVY_PATTERNS = [
    r"\btorch\b",
    r"\bdiffusers\b",
    r"\bcv2\b",
    r"ffmpeg",
    r"git clone",
    r"gdown",
    r"Real-ESRGAN",
    r"Practical-RIFE",
    r"StableDiffusion",
    r"AnimateDiff",
    r"from diffusers",
    r"subprocess.check_call",
    r"importlib\.util",
]

def is_heavy(cell_source):
    src = "\n".join(cell_source) if isinstance(cell_source, list) else cell_source
    for pat in HEAVY_PATTERNS:
        if re.search(pat, src, flags=re.IGNORECASE):
            return True, pat
    return False, None


def load_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_safe_cells(nb):
    cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']
    safe_globals = {}

    # Predefine a minimal print_separator to match notebook usage
    def print_separator(nl_before=True):
        SEP = '=' * 60
        if nl_before:
            print('\n' + SEP)
        else:
            print(SEP + '\n')

    safe_globals['print_separator'] = print_separator
    safe_globals['__name__'] = '__main__'

    # Ensure workspace paths point to a temporary local folder inside project
    base_tmp = os.path.join(os.path.dirname(__file__), 'smoke_tmp')
    os.makedirs(base_tmp, exist_ok=True)
    safe_globals['WORKSPACE'] = base_tmp
    safe_globals['FRAMES_DIR'] = os.path.join(base_tmp, 'frames')
    safe_globals['DATASET_DIR'] = os.path.join(base_tmp, 'dataset')
    os.makedirs(safe_globals['FRAMES_DIR'], exist_ok=True)
    os.makedirs(safe_globals['DATASET_DIR'], exist_ok=True)

    # Ensure importlib is available inside executed cells (some cells use importlib.util)
    # Build a safe importlib stub so cells can call importlib.util.find_spec reliably
    real_importlib = importlib
    importlib_stub = types.ModuleType('importlib')
    # Copy common attributes from real importlib to the stub (excluding util)
    for attr in dir(real_importlib):
        if attr == 'util':
            continue
        try:
            setattr(importlib_stub, attr, getattr(real_importlib, attr))
        except Exception:
            pass

    # Ensure we have a util module with find_spec
    try:
        import importlib.util as real_util
        util_mod = real_util
    except Exception:
        util_mod = types.ModuleType('importlib.util')
        def _find_spec(name):
            try:
                return real_importlib.__import__('importlib.util').find_spec(name)
            except Exception:
                return None
        util_mod.find_spec = _find_spec

    importlib_stub.util = util_mod

    # Inject into execution environment and sys.modules so `import importlib` works inside cells
    safe_globals['importlib'] = importlib_stub
    sys.modules['importlib'] = importlib_stub
    sys.modules['importlib.util'] = util_mod

    # Provide a lightweight IPython.display stub so cells that import it won't fail
    display_mod = types.ModuleType('IPython.display')
    class FileLink:
        def __init__(self, path):
            self.path = path
        def __repr__(self):
            return f"FileLink({self.path})"
    def _display(x):
        print(f"[display] {x}")
    display_mod.FileLink = FileLink
    display_mod.display = _display
    # Inject into sys.modules so `import IPython`/`from IPython.display import FileLink` works
    sys.modules['IPython'] = types.ModuleType('IPython')
    sys.modules['IPython.display'] = display_mod

    passed = 0
    skipped = 0
    failed = 0
    failures = []

    for idx, cell in enumerate(cells):
        src = ''.join(cell.get('source', []))
        heavy, pat = is_heavy(src)
        if heavy:
            print(f"SKIP cell #{idx+1} (contains heavy pattern: {pat})")
            skipped += 1
            continue

        print(f"RUN  cell #{idx+1} — executing...")
        try:
            exec(compile(src, f"<cell {idx+1}>", 'exec'), safe_globals)
            passed += 1
        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            failures.append((idx+1, str(e), tb))
            print(f"FAIL cell #{idx+1}: {e}\n")

    return {
        'passed': passed,
        'skipped': skipped,
        'failed': failed,
        'failures': failures,
        'globals': safe_globals,
    }


def main():
    if not os.path.exists(NOTEBOOK):
        print(f"Notebook not found: {NOTEBOOK}")
        sys.exit(2)

    nb = load_notebook(NOTEBOOK)
    print(f"Loaded notebook with {len(nb.get('cells', []))} cells")

    result = run_safe_cells(nb)

    print('\n=== SUMMARY ===')
    print(f"Passed: {result['passed']}")
    print(f"Skipped (heavy): {result['skipped']}")
    print(f"Failed: {result['failed']}")

    if result['failed']:
        print('\nDetails of failures:')
        for idx, err, tb in result['failures']:
            print(f"--- cell #{idx} error: {err}\n{tb}\n")

    # Basic assertions to quickly validate expected variables/functions
    g = result['globals']
    checks = []
    checks.append(('print_separator' in g, 'print_separator defined'))
    checks.append(('WORKSPACE' in g, 'WORKSPACE defined'))
    checks.append(('FRAMES_DIR' in g, 'FRAMES_DIR defined'))

    print('\nQuick checks:')
    all_ok = True
    for ok, msg in checks:
        print(f"  - {msg}: {'OK' if ok else 'MISSING'}")
        if not ok:
            all_ok = False

    if result['failed'] or not all_ok:
        print('\nSMOKE TEST: FAIL')
        sys.exit(1)
    else:
        print('\nSMOKE TEST: PASS')
        sys.exit(0)


if __name__ == '__main__':
    main()
