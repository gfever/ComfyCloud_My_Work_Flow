"""Prepare (install deps + write shim) and optionally run Real-ESRGAN inference in one step.

Usage examples (in Kaggle notebook cell or terminal):

# Just write shim and check syntax
python prepare_and_run_realesrgan.py --shim /kaggle/working/Real-ESRGAN/inference_with_shim_full.py

# Install deps, write shim and run inference with args passed to the shim
python prepare_and_run_realesrgan.py --install --run --shim /kaggle/working/Real-ESRGAN/inference_with_shim_full.py -- -- -n 4x-UltraSharp -i /kaggle/working/in -o /kaggle/working/out --fp32

Notes:
- Use "--" to separate this script's args from the args that should be forwarded to the shim/inference script.
- This script uses the current Python interpreter's pip to install packages so they end up in the same env.
"""
import argparse
import os
import sys
import textwrap
import subprocess
import py_compile

DEFAULT_DEPS = ["basicsr", "facexlib", "gfpgan", "realesrgan"]

SHIM = textwrap.dedent(r"""
import sys
import os
import types
import re
import runpy

# Compatibility shim for torchvision.transforms.functional_tensor and torchvision.utils
try:
    import torchvision.transforms.functional_tensor as _ft
except Exception:
    try:
        import torchvision.transforms.functional as _f
        mod_ft = types.ModuleType('torchvision.transforms.functional_tensor')
        mod_ft.rgb_to_grayscale = getattr(_f, 'rgb_to_grayscale', None)
        mod_ft.convert_image_dtype = getattr(_f, 'convert_image_dtype', None)
        sys.modules['torchvision.transforms.functional_tensor'] = mod_ft
    except Exception:
        mod_ft = types.ModuleType('torchvision.transforms.functional_tensor')
        def _rgb_to_grayscale(x):
            raise ImportError('rgb_to_grayscale not available')
        def _convert_image_dtype(x, dtype):
            raise ImportError('convert_image_dtype not available')
        mod_ft.rgb_to_grayscale = _rgb_to_grayscale
        mod_ft.convert_image_dtype = _convert_image_dtype
        sys.modules['torchvision.transforms.functional_tensor'] = mod_ft

# Minimal torchvision.utils.make_grid fallback
if 'torchvision.utils' not in sys.modules:
    mod_utils = types.ModuleType('torchvision.utils')
    def make_grid(x, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
        try:
            if isinstance(x, (list, tuple)) and len(x) > 0:
                return x[0]
            return x
        except Exception:
            return x
    mod_utils.make_grid = make_grid
    sys.modules['torchvision.utils'] = mod_utils

# Provide a minimal realesrgan.version module if missing
if 'realesrgan.version' not in sys.modules:
    vermod = types.ModuleType('realesrgan.version')
    vermod.__version__ = '0.3.0'
    vermod.__all__ = ['__version__']
    sys.modules['realesrgan.version'] = vermod

# Helper to infer numeric scale (netscale) from argv
def _infer_netscale(argv):
    for i, a in enumerate(argv):
        if a in ('-n', '--name', '--model') and i+1 < len(argv):
            m = re.match(r"(\d+)x", argv[i+1])
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
    for i, a in enumerate(argv):
        if a in ('-s', '--scale') and i+1 < len(argv):
            try:
                return int(argv[i+1])
            except Exception:
                pass
    return 4

# Try to detect a reasonable default model file inside ./weights/
def _detect_weight_file():
    try:
        wdir = os.path.join(os.getcwd(), 'weights')
        if not os.path.isdir(wdir):
            return None
        exts = ('.pth', '.pt', '.safetensors')
        candidates = [f for f in os.listdir(wdir) if f.lower().endswith(exts)]
        if not candidates:
            return None
        for name in candidates:
            if '4x' in name.lower() or 'ultrasharp' in name.lower():
                return os.path.join(wdir, name)
        return os.path.join(wdir, candidates[0])
    except Exception:
        return None

# Forward to the original inference_realesrgan.py but ensure safe globals are defined
if __name__ == '__main__':
    argv = sys.argv[1:]
    netscale = _infer_netscale(argv)
    detected_model = _detect_weight_file()
    script_path = os.path.join(os.getcwd(), 'inference_realesrgan.py')
    try:
        with open(script_path, 'r', encoding='utf-8') as _f:
            _code = _f.read()
    except Exception:
        sys.argv = [script_path] + argv
        runpy.run_path('inference_realesrgan.py', run_name='__main__')
    else:
        _globals = {
            '__name__': '__main__',
            '__file__': script_path,
            'netscale': netscale,
            'model': detected_model,
            'outscale': netscale,
            'scale': netscale,
        }
        _globals['sys'] = sys
        sys.argv = [script_path] + argv
        exec(compile(_code, script_path, 'exec'), _globals)
""")


def write_shim(shim_path: str):
    shim_path = os.fspath(shim_path)
    d = os.path.dirname(shim_path)
    # Try to create target directory; if not allowed (PermissionError), fall back to cwd
    try:
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
    except PermissionError:
        fallback = os.path.join(os.getcwd(), os.path.basename(shim_path))
        print(f"Warning: cannot create directory {d} (Permission denied). Falling back to: {fallback}")
        shim_path = fallback
    try:
        with open(shim_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(SHIM)
        return shim_path
    except PermissionError:
        # Final fallback: write a temp file in cwd
        import tempfile
        tf = tempfile.NamedTemporaryFile(delete=False, suffix='.py', prefix='inference_shim_', dir=os.getcwd())
        tf.write(SHIM.encode('utf-8'))
        tf.close()
        print(f"Warning: cannot write to {shim_path} (Permission denied). Wrote shim to temporary file: {tf.name}")
        return tf.name


def install_packages(packages):
    if not packages:
        return
    print('Installing packages:', packages)
    cmd = [sys.executable, '-m', 'pip', 'install', '-q'] + packages
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print('pip install failed with returncode', e.returncode)
        raise


def check_compile(path):
    try:
        py_compile.compile(path, doraise=True)
        print('Compilation ok:', path)
        return True
    except Exception as e:
        print('Compilation failed:', e)
        return False


def run_shim(shim_path, forward_args):
    cmd = [sys.executable, shim_path] + forward_args
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)


def main(argv=None):
    p = argparse.ArgumentParser(description='Install deps, write shim and optionally run Real-ESRGAN via the shim')
    p.add_argument('--install', action='store_true', help='Install default dependencies (basicsr, facexlib, gfpgan, realesrgan)')
    p.add_argument('--deps', nargs='*', default=None, help='List of pip packages to install (overrides default if provided)')
    p.add_argument('--shim', default='/kaggle/working/Real-ESRGAN/inference_with_shim_full.py', help='Path to write shim')
    p.add_argument('--run', action='store_true', help='Run inference via the written shim (forwards args after --)')
    p.add_argument('--no-check-compile', action='store_true', help='Do not py_compile check the written shim')
    p.add_argument('forward', nargs=argparse.REMAINDER, help='Arguments to forward to the shim/inference script (prefix with --)')

    args = p.parse_args(argv)

    if args.install:
        deps = args.deps if args.deps is not None and len(args.deps) > 0 else DEFAULT_DEPS
        install_packages(deps)

    print('Writing shim to', args.shim)
    write_shim(args.shim)

    if not args.no_check_compile:
        ok = check_compile(args.shim)
        if not ok:
            print('Aborting due to compile errors in shim')
            sys.exit(2)

    # 'forward' may start with '--' from CLI; normalize by removing leading '--' if present
    fwd = list(args.forward)
    if len(fwd) > 0 and fwd[0] == '--':
        fwd = fwd[1:]

    if args.run:
        try:
            run_shim(args.shim, fwd)
        except subprocess.CalledProcessError as e:
            print('Inference process failed with returncode', e.returncode)
            sys.exit(e.returncode)

if __name__ == '__main__':
    main()
