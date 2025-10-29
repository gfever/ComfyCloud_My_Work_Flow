"""Write a clean, indentation-safe Real-ESRGAN shim file suitable for Kaggle runtime.

Usage (from notebook or terminal):
    from kaggle_runtime_shim_writer import write_shim
    write_shim('/kaggle/working/Real-ESRGAN/inference_with_shim_full.py')

Or run as a script:
    python3 kaggle_runtime_shim_writer.py --out /kaggle/working/Real-ESRGAN/inference_with_shim_full.py

The script writes a sanitized shim that avoids fragile manual indentation and uses textwrap.dedent
so newlines/indentation are stable when written into the Kaggle working directory.
"""
import argparse
import os
import textwrap

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


def write_shim(out_path: str):
    """Write the predefined SHIM content to out_path, creating parent dirs as needed."""
    out_path = os.fspath(out_path)
    d = os.path.dirname(out_path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(SHIM)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', '-o', default='inference_with_shim_full.py', help='Output path for shim file')
    args = p.parse_args()
    outp = args.out
    try:
        write_shim(outp)
        print('Wrote shim to', outp)
    except Exception as e:
        print('Failed to write shim to', outp, ':', e)


if __name__ == '__main__':
    main()

