import sys, types, os, re, runpy

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
        # best-effort fallback: create a stub module with conservative placeholders
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
            # If list/tuple, return first tensor to keep imports working in simple cases
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
        # common extensions
        exts = ('.pth', '.pt', '.safetensors')
        candidates = [f for f in os.listdir(wdir) if f.lower().endswith(exts)]
        if not candidates:
            return None
        # prefer files that contain '4x' or 'ultrasharp' in name
        for name in candidates:
            if '4x' in name.lower() or 'ultrasharp' in name.lower():
                return os.path.join(wdir, name)
        # otherwise return the first candidate
        return os.path.join(wdir, candidates[0])
    except Exception:
        return None

# Forward to the original inference_realesrgan.py but ensure safe globals are defined
if __name__ == '__main__':
    argv = sys.argv[1:]
    netscale = _infer_netscale(argv)

    # try to detect a local model in ./weights
    detected_model = _detect_weight_file()

    script_path = os.path.join(os.getcwd(), 'inference_realesrgan.py')
    try:
        with open(script_path, 'r', encoding='utf-8') as _f:
            _code = _f.read()
    except Exception:
        # fallback: run by runpy if file cannot be read
        sys.argv = [script_path] + argv
        runpy.run_path('inference_realesrgan.py', run_name='__main__')
    else:
        # Prepare globals for exec. Provide safe defaults to avoid UnboundLocalError
        _globals = {
            '__name__': '__main__',
            '__file__': script_path,
            'netscale': netscale,
            'model': detected_model,   # try to predefine a model path if available
            'outscale': netscale,      # sensible default for scripts expecting outscale/scale
            'scale': netscale,
        }
        # make sys available inside executed script and set argv
        _globals['sys'] = sys
        sys.argv = [script_path] + argv
        # Execute the target script in the prepared global context
        exec(compile(_code, script_path, 'exec'), _globals)

