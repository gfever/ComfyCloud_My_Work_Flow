import sys, types
try:
    import torchvision.transforms.functional_tensor as ft
except Exception:
    try:
        import torchvision.transforms.functional as f
        mod = types.ModuleType('torchvision.transforms.functional_tensor')
        if hasattr(f, 'rgb_to_grayscale'):
            mod.rgb_to_grayscale = f.rgb_to_grayscale
        if hasattr(f, 'convert_image_dtype'):
            mod.convert_image_dtype = getattr(f, 'convert_image_dtype', None)
        sys.modules['torchvision.transforms.functional_tensor'] = mod
    except Exception:
        pass

# forward to the original script
import runpy
runpy.run_path('inference_realesrgan.py', run_name='__main__')

