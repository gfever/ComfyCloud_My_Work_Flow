import os, sys, subprocess, time
# minimal context vars for syntax check
WORKSPACE = os.getcwd()
FRAMES_DIR = WORKSPACE + '/frames'
DATASET_DIR = WORKSPACE + '/dataset'
UPSCALE_MODEL_NAME = '4x-UltraSharp.pth'

INFERENCE_LOG = os.path.join(WORKSPACE, 'logs', 'real_esrgan_inference.log')

# simulate block
run_cmd = [sys.executable, 'inference_realesrgan.py', '-n', '4x-UltraSharp', '-i', FRAMES_DIR, '-o', FRAMES_DIR + '_upscaled', '--fp32']
try:
    res = subprocess.run(run_cmd, capture_output=True, text=True, timeout=3600)
    if res.returncode == 0:
        pass
    else:
        stderr_lower = (res.stderr or '').lower()
        if 'functional_tensor' in stderr_lower or "no module named 'torchvision.transforms.functional_tensor'" in (res.stderr or ''):
            try:
                shim_path = os.path.join(os.getcwd(), 'inference_with_torchvision_shim.py')
                shim_lines = [
                    "import sys, types",
                    "try:",
                    "    import torchvision.transforms.functional_tensor as ft",
                    "except Exception:",
                    "    try:",
                    "        import torchvision.transforms.functional as f",
                    "        mod = types.ModuleType('torchvision.transforms.functional_tensor')",
                    "        if hasattr(f, 'rgb_to_grayscale'):",
                    "            mod.rgb_to_grayscale = f.rgb_to_grayscale",
                    "        if hasattr(f, 'convert_image_dtype'):",
                    "            mod.convert_image_dtype = getattr(f, 'convert_image_dtype', None)",
                    "        sys.modules['torchvision.transforms.functional_tensor'] = mod",
                    "    except Exception:",
                    "        pass",
                    "",
                    "# forward to the original script",
                    "import runpy",
                    "runpy.run_path('inference_realesrgan.py', run_name='__main__')",
                ]
                with open(shim_path, 'w', encoding='utf-8') as sf:
                    sf.write('\n'.join(shim_lines) + '\n')

                res_shim = subprocess.run([sys.executable, shim_path, '-n', '4x-UltraSharp', '-i', FRAMES_DIR, '-o', FRAMES_DIR + '_upscaled', '--fp32'], capture_output=True, text=True, timeout=3600)
                if res_shim.returncode == 0:
                    res = res_shim
                else:
                    install_cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', 'torchvision', '-f', 'https://download.pytorch.org/whl/torch_stable.html']
                    res_install = subprocess.run(install_cmd, capture_output=True, text=True, timeout=600)
                    if res_install.returncode == 0:
                        res2 = subprocess.run(run_cmd, capture_output=True, text=True, timeout=3600)
                        if res2.returncode == 0:
                            res = res2
                        else:
                            raise RuntimeError('retry failed')
                    else:
                        raise RuntimeError('install failed')
            except Exception:
                raise
        else:
            raise RuntimeError('inference failed')
except Exception:
    raise

