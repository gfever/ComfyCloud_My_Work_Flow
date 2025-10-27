del pipe
torch.cuda.empty_cache()

print(f"✓ Кадры сохранены в: {FRAMES_DIR}")
# === UPSCALE С REAL-ESRGAN (ОПЦИОНАЛЬНО) ===
print_separator()
print("📈 АПСКЕЙЛ КАДРОВ С REAL-ESRGAN")
print_separator(nl_before=False)

# Защитные значения, чтобы ячейка была idempotent при запуске отдельно
import os
if 'WORKSPACE' not in globals():
    WORKSPACE = os.getcwd()
    print(f"⚠️ Переменная WORKSPACE не найдена — использую: {WORKSPACE}")
if 'FRAMES_DIR' not in globals():
    FRAMES_DIR = f"{WORKSPACE}/frames"
    print(f"⚠️ Переменная FRAMES_DIR не найдена — использую: {FRAMES_DIR}")
if 'DATASET_DIR' not in globals():
    DATASET_DIR = f"{WORKSPACE}/dataset"
    print(f"⚠️ Переменная DATASET_DIR не найдена — использую: {DATASET_DIR}")

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

upscale_model = f"{DATASET_DIR}/{UPSCALE_MODEL_NAME}"

if os.path.exists(upscale_model):
    # Клонируем Real-ESRGAN если еще не установлен
    if not os.path.exists(f"{WORKSPACE}/Real-ESRGAN"):
        print("Установка Real-ESRGAN...")
        import subprocess, os
        subprocess.check_call(["git", "clone", "https://github.com/xinntao/Real-ESRGAN", f"{WORKSPACE}/Real-ESRGAN"])
    # Переходим в папку репозитория
    os.chdir(f"{WORKSPACE}/Real-ESRGAN")
    # Устанавливаем зависимости через pip и логируем подробный вывод в файл
    LOG_DIR = os.path.join(WORKSPACE, 'logs')
    os.makedirs(LOG_DIR, exist_ok=True)
    INSTALL_LOG = os.path.join(LOG_DIR, 'real_esrgan_install.log')
    INFERENCE_LOG = os.path.join(LOG_DIR, 'real_esrgan_inference.log')

    def try_install_packages(pkg_list, install_log_path=INSTALL_LOG):
        import subprocess, sys, time
        cmd = [sys.executable, '-m', 'pip', 'install'] + pkg_list + ['--verbose']
        print('Running pip install (verbose). Full log will be written to:', install_log_path)
        with open(install_log_path, 'w', encoding='utf-8') as lf:
            lf.write(f'Command: {" ".join(cmd)}\n')
            lf.write('Timestamp: ' + time.strftime('%Y-%m-%d %H:%M:%S') + '\n\n')
            try:
                # 1) Try to install system build deps (works in Kaggle/Colab if allowed)
                try:
                    lf.write('\n=== Attempting apt-get install build deps ===\n')
                    print('Attempting apt-get install build-essential, cmake, libgl1-mesa-dev, libjpeg-dev (may require root)')
                    subprocess.run(['apt-get', 'update'], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    res_apt = subprocess.run(['apt-get', 'install', '-y', 'build-essential', 'cmake', 'libgl1-mesa-dev', 'libjpeg-dev'], capture_output=True, text=True, timeout=300)
                    lf.write('=== APT STDOUT ===\n')
                    lf.write(res_apt.stdout or '')
                    lf.write('\n=== APT STDERR ===\n')
                    lf.write(res_apt.stderr or '')
                except Exception as _e:
                    lf.write('\n=== APT STEP FAILED ===\n')
                    lf.write(repr(_e))
                    lf.write('\n')
                    print('Note: apt-get step failed or not permitted in this environment; continuing to pip steps')

                # 2) Upgrade core Python packaging tools to reduce build errors
                lf.write('\n=== Upgrading pip, wheel, setuptools, setuptools_scm ===\n')
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip', 'wheel', 'setuptools', 'setuptools_scm'], capture_output=True, text=True, timeout=300)
                lf.write('Upgraded pip/setuptools/wheel/setuptools_scm\n')

                res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                lf.write('=== STDOUT ===\n')
                lf.write(res.stdout or '')
                lf.write('\n=== STDERR ===\n')
                lf.write(res.stderr or '')
                lf.flush()
                if res.returncode == 0:
                    print('✓ packages installed successfully; verbose log saved to', install_log_path)
                    return True
                else:
                    print('✗ pip install failed with return code', res.returncode)
                    raise RuntimeError(f'pip install failed (return code {res.returncode}). See log: {install_log_path}')
            except Exception as e:
                lf.write('\n=== EXCEPTION ===\n')
                lf.write(repr(e))
                lf.flush()
                print('✗ Exception during pip install; see log:', install_log_path)
                raise

    # Попытка установки. При ошибке поднимаем исключение, чтобы пользователь видел лог и мог диагностировать.
    try:
        try_install_packages(['basicsr', 'facexlib', 'gfpgan', 'realesrgan'])
    except Exception as e:
        # Явно сообщаем пользователю и прерываем выполнение, но даём путь к логам.
        raise RuntimeError(f"Не удалось установить зависимости Real-ESRGAN: {e}\nСм. лог: {INSTALL_LOG}")

    # Копируем модель в weights
    import shutil
    os.makedirs('weights', exist_ok=True)
    shutil.copy(upscale_model, 'weights/')

    # Апскейл каждого кадра (запускаем скрипт Real-ESRGAN через Python)
    print("\nЗапуск апскейла...")
    # Запуск inference с логированием stdout/stderr в файл
    with open(INFERENCE_LOG, 'w', encoding='utf-8') as lf:
        lf.write('Running inference_realesrgan.py\n')
        lf.flush()
        try:
            res = subprocess.run([sys.executable, 'inference_realesrgan.py', '-n', '4x-UltraSharp', '-i', FRAMES_DIR, '-o', FRAMES_DIR + '_upscaled', '--fp32'], capture_output=True, text=True, timeout=3600)
            lf.write('=== STDOUT ===\n')
            lf.write(res.stdout or '')
            lf.write('\n=== STDERR ===\n')
            lf.write(res.stderr or '')
            lf.flush()
            if res.returncode != 0:
                print('\n⚠️ Real-ESRGAN inference failed with return code', res.returncode)
                print('See inference log:', INFERENCE_LOG)
                # If failure due to missing torchvision functional_tensor, try to install/upgrade torchvision and retry once
                stderr_lower = (res.stderr or '').lower()
                if 'functional_tensor' in stderr_lower or "no module named 'torchvision.transforms.functional_tensor'" in (res.stderr or ''):
                    lf.write('\nDetected missing torchvision functional_tensor. Attempting to install/upgrade torchvision and retry inference.\n')
                    lf.flush()
                    try:
                        # Attempt to install torchvision (will pick a compatible wheel if available)
                        install_cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', 'torchvision', '-f', 'https://download.pytorch.org/whl/torch_stable.html']
                        lf.write('Retry install command: ' + ' '.join(install_cmd) + '\n')
                        lf.flush()

                        # Попробуем сначала обойти проблему без установки: запустим inference через shim,
                        # который создаст модуль torchvision.transforms.functional_tensor, если его нет,
                        # переадресовав вызовы на torchvision.transforms.functional.
                        try:
                            lf.write('\nAttempting shim run (provides torchvision.transforms.functional_tensor at runtime)\n')
                            lf.flush()
                            shim_path = os.path.join(os.getcwd(), 'inference_with_torchvision_shim.py')
                            with open(shim_path, 'w', encoding='utf-8') as sf:
                                shim_lines = [
                                    "import sys, types, os, re",
                                    "# Provide torchvision.transforms.functional_tensor shim when missing",
                                    "try:",
                                    "    import torchvision.transforms.functional_tensor as ft",
                                    "except Exception:",
                                    "    try:",
                                    "        import torchvision.transforms.functional as f",
                                    "        mod = types.ModuleType('torchvision.transforms.functional_tensor')",
                                    "        # copy a few commonly-used functions if present",
                                    "        if hasattr(f, 'rgb_to_grayscale'):",
                                    "            mod.rgb_to_grayscale = f.rgb_to_grayscale",
                                    "        if hasattr(f, 'convert_image_dtype'):",
                                    "            mod.convert_image_dtype = getattr(f, 'convert_image_dtype', None)",
                                    "        sys.modules['torchvision.transforms.functional_tensor'] = mod",
                                    "    except Exception:",
                                    "        pass",
                                    "",
                                    "# Helper: infer numeric scale (e.g. '4x-UltraSharp' -> 4) from argv",
                                    "def _infer_netscale(argv):",
                                    "    # argv is a list of command-line args (no program name)",
                                    "    for i, a in enumerate(argv):",
                                    "        # check flags that accept a model name (common Real-ESRGAN uses '-n')",
                                    "        if a in ('-n','--name','--model') and i+1 < len(argv):",
                                    "            m = re.match(r'(\\d+)x', argv[i+1])",
                                    "            if m:",
                                    "                return int(m.group(1))",
                                    "    # attempt to find direct --scale/-s",
                                    "    for i, a in enumerate(argv):",
                                    "        if a in ('-s','--scale') and i+1 < len(argv):",
                                    "            try:",
                                    "                return int(argv[i+1])",
                                    "            except Exception:",
                                    "                pass",
                                    "    # fallback default",
                                    "    return 4",
                                    "",
                                    "# compute netscale from sys.argv (exclude program name)",
                                    "netscale = _infer_netscale(sys.argv[1:])",
                                    "",
                                    "# Execute the original script in a prepared __main__ namespace so it sees 'netscale'",
                                    "script_path = os.path.join(os.getcwd(), 'inference_realesrgan.py')",
                                    "with open(script_path, 'r', encoding='utf-8') as _f:",
                                    "    _code = _f.read()",
                                    "_globals = {'__name__': '__main__', '__file__': script_path, 'netscale': netscale}",
                                    "exec(compile(_code, script_path, 'exec'), _globals)",
                                ]
                                sf.write('\n'.join(shim_lines) + '\n')

                            res_shim = subprocess.run([sys.executable, shim_path, '-n', '4x-UltraSharp', '-i', FRAMES_DIR, '-o', FRAMES_DIR + '_upscaled', '--fp32'], capture_output=True, text=True, timeout=3600)
                            lf.write('\n=== SHIM STDOUT ===\n')
                            lf.write(res_shim.stdout or '')
                            lf.write('\n=== SHIM STDERR ===\n')
                            lf.write(res_shim.stderr or '')
                            lf.flush()

                            if res_shim.returncode == 0:
                                lf.write('\nShim run succeeded; log saved to inference log.\n')
                                lf.flush()
                                # используем результат shim'а
                                res = res_shim
                            else:
                                lf.write('\nShim run failed; will attempt to install/upgrade torchvision and retry inference.\n')
                                lf.flush()
                                # fallback: попытаемся установить torchvision как раньше
                                res_install = subprocess.run(install_cmd, capture_output=True, text=True, timeout=600)
                                lf.write('=== TORCHVISION INSTALL STDOUT ===\n')
                                lf.write(res_install.stdout or '')
                                lf.write('\n=== TORCHVISION INSTALL STDERR ===\n')
                                lf.write(res_install.stderr or '')
                                lf.flush()

                                if res_install.returncode == 0:
                                    lf.write('\ntorchvision installed/upgraded; retrying inference...\n')
                                    lf.flush()
                                    # retry inference
                                    res2 = subprocess.run([sys.executable, 'inference_realesrgan.py', '-n', '4x-UltraSharp', '-i', FRAMES_DIR, '-o', FRAMES_DIR + '_upscaled', '--fp32'], capture_output=True, text=True, timeout=3600)
                                    lf.write('\n=== RETRY STDOUT ===\n')
                                    lf.write(res2.stdout or '')
                                    lf.write('\n=== RETRY STDERR ===\n')
                                    lf.write(res2.stderr or '')
                                    lf.flush()
                                    if res2.returncode == 0:
                                        print('✓ Real-ESRGAN inference completed after installing torchvision; log saved to', INFERENCE_LOG)
                                        # replace res with res2 so downstream continues
                                        res = res2
                                    else:
                                        print('\n✗ Retry also failed. See inference log:', INFERENCE_LOG)
                                        raise RuntimeError(f'Real-ESRGAN inference retry failed (return code {res2.returncode}). See log: {INFERENCE_LOG}')
                                else:
                                    print('\n✗ Could not install/upgrade torchvision. See install log in the inference log file.')
                                    raise RuntimeError(f'Could not install torchvision (return code {res_install.returncode}). See log: {INFERENCE_LOG}')
                        except Exception as _e:
                            lf.write('\n=== EXCEPTION DURING TORCHVISION SHIM/INSTALL/RETRY ===\n')
                            lf.write(repr(_e))
                            lf.flush()
                            print('✗ Exception during torchvision shim/install/retry. See inference log:', INFERENCE_LOG)
                            raise
                else:
                    raise RuntimeError(f'Real-ESRGAN inference failed (return code {res.returncode}). See log: {INFERENCE_LOG}')
            else:
                print('✓ Real-ESRGAN inference completed; log saved to', INFERENCE_LOG)
        except Exception as e:
            print('✗ Exception during Real-ESRGAN inference; see log:', INFERENCE_LOG)
            raise

    # Заменяем оригинальные кадры апскейленными
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    shutil.move(FRAMES_DIR + '_upscaled', FRAMES_DIR)

    # Переименовываем обратно (Real-ESRGAN добавляет суффикс)
    import glob
    upscaled_files = sorted(glob.glob(f"{FRAMES_DIR}/*_out.png"))
    for i, filepath in enumerate(upscaled_files):
        os.rename(filepath, f"{FRAMES_DIR}/{i}.png")

    print("\n✓ Апскейл завершен!")
    print('Install log:', INSTALL_LOG)
    print('Inference log:', INFERENCE_LOG)
    os.chdir(WORKSPACE)
else:
    # Попытаемся найти модель в нескольких стандартных местах и дать пользователю понятную инструкцию
    candidates = [
        os.path.join(DATASET_DIR, UPSCALE_MODEL_NAME),
        os.path.join(WORKSPACE, 'dataset', UPSCALE_MODEL_NAME),
        os.path.join(WORKSPACE, 'models', UPSCALE_MODEL_NAME),
        '/kaggle/input/comfyui-models-gojo/' + UPSCALE_MODEL_NAME,
        os.path.expanduser('~/models/' + UPSCALE_MODEL_NAME),
    ]
    found = None
    for p in candidates:
        if p and os.path.exists(p):
            found = p
            break

    if found:
        print(f"ℹ️ Модель Upscale найдена: {found} — обновляю путь и запускаю апскейл.")
        upscale_model = found
        # Повторяем логику запуска апскейла (минимально): клонирование/копирование/запуск
        if not os.path.exists(f"{WORKSPACE}/Real-ESRGAN"):
            print("Установка Real-ESRGAN...")
            import subprocess, os
            subprocess.check_call(["git", "clone", "https://github.com/xinntao/Real-ESRGAN", f"{WORKSPACE}/Real-ESRGAN"])
        os.chdir(f"{WORKSPACE}/Real-ESRGAN")
        SKIP_REAL_ESRGAN = False
        ok = try_install_packages(['basicsr', 'facexlib', 'gfpgan', 'realesrgan'])
        if not ok:
            SKIP_REAL_ESRGAN = True
            print('\n⚠️ Установка зависимостей не удалась. Апскейл пропущен.')
        import shutil, glob
        os.makedirs('weights', exist_ok=True)
        shutil.copy(upscale_model, 'weights/')
        print("\nЗапуск апскейла...")
        if not SKIP_REAL_ESRGAN:
            try:
                subprocess.check_call([sys.executable, 'inference_realesrgan.py', '-n', '4x-UltraSharp', '-i', FRAMES_DIR, '-o', FRAMES_DIR + '_upscaled', '--fp32'])
            except subprocess.CalledProcessError as e:
                print('\n⚠️ Real-ESRGAN inference failed:', e)
                print('Апскейл пропущен — продолжим без апскейла')
                SKIP_REAL_ESRGAN = True
        else:
            print('Апскейл пропущен из-за ошибки установки зависимостей')
        shutil.rmtree(FRAMES_DIR, ignore_errors=True)
        shutil.move(FRAMES_DIR + '_upscaled', FRAMES_DIR)
        upscaled_files = sorted(glob.glob(f"{FRAMES_DIR}/*_out.png"))
        for i, filepath in enumerate(upscaled_files):
            os.rename(filepath, f"{FRAMES_DIR}/{i}.png")
        print("\n✓ Апскейл завершен!")
        os.chdir(WORKSPACE)
    else:
        print("⚠️ Upscale модель не найдена — пропускаем апскейл.")
        print("   Проверьте, что файл '4x-UltraSharp.pth' присутствует в одной из директорий:")
        for c in candidates:
            print(f"     - {c}")
        print("   Или поместите модель в папку вашего датасета и установите DATASET_DIR в начале ноутбука, например:")
        print("     DATASET_DIR = '/kaggle/input/comfyui-models-gojo'")
        print("   После размещения модели перезапустите kernel и выполните ячейки заново.")
