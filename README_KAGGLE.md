README — Как импортировать и запустить notebook в Kaggle

Кратко
------
Этот файл описывает, как быстро импортировать обновлённый ноутбук `kaggle_simple_cells_v4_clean.ipynb` и `prompts.json` в среду Kaggle, как создать датасет через Kaggle CLI и как подготовить Hugging Face token и ipywidgets.

Файлы в репозитории
-------------------
- `kaggle_simple_cells_v4_clean.ipynb` — основной notebook (обновлённый). 
- `prompts.json` — дефолтные промпты (GUI будет подхватывать его при запуске).
- `create_kaggle_dataset.py` — скрипт для упаковки файлов и вызова `kaggle datasets create`.

Рекомендованный порядок операций в Kaggle (веб-интерфейс)
-------------------------------------------------------
1. Откройте Kaggle.com → Notebooks → New Notebook → Upload Notebook → выберите `kaggle_simple_cells_v4_clean.ipynb`.
2. В правой панели "Add data" прикрепите датасеты, которые нужны (модели, ваш датасет с весами, и/или датасет с Hugging Face token):
   - Если у вас есть Hugging Face token, создайте локально файл `token.txt` с содержимым токена и загрузите как новый dataset, затем прикрепите к ноутбуку. В ноутбуке он будет доступен как `/kaggle/input/<dataset-name>/token.txt` — наш notebook ищет `/kaggle/input/hf-token/token.txt`.
3. В Settings включите Accelerator → GPU (если нужно GPU).
4. Запустите ядро (Restart session) после установки зависимостей и/или добавления токена (см. ниже).

Если вы хотите загружать файлы в Kaggle как dataset (CLI)
--------------------------------------------------------
1) Установите и настройте Kaggle CLI локально (https://www.kaggle.com/docs/api):
   - поместите `kaggle.json` в `%USERPROFILE%\.kaggle\kaggle.json` на Windows (или `~/.kaggle/kaggle.json` на Linux/Mac).
2) Используйте скрипт `create_kaggle_dataset.py` (в репозитории) — он упаковует `kaggle_simple_cells_v4_clean.ipynb` и `prompts.json` в каталог и вызовет `kaggle datasets create`.

Пример (локально, bash / cmd):

- Linux / macOS (bash):
```bash
python create_kaggle_dataset.py --folder-name comfy-gojo-dataset
# Затем, если команда выполнена, dataset будет создан в вашем аккаунте
```

- Windows (cmd.exe):
```cmd
python create_kaggle_dataset.py --folder-name comfy-gojo-dataset
```

Примечание: скрипт вызывает `kaggle datasets create -p <folder>` и потребует, чтобы ваш `kaggle` был аутентифицирован.

Установка ipywidgets и зависимостей
----------------------------------
В ноутбуке я автоматически добавил установку `ipywidgets` в секцию "УСТАНОВКА ЗАВИСИМОСТЕЙ". Если вы хотите вручную установить/проверить, выполните ячейку с командой:

```bash
!pip install -q ipywidgets
```

После установки `ipywidgets` обязательно перезапустите kernel (Runtime → Restart session) — иначе GUI-виджеты не отрисуются.

Hugging Face token
------------------
- Рекомендуемый способ для Kaggle: создать dataset с файлом `token.txt` и подключить его, тогда ноутбук автоматически прочитает `/kaggle/input/hf-token/token.txt` и установит переменную `HUGGINGFACE_HUB_TOKEN`.
- Альтернатива: напрямую установить переменную в ноутбуке (в отдельной ячейке):

```python
import os
os.environ['HUGGINGFACE_HUB_TOKEN'] = 'ВАШ_ТОКЕН'
```

Порядок запуска ячеек (после загрузки ноутбука в Kaggle)
------------------------------------------------------
1. Ранние ячейки: ENV / DIAGNOSTICS (ранняя настройка окружения).  
2. `KAGGLE SETUP & CHECKS` — проверка HF token и ipywidgets.  
3. PROMPTS (GUI) — настройте prompt, нажмите Update PROMPT.  
4. (Опционально) `ЗАГРУЗКА ANIMATEDIFF PIPELINE` — подготовьте pipeline заранее.  
5. `ГЕНЕРАЦИЯ КАДРОВ / АНИМАЦИИ` — запустите генерацию.  
6. Нажмите `Unload pipe` или выполните `unload_pipe()` — чтобы освободить VRAM.

Быстрый тест (экономный режим)
------------------------------
Для первого прогона измените ячейку настроек:
```python
WIDTH = 256
HEIGHT = 256
NUM_FRAMES = 4
STEPS = 15
```
Это позволит быстро проверить рабочий процесс.

Если нужна помощь с созданием датасета в вашем аккаунте Kaggle — скажите, и я подготовлю команды/JSON для Kaggle CLI, подходящие к вашему аккаунту (вам потребуется kaggle.json).

---
Файл `create_kaggle_dataset.py` объяснён ниже и находится в этом репозитории.


