ComfyGojo Dataset
=================

Содержит:
- `kaggle_simple_cells_v4_clean.ipynb` — основной notebook для генерации анимации (AnimateDiff / RIFE / Real-ESRGAN);
- `prompts.json` — дефолтный набор промптов (BASE_PROMPT, MOTION_PROMPT, EXTRA_PROMPT, NEGATIVE_PROMPT);

Как использовать
----------------
1. В вашем ноутбуке в Kaggle: справа → Add data → найдите и прикрепите `noxfvr/comfy-gojo-dataset`.
2. Перезапустите kernel (Runtime → Restart session).
3. Запустите ячейку "KAGGLE SETUP & CHECKS" в ноутбуке `kaggle_simple_cells_v4_clean.ipynb`.
4. Запустите ячейку "SMOKE TEST" — она подтвердит, что `prompts.json` и (опционально) HF-token доступны.
5. Откройте ячейку PROMPTS (GUI) для редактирования промптов или загрузите `prompts.json` через GUI.

Hugging Face token
------------------
- Для загрузки моделей с Hugging Face создайте отдельный приватный датасет с файлом `token.txt` (содержит ваш токен) и прикрепите его к ноутбуку.
- Не храните `token.txt` вместе с публичными файлами.

Обновление датасета локально
---------------------------
Если вы меняете `prompts.json` или notebook и хотите опубликовать обновление через kaggle CLI:

```bash
# в PowerShell или bash (при наличии настроенного kaggle.json)
kaggle datasets version -p "C:\PHPStormProjects\ComfyCloud_My_Work_Flow\comfy-gojo-dataset" -m "Update prompts / README"
```

Лицензия
--------
По умолчанию `CC0-1.0` (см. dataset-metadata.json). Если вы хотите другую лицензию — отредактируйте `dataset-metadata.json` перед публикацией.

Контакты / примечания
---------------------
Если будут вопросы по использованию notebook или интеграции с Hugging Face — пришлите вывод ячеек `KAGGLE SETUP & CHECKS` и `SMOKE TEST`, я помогу диагностировать.

