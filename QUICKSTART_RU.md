# 🎬 Быстрый старт - Генерация видео на Kaggle

## Какой файл использовать?

### 🌟 Хочу настоящую плавную анимацию
→ Используй **`kaggle_animated.ipynb`**
- Персонажи двигаются естественно
- Плавные переходы и движения
- Лучшее качество

### ⚡ Хочу быстро и просто
→ Используй **`kaggle_simple.ipynb`**
- Включи `USE_ANIMATEDIFF = True` для анимации
- Или `USE_ANIMATEDIFF = False` для быстрой генерации

---

## 🚀 Пошаговая инструкция

### 1. Создай датасет на Kaggle
1. Зайди на [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. New Dataset
3. Загрузи файл: `sd_xl_base_1.0.safetensors` (модель SD XL)
4. Опционально: `4x-UltraSharp.pth` (upscale)
5. Make Public или Private

### 2. Создай Notebook
1. [kaggle.com/code](https://www.kaggle.com/code) → New Notebook
2. Settings (справа):
   - **Accelerator:** GPU T4 x2 (или P100)
   - **Persistence:** ON ✓
   - **Internet:** ON ✓

### 3. Подключи датасет
1. Add Input (справа)
2. Datasets → найди свой датасет
3. Add

### 4. Скопируй код
1. Открой `kaggle_animated.ipynb` (или `kaggle_simple.ipynb`)
2. Скопируй весь код
3. Вставь в Kaggle Notebook

### 5. Настрой промпт
```python
# Найди эти строки в начале:
PROMPT = "твое описание здесь"
```

**Примеры:**
```python
# Аниме персонаж
PROMPT = "gojo satoru turning head slowly, confident smirk, white hair flowing, black blindfold, anime style, smooth motion"

# Природа
PROMPT = "sakura petals falling, wind blowing branches, pink flowers, spring day, cinematic"

# Киберпанк
PROMPT = "cyberpunk city, neon lights, rain, camera moving forward, reflections"
```

### 6. Запусти!
1. Run All (Ctrl+Enter или кнопка вверху)
2. Жди 10-20 минут
3. Скачай готовое видео!

---

## ⚙️ Основные настройки

### Обязательные (меняй под себя):
```python
PROMPT = "что должно быть в видео"
NEGATIVE_PROMPT = "чего НЕ должно быть"
```

### Размер видео:
```python
WIDTH = 512   # ширина
HEIGHT = 768  # высота (вертикальное для shorts/reels)

# Горизонтальное:
WIDTH = 768
HEIGHT = 512

# Квадрат:
WIDTH = 512
HEIGHT = 512
```

### Анимация:
```python
USE_ANIMATEDIFF = True   # True = плавная анимация
USE_ANIMATEDIFF = False  # False = быстрее, но менее плавно

USE_RIFE = True   # Дополнительная интерполяция (плавнее)
USE_RIFE = False  # Без интерполяции (быстрее)
```

### Длина видео:
```python
NUM_FRAMES = 16  # кадров в анимации (16-24)

RIFE_EXP = 4  # 16→256 кадров (~30 сек)
RIFE_EXP = 3  # 16→128 кадров (~15 сек)
```

---

## 💡 Советы по промптам

### ✅ Хорошие промпты (для анимации):
```python
"character turning head, blinking eyes, smooth motion"
"hair flowing in wind, gentle movement"
"camera slowly zooming in, cinematic"
"petals falling, leaves rustling, natural motion"
```

### ❌ Плохие промпты:
```python
"beautiful picture"  # нет описания движения
"high quality 8k"    # только качество, нет действия
"anime girl"         # слишком общее
```

### Negative промпт (что убрать):
```python
NEGATIVE_PROMPT = "static, frozen, choppy animation, blurry, low quality, watermark, text"
```

---

## 🎯 Пресеты

### Максимальное качество (медленно ~20 мин):
```python
USE_ANIMATEDIFF = True
NUM_FRAMES = 24
USE_RIFE = True
RIFE_EXP = 4
STEPS = 30
WIDTH = 512
HEIGHT = 768
```

### Рекомендуется (баланс ~15 мин):
```python
USE_ANIMATEDIFF = True
NUM_FRAMES = 16
USE_RIFE = True
RIFE_EXP = 4
STEPS = 25
WIDTH = 512
HEIGHT = 768
```

### Быстро для теста (~8 мин):
```python
USE_ANIMATEDIFF = True
NUM_FRAMES = 16
USE_RIFE = False
STEPS = 20
WIDTH = 512
HEIGHT = 512
```

### Очень быстро (~5 мин):
```python
USE_ANIMATEDIFF = False
NUM_FRAMES = 8
USE_RIFE = False
STEPS = 15
WIDTH = 512
HEIGHT = 512
```

---

## 🐛 Проблемы?

### "Model not found"
→ Проверь путь к датасету в коде:
```python
DATASET_DIR = "/kaggle/input/ИМЯ-ТВОЕГО-ДАТАСЕТА"
```

### "Out of memory"
→ Уменьши размер:
```python
WIDTH = 384
HEIGHT = 576
NUM_FRAMES = 12
```

### Видео "дергается"
→ Используй AnimateDiff:
```python
USE_ANIMATEDIFF = True
```

### Очень медленно
→ Отключи RIFE для теста:
```python
USE_RIFE = False
```

---

## 📥 Где найти модели?

### SD XL Base (обязательно):
- [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- Или скрипт скачает автоматически

### Upscale (опционально):
- [4x-UltraSharp](https://openmodeldb.info/models/4x-UltraSharp)

---

## ✨ Результат

После выполнения получишь:
- 📹 Видео файл .mp4
- 🎬 30+ секунд плавной анимации
- 📱 Готово для YouTube Shorts / TikTok / Reels
- 🎨 Качество зависит от настроек

**Время:** 10-20 минут  
**Бесплатно** на Kaggle GPU!

---

## 🔗 Полезные ссылки

- [Stable Diffusion Prompts](https://lexica.art/) - примеры промптов
- [Kaggle GPU Quotas](https://www.kaggle.com/code) - лимиты GPU
- [AnimateDiff Guide](https://github.com/guoyww/AnimateDiff) - документация

---

**Вопросы?** Смотри подробный `README_RU.md`

