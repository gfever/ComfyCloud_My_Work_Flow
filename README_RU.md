# 🎬 ComfyCloud Video Generation

Скрипты для генерации видео на Kaggle с использованием Stable Diffusion + RIFE интерполяции.

## 📁 Файлы

### ⭐⭐ **kaggle_animated.ipynb** - ЛУЧШИЙ ДЛЯ АНИМАЦИИ
**Настоящая анимация с AnimateDiff**

**Преимущества:**
- ✅ Настоящая плавная анимация (не просто склейка кадров)
- ✅ Персонажи двигаются естественно
- ✅ Простая настройка через переменные
- ✅ Можно добавить RIFE для ещё большей плавности

**Что нужно менять:**
```python
PROMPT = "gojo satoru turning head, blinking, smooth motion"
USE_ANIMATEDIFF = True  # True = анимация
USE_RIFE = True  # Дополнительная интерполяция
NUM_FRAMES = 16  # Кадров анимации (16-24)
```

**Время выполнения:** ~8-15 минут
**Качество:** ⭐⭐⭐⭐⭐ (плавная анимация)

---

### ⭐ **kaggle_simple.ipynb** - УНИВЕРСАЛЬНЫЙ
**Два режима: статичные кадры ИЛИ AnimateDiff**

**Преимущества:**
- ✅ Переключение режимов одной переменной
- ✅ Быстрый запуск
- ✅ Простая настройка

**Режимы:**
```python
# Режим 1: Настоящая анимация
USE_ANIMATEDIFF = True  # AnimateDiff

# Режим 2: Статичные кадры + RIFE
USE_ANIMATEDIFF = False  # Обычная генерация
```

**Время выполнения:** 
- С AnimateDiff: ~10-15 минут
- Без AnimateDiff: ~5-10 минут

---

### kaggle4OPT.ipynb
**Версия с ComfyUI**

**Когда использовать:**
- Нужны продвинутые возможности ComfyUI
- Нужен batch processing
- Используете кастомные workflows

**Недостатки:**
- Долгая установка ComfyUI (~3-5 минут)
- Сложнее менять параметры
- Больше кода

---

### kaggle3.ipynb
**Старая версия** - не рекомендуется, оставлена для истории

---

## 🚀 Как использовать

### 1️⃣ Создайте датасет на Kaggle
Загрузите в датасет:
- `sd_xl_base_1.0.safetensors` - модель Stable Diffusion XL
- `4x-UltraSharp.pth` - upscale модель (опционально)

### 2️⃣ Создайте новый Notebook на Kaggle
- GPU: T4 или P100
- Persistence: On (для сохранения моделей между запусками)

### 3️⃣ Подключите датасет
Settings → Add Input → Datasets → ваш датасет

### 4️⃣ Скопируйте код
Используйте `kaggle_simple.ipynb` для простой генерации

### 5️⃣ Измените настройки
Отредактируйте секцию в начале файла:
```python
PROMPT = "ваш промпт здесь"
```

### 6️⃣ Запустите
Run All → ждите результат!

---

## 🎬 Статичные кадры VS Анимация

### Статичные кадры + RIFE (USE_ANIMATEDIFF = False)
- Генерируется 8-12 отдельных картинок с разными seed
- RIFE интерполирует между ними
- **Плюсы:** Быстрее, меньше VRAM
- **Минусы:** Движения могут быть резкими, "телепортация" объектов
- **Лучше для:** Статичные сцены, пейзажи, медленные переходы

### AnimateDiff анимация (USE_ANIMATEDIFF = True)
- Генерируется настоящая анимация 16-24 кадра
- Модель понимает движение и создает плавные переходы
- **Плюсы:** Естественные движения, плавная анимация
- **Минусы:** Медленнее, больше VRAM
- **Лучше для:** Персонажи, лица, движения, действия

### Комбинация (AnimateDiff + RIFE)
- Сначала AnimateDiff создает 16 плавных кадров
- Потом RIFE интерполирует до 256+ кадров
- **Результат:** Супер-плавное видео высокого качества
- **Время:** ~10-20 минут, но качество максимальное

---

## 🎨 Примеры промптов

### Для AnimateDiff (движение важно!)
```python
# Аниме персонаж с движением
PROMPT = "gojo satoru turning head slowly, blinking eyes, confident smirk, white hair flowing in wind, black blindfold, anime style, smooth motion, cinematic, 8k"
NEGATIVE_PROMPT = "static, frozen, choppy animation, blurry, low quality"

# Природа в движении
PROMPT = "beautiful sakura tree, petals falling gently, wind blowing branches, cherry blossoms, spring day, smooth camera pan, cinematic nature documentary"

# Камера в движении
PROMPT = "cyberpunk city street, neon lights, camera dolly forward, rain drops, reflections, blade runner style, smooth camera movement"
```

### Для статичных кадров (RIFE интерполяция)
```python
# Медленные переходы
PROMPT = "beautiful mountain landscape, golden hour, sunset colors changing, peaceful atmosphere, 8k"

# Портрет с изменениями
PROMPT = "gojo satoru portrait, anime style, different subtle expressions, professional lighting, 8k"
```

---

## ⚙️ Параметры

### Размер изображения
- **512x768** - вертикальное (shorts/reels) - БЫСТРО
- **768x512** - горизонтальное - БЫСТРО
- **1024x1024** - квадрат - СРЕДНЕ
- **1536x1024** - широкоформат - МЕДЛЕННО

### Steps (качество)
- **15** - быстро, среднее качество
- **20** - баланс (рекомендуется)
- **30** - высокое качество, медленно

### CFG Scale
- **5-7** - больше креативности
- **7-10** - баланс (рекомендуется)
- **10-15** - точное следование промпту

### RIFE Interpolation
- **exp=3** - 8→64 кадров (~7 сек видео)
- **exp=4** - 8→128 кадров (~15 сек видео)
- **exp=5** - 8→256 кадров (~30 сек видео) ⭐

---

## 🐛 Проблемы и решения

### "Model not found"
Проверьте путь к датасету в переменной `DATASET_DIR`

### "Out of memory"
Уменьшите WIDTH/HEIGHT или NUM_FRAMES

### "RIFE не создал видео"
Скрипт автоматически создаст fallback видео из исходных кадров

### Видео получается "дергающимся"
- Увеличьте NUM_FRAMES (12-16 вместо 8)
- Уменьшите разницу между seed (используйте 42, 42.5, 43...)

---

## 📊 Время выполнения (Kaggle T4)

### kaggle_animated.ipynb (AnimateDiff):
| Этап | Время |
|------|-------|
| Установка зависимостей (1 раз) | ~2 мин |
| Генерация анимации 16 кадров | ~5-7 мин |
| RIFE интерполяция 16→256 | ~5-10 мин |
| **ИТОГО** | ~12-19 мин |
| **Качество** | ⭐⭐⭐⭐⭐ |

### kaggle_simple.ipynb (с AnimateDiff):
| Этап | Время |
|------|-------|
| Установка зависимостей (1 раз) | ~2 мин |
| Генерация анимации 16 кадров | ~5-7 мин |
| RIFE интерполяция (опционально) | ~5-10 мин |
| **ИТОГО** | ~10-17 мин |

### kaggle_simple.ipynb (статичные кадры):
| Этап | Время |
|------|-------|
| Установка зависимостей (1 раз) | ~2 мин |
| Генерация 8 кадров 512x768 | ~3-5 мин |
| RIFE интерполяция 8→256 | ~5-10 мин |
| **ИТОГО** | ~10-17 мин |
| **Качество** | ⭐⭐⭐ |

С ComfyUI добавляется +5 минут на установку.

---

## 💡 Советы

1. **Для анимации используйте AnimateDiff** - гораздо плавнее чем статичные кадры
2. **Описывайте движение в промпте:**
   - AnimateDiff: "turning head", "blinking", "hair flowing", "camera movement"
   - Статичные: можно не указывать движение
3. **Negative промпт для анимации:** "static, frozen, choppy animation"
4. **Первый запуск всегда дольше** - модели скачиваются
5. **Включите Persistence** в Kaggle для сохранения между запусками
6. **Начните с малых параметров** - 512x768, 16 кадров для теста
7. **RIFE работает лучше с AnimateDiff** - дает супер-плавное видео
8. **Для вертикальных видео (shorts/reels):** WIDTH=512, HEIGHT=768

### Оптимальные настройки:

**Максимальное качество (медленно):**
```python
USE_ANIMATEDIFF = True
NUM_FRAMES = 24
USE_RIFE = True
RIFE_EXP = 4
WIDTH = 512
HEIGHT = 768
```

**Баланс (рекомендуется):**
```python
USE_ANIMATEDIFF = True
NUM_FRAMES = 16
USE_RIFE = True
RIFE_EXP = 4
WIDTH = 512
HEIGHT = 768
```

**Быстро (для тестов):**
```python
USE_ANIMATEDIFF = False
NUM_FRAMES = 8
USE_RIFE = False
WIDTH = 512
HEIGHT = 512
```

---

## 🔗 Полезные ссылки

- [Stable Diffusion Prompting Guide](https://stable-diffusion-art.com/prompt-guide/)
- [RIFE GitHub](https://github.com/hzwer/Practical-RIFE)
- [Kaggle GPU Quotas](https://www.kaggle.com/code)

---

## 📝 Changelog

### v4 (2025-10-25) - kaggle_animated.ipynb + AnimateDiff
- ✅ Добавлен AnimateDiff для настоящей анимации
- ✅ Плавные естественные движения персонажей
- ✅ Два режима в kaggle_simple.ipynb: AnimateDiff ИЛИ статичные кадры
- ✅ Комбинация AnimateDiff + RIFE для супер-плавного видео
- ✅ Примеры промптов для анимации
- ✅ Подробное сравнение методов

### v3 (2025-10-25) - kaggle_simple.ipynb
- ✅ Убран ComfyUI, используется diffusers напрямую
- ✅ Добавлена секция настроек вверху
- ✅ Упрощена структура кода
- ✅ Ускорен запуск в 3-5 раз

### v2 - kaggle4OPT.ipynb  
- Оптимизация batch generation
- Исправлена ошибка с именами файлов RIFE

### v1 - kaggle3.ipynb
- Первая версия с ComfyUI

