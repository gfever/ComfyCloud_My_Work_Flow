# 🎯 Схема выбора файла

```
                      Нужно видео?
                           |
                          Да
                           |
           ┌───────────────┴───────────────┐
           |                               |
    Важна анимация?                   Быстрый тест?
           |                               |
          Да                              Да
           |                               |
    ┌──────┴──────┐                        └──> kaggle_simple.ipynb
    |             |                             USE_ANIMATEDIFF = False
Персонаж?    Природа?                           USE_RIFE = False
    |             |                             ~5-7 минут
   Да            Да
    |             |
    └──────┬──────┘
           |
   Максимальное качество?
           |
    ┌──────┴──────┐
    |             |
   Да            Нет
    |             |
    v             v
kaggle_animated  kaggle_simple
  .ipynb           .ipynb
                USE_ANIMATEDIFF = True
                USE_RIFE = True

⏱️ ~15-20 min    ⏱️ ~8-12 min
⭐⭐⭐⭐⭐⭐        ⭐⭐⭐⭐⭐
```

---

# 📋 Чек-лист перед запуском

## На Kaggle:
- [ ] GPU включен (T4 или P100)
- [ ] Persistence включен
- [ ] Internet включен
- [ ] Датасет подключен

## В коде:
- [ ] PROMPT изменен на свой
- [ ] NEGATIVE_PROMPT добавлен
- [ ] WIDTH/HEIGHT выбраны (512x768 для shorts)
- [ ] USE_ANIMATEDIFF = True (для анимации)
- [ ] NUM_FRAMES установлен (16-24)

## Для анимации (важно!):
- [ ] В промпте есть описание движения:
  - "turning head"
  - "blinking eyes"
  - "hair flowing"
  - "camera movement"
- [ ] В negative промпте: "static, frozen, choppy"

---

# 🎨 Шаблоны промптов

## Шаблон 1: Персонаж
```python
PROMPT = "[персонаж] [действие: turning/blinking/smiling], [описание: волосы/одежда], [стиль: anime/realistic], smooth motion, cinematic lighting, 8k"

NEGATIVE_PROMPT = "static, frozen, choppy animation, blurry, deformed, low quality"

# Пример:
PROMPT = "gojo satoru turning head slowly and blinking, white spiky hair flowing, black blindfold, confident expression, anime style, smooth motion, cinematic lighting, 8k"
```

## Шаблон 2: Природа
```python
PROMPT = "[объект] [движение: falling/flowing/swaying], [погода/освещение], [стиль], smooth motion, cinematic"

NEGATIVE_PROMPT = "static, frozen, low quality, blurry"

# Пример:
PROMPT = "cherry blossom tree, pink petals falling gently, wind blowing branches, golden hour lighting, spring atmosphere, smooth motion, cinematic nature style, 8k"
```

## Шаблон 3: Камера движется
```python
PROMPT = "[сцена] [движение камеры: dolly/pan/zoom], [детали], cinematic camera movement, smooth motion"

NEGATIVE_PROMPT = "static camera, frozen, choppy, shaky"

# Пример:
PROMPT = "cyberpunk street at night, neon signs glowing, camera slowly dollying forward, rain falling, wet reflections, blade runner style, smooth camera movement, cinematic, 8k"
```

---

# ⚙️ Готовые конфиги

## Конфиг A: Вертикальное для Shorts
```python
WIDTH = 512
HEIGHT = 768
USE_ANIMATEDIFF = True
NUM_FRAMES = 16
USE_RIFE = True
RIFE_EXP = 4
STEPS = 25
FPS = 8

# Результат: 30+ сек вертикального видео
# Время: ~15 мин
```

## Конфиг B: Горизонтальное для YouTube
```python
WIDTH = 768
HEIGHT = 512
USE_ANIMATEDIFF = True
NUM_FRAMES = 16
USE_RIFE = True
RIFE_EXP = 4
STEPS = 25
FPS = 8

# Результат: 30+ сек горизонтального видео
# Время: ~15 мин
```

## Конфиг C: Быстрый тест
```python
WIDTH = 512
HEIGHT = 512
USE_ANIMATEDIFF = True
NUM_FRAMES = 16
USE_RIFE = False
STEPS = 20
FPS = 8

# Результат: ~2 сек анимация для проверки промпта
# Время: ~7 мин
```

## Конфиг D: Максимальное качество
```python
WIDTH = 512
HEIGHT = 768
USE_ANIMATEDIFF = True
NUM_FRAMES = 24
USE_RIFE = True
RIFE_EXP = 4
STEPS = 30
FPS = 8

# Результат: ~45 сек супер-плавного видео
# Время: ~20 мин
```

---

# 🔍 Troubleshooting

## Проблема: "Model not found"
**Причина:** Неверный путь к датасету  
**Решение:** 
```python
# Проверь имя датасета в Kaggle
DATASET_DIR = "/kaggle/input/ИМЯ-ТВОЕГО-ДАТАСЕТА"
```

## Проблема: "Out of memory"
**Причина:** Слишком большое разрешение  
**Решение:**
```python
WIDTH = 384  # вместо 512
HEIGHT = 576  # вместо 768
NUM_FRAMES = 12  # вместо 16
```

## Проблема: Видео "дергается"
**Причина:** Используются статичные кадры  
**Решение:**
```python
USE_ANIMATEDIFF = True  # включить анимацию
```

## Проблема: Очень медленно
**Причина:** Включен RIFE  
**Решение для теста:**
```python
USE_RIFE = False  # отключить RIFE
# Получишь короткое видео, но быстро
```

## Проблема: Персонаж не двигается
**Причина:** Нет описания движения в промпте  
**Решение:**
```python
# Было:
PROMPT = "gojo satoru, anime style"

# Стало:
PROMPT = "gojo satoru turning head slowly, blinking, smooth motion, anime style"
```

## Проблема: "RIFE не создал видео"
**Причина:** Ошибка в RIFE  
**Автоматическое решение:** Скрипт создаст базовое видео без RIFE

---

# 📈 Оптимизация качества

## Для лучшей анимации:

1. **Описывай КОНКРЕТНОЕ движение:**
   - ✅ "turning head 30 degrees to the left"
   - ❌ "moving"

2. **Одно движение за раз:**
   - ✅ "blinking slowly"
   - ❌ "jumping, running, dancing" (слишком много)

3. **Используй "smooth motion":**
   - Добавь в конец промпта

4. **Negative промпт для анимации:**
   - Обязательно: "static, frozen, choppy animation"

5. **Больше steps для анимации:**
   - Минимум 25 (вместо 20)

6. **Комбинация AnimateDiff + RIFE:**
   - Максимальная плавность

---

# 🎓 Продвинутые техники

## Контроль скорости движения:
```python
PROMPT = "... turning head SLOWLY ..."  # медленное движение
PROMPT = "... quick blink ..."          # быстрое движение
```

## Несколько объектов:
```python
PROMPT = "gojo satoru turning head, WHILE hair flowing in wind, AND eyes blinking"
# "WHILE" и "AND" помогают связать движения
```

## Направление движения:
```python
PROMPT = "... turning from left to right ..."
PROMPT = "... camera panning from right to left ..."
```

## Сложная анимация (требует больше кадров):
```python
NUM_FRAMES = 24  # вместо 16
STEPS = 30       # вместо 25
```

---

**Готово! Используй эти шаблоны и создавай крутые видео! 🎬✨**

