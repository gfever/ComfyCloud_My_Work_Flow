# 🎬 ComfyCloud Video Generation - AnimateDiff + RIFE

Генерация плавных анимированных видео на Kaggle GPU (бесплатно!) используя:
- **AnimateDiff** - настоящая анимация с естественными движениями
- **RIFE** - интерполяция для супер-плавного видео
- **Stable Diffusion XL** - качественная генерация

## 🚀 Быстрый старт

### Для анимации (рекомендуется):
1. Открой **`kaggle_animated.ipynb`** или **`kaggle_simple.ipynb`**
2. Измени промпт:
   ```python
   PROMPT = "gojo satoru turning head, blinking, smooth motion"
   ```
3. Запусти на Kaggle GPU
4. Получи 30+ секунд плавного видео!

📖 **[Подробная инструкция → QUICKSTART_RU.md](QUICKSTART_RU.md)**

---

## 📁 Файлы проекта

### ⭐⭐⭐ `kaggle_animated.ipynb` 
**Лучший для анимации!** AnimateDiff + RIFE = максимальная плавность
- Настоящая анимация (не склейка кадров)
- Естественные движения персонажей
- ~15-20 минут на выполнение

### ⭐⭐ `kaggle_simple.ipynb`
**Универсальный!** Два режима: AnimateDiff ИЛИ статичные кадры
- Переключение одной переменной
- Гибкость в настройках
- ~8-17 минут на выполнение

### 📚 Документация
- **`QUICKSTART_RU.md`** - пошаговая инструкция для новичков
- **`README_RU.md`** - полная документация, примеры, настройки

---

## 🎨 Пример промпта

```python
# Для анимации (AnimateDiff)
PROMPT = "cinematic portrait of gojo satoru, turning head slowly, confident smirk, white hair flowing in wind, black blindfold, anime style, smooth motion, 8k"
NEGATIVE_PROMPT = "static, frozen, choppy animation, blurry, low quality"

# Настройки
USE_ANIMATEDIFF = True  # Настоящая анимация
NUM_FRAMES = 16         # Кадров
USE_RIFE = True         # Интерполяция для плавности
WIDTH = 512             # Вертикальное для shorts
HEIGHT = 768
```

---

## 📊 Сравнение методов

| Метод | Время | Качество | Лучше для |
|-------|-------|----------|-----------|
| **AnimateDiff + RIFE** | ~15-20 мин | ⭐⭐⭐⭐⭐⭐ | Персонажи, лица, движения |
| **AnimateDiff** | ~8-12 мин | ⭐⭐⭐⭐⭐ | Быстрая анимация |
| **Статичные + RIFE** | ~10-15 мин | ⭐⭐⭐ | Пейзажи, переходы |

---

## 💡 Ключевые особенности

✅ **Без ComfyUI** - быстрая установка, меньше зависимостей  
✅ **AnimateDiff** - настоящая анимация с пониманием движения  
✅ **RIFE интерполяция** - 16→256 кадров для плавности  
✅ **Простая настройка** - все параметры в начале файла  
✅ **Бесплатно** - работает на Kaggle GPU  
✅ **Быстро** - 8-20 минут до готового видео  

---

## 🎯 Что можно создать?

- 📱 YouTube Shorts / TikTok / Reels (вертикальное видео)
- 🎬 Киношные сцены с движением камеры
- 👤 Анимированные портреты персонажей
- 🌸 Природа в движении (падающие лепестки, ветер)
- 🌃 Киберпанк сцены с неоновыми огнями

---

## 📖 Документация

- **[QUICKSTART_RU.md](QUICKSTART_RU.md)** - начни отсюда!
- **[README_RU.md](README_RU.md)** - полное руководство

---

## 🔗 Технологии

- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [AnimateDiff](https://github.com/guoyww/AnimateDiff)
- [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
- [Diffusers](https://github.com/huggingface/diffusers)

---

**Создано для простой генерации качественных анимированных видео на Kaggle!**

---

# ComfyCloud_My_Work_Flow
======================

Короткая инструкция по использованию ноутбука `ComfyCloud_My_Work_Flow.ipynb`.

1) Запуск в Google Colab (рекомендуется для туннелирования и GPU):
   - Откройте ноутбук через кнопку "Open in Colab" в начале файла или по ссылке:
     https://colab.research.google.com/github/gfever/ComfyCloud_My_Work_Flow/blob/main/ComfyCloud_My_Work_Flow.ipynb
   - Выполните ячейки сверху вниз. В финальной ячейке скрипт автоматически попытается установить `cloudflared` в Colab (dpkg) и создать публичный туннель; в случае успеха в выводе появится URL вида `https://...trycloudflare.com`.

2) Запуск локально на Windows (если вы не хотите Colab):
   - Не нужно запускать финальную ячейку, если хотите работать локально.
   - Откройте PowerShell/командную строку в папке `AI/ComfyUI` и выполните:
     ```powershell
     python main.py --dont-print-server
     ```
   - Для доступа откройте в браузере: http://127.0.0.1:8188

3) Очистка outputs перед коммитом (уже настроено в этом репозитории):
   - В репозитории добавлен `.gitattributes` и установлен `nbstripout` hook — при коммите ноутбуков выводы будут автоматически удаляться.

4) Работа с GitHub (push):
   - Если у вас есть удалённый репозиторий и настроен remote `origin`, выполните:
     ```powershell
     git push origin HEAD
     ```
   - GitHub требует Personal Access Token (PAT) для HTTPS push (пароли не поддерживаются). Если хотите, помогу с инструкцией по созданию PAT.

5) Примечания по cloudflared и Colab:
   - Автоматическая установка `cloudflared` выполняется только в Colab/Linux (через dpkg). На Windows туннелирование автоматически не будет запущено — установите cloudflared вручную, если нужно.

Если хотите, могу:
- Выполнить `git push` за вас (вам потребуется ввести учётные данные в терминале), либо
- Подсказать, как создать PAT и настроить push без запроса пароля.

Автор: изменения подготовлены автоматически (локальная очистка outputs + nbstripout + README).

