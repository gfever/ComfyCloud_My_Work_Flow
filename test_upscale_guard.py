# Тест для проверки защитной части ячейки апскейла
import os
# Эмулируем запуск ячейки в чистом глобальном пространстве
g = {}
code = '''
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

print('RESULTS:', WORKSPACE, FRAMES_DIR, DATASET_DIR)
'''

exec(code, g)
print('\nScript finished.')

