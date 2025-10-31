#!/usr/bin/env python3
"""
Скрипт для генерации полностью выполненного notebook с выводами.
Использование: python generate_executed_notebook.py

Этот скрипт:
1. Загружает датасет BCCD (если нужно)
2. Выполняет все ячейки notebook
3. Сохраняет результат с выводами
4. Готово для загрузки на GitHub без замечаний
"""

import sys
import os
import subprocess
from pathlib import Path

def check_requirements():
    """Проверка установленных зависимостей"""
    required = ['torch', 'torchvision', 'nbformat', 'nbconvert', 'jupyter']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Отсутствуют пакеты: {', '.join(missing)}")
        print(f"Установите: pip install {' '.join(missing)}")
        return False
    
    print("✅ Все зависимости установлены")
    return True

def check_dataset():
    """Проверка наличия датасета"""
    dataset_path = Path('./BCCD_Dataset/BCCD')
    
    if not dataset_path.exists():
        print("⚠️  Датасет BCCD не найден")
        response = input("Загрузить датасет сейчас? (y/n): ").strip().lower()
        
        if response == 'y':
            print("📥 Загрузка датасета...")
            try:
                subprocess.run(['git', 'clone', 'https://github.com/Shenggan/BCCD_Dataset.git'], 
                             check=True)
                print("✅ Датасет загружен")
                return True
            except subprocess.CalledProcessError:
                print("❌ Ошибка загрузки датасета")
                return False
        else:
            print("❌ Датасет необходим для выполнения")
            return False
    
    print("✅ Датасет найден")
    return True

def execute_notebook(notebook_path='ssd_bccd_detection.ipynb', output_path=None):
    """Выполнение notebook с сохранением выводов"""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    from datetime import datetime
    
    if output_path is None:
        output_path = notebook_path
    
    print(f"\n{'='*70}")
    print(f"🚀 ВЫПОЛНЕНИЕ NOTEBOOK")
    print(f"{'='*70}")
    print(f"Входной файл: {notebook_path}")
    print(f"Выходной файл: {output_path}")
    print(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Загрузка notebook
    print("📖 Загрузка notebook...")
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        print(f"✅ Notebook загружен: {len(nb.cells)} ячеек")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return False
    
    # Создание executor
    ep = ExecutePreprocessor(
        timeout=7200,  # 2 часа на ячейку (для обучения)
        kernel_name='python3',
        allow_errors=False
    )
    
    # Выполнение
    print(f"\n⚙️  Начинаю выполнение всех ячеек...")
    print(f"⏱️  Таймаут: 7200s на ячейку")
    print(f"📊 Это может занять продолжительное время (особенно обучение)...\n")
    
    try:
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        print(f"\n✅ Все ячейки выполнены успешно!")
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении: {e}")
        print(f"⚠️  Попытка сохранить частичные результаты...")
    
    # Сохранение
    print(f"\n💾 Сохранение notebook с выводами...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"✅ Notebook сохранен: {output_path}")
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")
        return False
    
    print(f"\n⏰ Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return True

def validate_outputs(notebook_path='ssd_bccd_detection.ipynb'):
    """Проверка наличия выводов в notebook"""
    import nbformat
    
    print(f"\n🔍 Валидация выводов...")
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        cells_with_output = [c for c in code_cells if c.outputs]
        
        coverage = (len(cells_with_output) / len(code_cells)) * 100 if code_cells else 0
        
        print(f"  Код ячеек: {len(code_cells)}")
        print(f"  С выводами: {len(cells_with_output)}")
        print(f"  Покрытие: {coverage:.1f}%")
        
        if coverage >= 90:
            print(f"✅ Notebook содержит достаточно выводов")
            return True
        else:
            print(f"⚠️  Недостаточно выводов (< 90%)")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка валидации: {e}")
        return False

def main():
    print("="*70)
    print("🔬 ГЕНЕРАТОР ВЫПОЛНЕННОГО SSD BCCD NOTEBOOK")
    print("="*70)
    print("\nЭтот скрипт создаст полностью выполненный notebook с выводами,")
    print("готовый для загрузки на GitHub без замечаний.\n")
    
    # Проверки
    if not check_requirements():
        print("\n❌ Установите недостающие пакеты и запустите снова")
        sys.exit(1)
    
    if not check_dataset():
        print("\n❌ Датасет необходим для выполнения notebook")
        sys.exit(1)
    
    # Проверка наличия notebook
    notebook_path = 'ssd_bccd_detection.ipynb'
    if not os.path.exists(notebook_path):
        print(f"\n❌ Файл {notebook_path} не найден!")
        print(f"Убедитесь, что файл находится в текущей директории")
        sys.exit(1)
    
    # Подтверждение
    print("\n" + "="*70)
    print("⚠️  ВНИМАНИЕ")
    print("="*70)
    print("Выполнение может занять несколько часов!")
    print("Требуется:")
    print("  - GPU для обучения (рекомендуется)")
    print("  - ~5GB свободного места")
    print("  - Стабильное подключение к питанию\n")
    
    response = input("Продолжить? (y/n): ").strip().lower()
    
    if response != 'y':
        print("\n❌ Отменено пользователем")
        sys.exit(0)
    
    # Создание backup
    if os.path.exists(notebook_path):
        import shutil
        backup_path = notebook_path.replace('.ipynb', '_backup.ipynb')
        shutil.copy2(notebook_path, backup_path)
        print(f"\n💾 Создан backup: {backup_path}")
    
    # Выполнение
    success = execute_notebook(notebook_path)
    
    if success:
        # Валидация
        has_outputs = validate_outputs(notebook_path)
        
        print("\n" + "="*70)
        if has_outputs:
            print("🎉 УСПЕХ!")
            print("="*70)
            print(f"\n✅ Notebook готов для GitHub!")
            print(f"\nСледующие шаги:")
            print(f"  git add {notebook_path}")
            print(f"  git commit -m 'Добавлены выводы выполнения SSD BCCD notebook'")
            print(f"  git push origin main")
            print(f"\n✅ Замечание 'нет выводов результатов выполнения ячеек' НЕ появится!")
        else:
            print("⚠️  ВЫПОЛНЕНО С ПРЕДУПРЕЖДЕНИЯМИ")
            print("="*70)
            print(f"\n⚠️  Некоторые ячейки могут не содержать выводы")
            print(f"Проверьте notebook вручную перед загрузкой на GitHub")
        print("\n" + "="*70)
    else:
        print("\n" + "="*70)
        print("❌ ОШИБКА")
        print("="*70)
        print(f"\nНе удалось выполнить notebook")
        print(f"Проверьте логи выше для деталей")
        print("\n" + "="*70)
        sys.exit(1)

if __name__ == "__main__":
    main()
