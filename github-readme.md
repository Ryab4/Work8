# SSD Object Detection для BCCD Dataset

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Полная реализация **Single Shot MultiBox Detector (SSD300)** для детекции клеток крови на датасете BCCD (Blood Cell Count and Detection).

## 🎯 Решаемая задача

Детекция трех типов клеток крови на медицинских изображениях:
- **WBC** (White Blood Cells) - Лейкоциты
- **RBC** (Red Blood Cells) - Эритроциты  
- **Platelets** - Тромбоциты

## ✨ Особенности

- ✅ **Полная реализация SSD300** с VGG-16 backbone
- ✅ **MultiBox Loss** с hard negative mining
- ✅ **Prior boxes** на 6 масштабах (8732 anchors)
- ✅ **Все выводы выполнения сохранены** - готово к использованию
- ✅ **Визуализации** на каждом этапе обучения и детекции
- ✅ **Подробные комментарии** и документация

## 📊 Датасет

[BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset) - 364 изображения клеток крови (640×480) с аннотациями в формате Pascal VOC.

**Распределение:**
- Train: ~260 изображений
- Validation: ~70 изображений
- Test: ~34 изображения

## 🚀 Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/Ryab4/Work8.git
cd Work8
```

### 2. Установка зависимостей

```bash
pip install torch torchvision pillow matplotlib opencv-python tqdm lxml numpy
```

### 3. Загрузка датасета

```bash
git clone https://github.com/Shenggan/BCCD_Dataset.git
```

### 4. Запуск notebook

```bash
jupyter notebook ssd_bccd_detection.ipynb
```

**ВАЖНО:** Notebook уже содержит все выводы выполнения! Вы можете сразу просмотреть результаты без необходимости длительного обучения.

### 5. Для нового обучения

Если хотите обучить модель заново:

```bash
# В Jupyter: Kernel -> Restart & Run All
# Обучение займет 2-4 часа на GPU
```

## 📁 Структура проекта

```
.
├── ssd_bccd_detection.ipynb      # 🎓 Основной notebook с ВЫВОДАМИ
├── execute_ssd_notebook.py       # 🤖 Скрипт автоматизации
├── README.md                      # 📖 Документация
└── checkpoints/                   # 💾 Сохраненные модели (после обучения)
    └── best_model.pth
```

## 🏗️ Архитектура SSD300

```
Input (300×300) 
    ↓
VGG-16 Base Network
    ↓
Auxiliary Convolutions (multi-scale features)
    ↓
Prediction Convolutions
    ├─ Localization (bbox offsets)
    └─ Classification (class scores)
    ↓
Outputs:
    ├─ 8732 predicted boxes
    └─ Confidence scores for 4 classes
```

### Prior Boxes

- **Feature maps:** 38×38, 19×19, 10×10, 5×5, 3×3, 1×1
- **Scales:** 0.1, 0.2, 0.375, 0.55, 0.725, 0.9
- **Aspect ratios:** [2], [2, 3], [2, 3], [2, 3], [2], [2]
- **Total anchors:** 8732

## 📈 Результаты

### График обучения
Визуализация train/validation loss по эпохам (см. в notebook)

### Примеры детекций
Визуализация результатов на тестовых изображениях с:
- Bounding boxes с цветовой кодировкой по классам
- Confidence scores для каждой детекции
- Сравнение Ground Truth vs Predictions

## 🎓 Теория и референсы

Реализация основана на:

1. **[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)** - оригинальная статья
2. **[PyTorch Tutorial to Object Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)** - имплементация
3. **[D2L.ai SSD Chapter](https://d2l.ai/chapter_computer-vision/ssd.html)** - теоретическая часть

## 💻 Требования

- **Python:** 3.7+
- **PyTorch:** 1.7+ (с CUDA для GPU)
- **RAM:** 8GB+
- **GPU:** Рекомендуется CUDA-compatible (обучение на CPU займет ~10-20 часов)
- **Место на диске:** ~5GB

## 🔧 Конфигурация

Основные параметры в notebook (ячейка 2):

```python
IMAGE_SIZE = 300          # Размер входного изображения
BATCH_SIZE = 8            # Размер батча
NUM_EPOCHS = 50           # Количество эпох
LEARNING_RATE = 1e-3      # Learning rate
IOU_THRESHOLD = 0.5       # IoU для matching
NMS_THRESHOLD = 0.45      # NMS threshold
```

## 📝 Использование для inference

```python
# Загрузка обученной модели
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# Детекция на новом изображении
det_boxes, det_labels, det_scores = detect_objects(
    model, images, 
    min_score=0.2, 
    max_overlap=0.45
)
```

## 🎯 Результаты детекции

После обучения модель способна:
- Детектировать множественные объекты на изображении
- Различать 3 класса клеток крови
- Предсказывать bounding boxes с высокой точностью
- Фильтровать ложные срабатывания через NMS

## 🐛 Решение проблем

### GPU Out of Memory
```python
BATCH_SIZE = 4  # Уменьшите batch size
```

### Kernel died
```python
NUM_EPOCHS = 10  # Уменьшите количество эпох для теста
```

### Dataset not found
```bash
git clone https://github.com/Shenggan/BCCD_Dataset.git
```

## 📚 Что реализовано в notebook

### Этапы:

1. ✅ Загрузка и анализ датасета
2. ✅ Создание prior boxes
3. ✅ Реализация SSD300 архитектуры
4. ✅ MultiBox Loss с hard negative mining
5. ✅ Training loop с validation
6. ✅ Детекция с NMS
7. ✅ Визуализация результатов
8. ✅ Сравнение с Ground Truth

### Компоненты:

- VGG-16 base network (pretrained на ImageNet)
- Auxiliary convolutions для multi-scale features
- Prediction heads (localization + classification)
- Prior boxes generator
- MultiBox Loss (Smooth L1 + Cross-Entropy)
- Non-Maximum Suppression
- Визуализация и метрики

## 🤝 Вклад

Contributions, issues и feature requests приветствуются!

## 📄 Лицензия

MIT License - см. [LICENSE](LICENSE)

## 👤 Автор

Реализация SSD для детекции клеток крови на датасете BCCD.

## 🙏 Благодарности

- [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset) - за датасет
- [sgrvinod](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) - за PyTorch имплементацию SSD
- [D2L.ai](https://d2l.ai/) - за образовательные материалы

---

**⚠️ Примечание:** Notebook уже содержит все выводы выполнения. Вы можете просмотреть результаты без необходимости полного переобучения модели.
