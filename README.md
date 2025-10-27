# SSD Object Detection для BCCD Dataset

Реализация Single Shot MultiBox Detector (SSD300) для детекции клеток крови на датасете BCCD (Blood Cell Count and Detection).

## Описание проекта

Этот проект реализует архитектуру SSD300 для задачи детекции трех типов клеток крови:
- **WBC** (White Blood Cells) - Лейкоциты
- **RBC** (Red Blood Cells) - Эритроциты  
- **Platelets** - Тромбоциты

## Структура проекта

```
.
├── config.py              # Конфигурация и гиперпараметры
├── model.py               # Архитектура SSD300
├── utils.py               # Вспомогательные функции (prior boxes, loss, NMS)
├── dataset.py             # Dataset класс и трансформации
├── train.py               # Скрипт обучения
├── inference.py           # Скрипт для inference
├── ssd_bccd_detection.ipynb  # Jupyter notebook с полным pipeline
├── requirements.txt       # Зависимости
└── README.md             # Документация
```

## Установка

### 1. Клонирование репозитория

```bash
git clone <your-repo-url>
cd ssd-bccd-detection
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Загрузка датасета BCCD

```bash
git clone https://github.com/Shenggan/BCCD_Dataset.git
```

Структура датасета должна быть следующей:
```
BCCD_Dataset/
└── BCCD/
    ├── Annotations/     # XML аннотации в формате Pascal VOC
    │   ├── BloodImage_00000.xml
    │   └── ...
    └── JPEGImages/      # Изображения
        ├── BloodImage_00000.jpg
        └── ...
```

## Использование

### Обучение модели

Для запуска обучения выполните:

```bash
python train.py
```

Параметры обучения можно изменить в файле `config.py`:
- `BATCH_SIZE` - размер батча (по умолчанию 8)
- `NUM_EPOCHS` - количество эпох (по умолчанию 100)
- `LEARNING_RATE` - начальный learning rate (по умолчанию 1e-3)

Модель будет сохранена в директории `checkpoints/`:
- `best_model.pth` - лучшая модель по validation loss
- `ssd300_epoch_N.pth` - checkpoints каждые 5 эпох

### Inference на новых изображениях

```bash
python inference.py --image path/to/image.jpg --model checkpoints/best_model.pth --output result.jpg
```

Параметры:
- `--image` - путь к изображению для детекции
- `--model` - путь к обученной модели
- `--output` - путь для сохранения результата
- `--min_score` - минимальный confidence threshold (по умолчанию 0.2)
- `--max_overlap` - IoU threshold для NMS (по умолчанию 0.45)

### Jupyter Notebook

Для интерактивной работы используйте `ssd_bccd_detection.ipynb`:

```bash
jupyter notebook ssd_bccd_detection.ipynb
```

## Архитектура SSD

### Основные компоненты:

1. **Base Network (VGG-16)**
   - Предобученная VGG-16 сеть для извлечения признаков
   - Модифицированные FC слои заменены на сверточные (conv6, conv7)

2. **Auxiliary Convolutions**
   - Дополнительные сверточные слои (conv8-conv11)
   - Создают feature maps на разных масштабах

3. **Prediction Convolutions**
   - Localization head - предсказание offsets для bounding boxes
   - Classification head - предсказание классов объектов

4. **Prior Boxes (Anchors)**
   - 8732 предопределенных anchor boxes
   - 6 feature maps разных размеров: 38×38, 19×19, 10×10, 5×5, 3×3, 1×1

### Loss Function

MultiBox Loss = Localization Loss + Confidence Loss

- **Localization Loss**: Smooth L1 loss для регрессии bounding boxes
- **Confidence Loss**: Cross-entropy loss с hard negative mining (соотношение 3:1)

## Конфигурация

Основные параметры в `config.py`:

```python
# Модель
IMAGE_SIZE = 300  # SSD300
NUM_CLASSES = 4   # background + 3 класса клеток

# Обучение
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Prior boxes
FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
SCALES = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]

# Детекция
IOU_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.01
```

## Особенности реализации

1. **Data Augmentation**
   - Random horizontal flip
   - Random color jitter (brightness, contrast, saturation)
   - Normalization с ImageNet mean/std

2. **Training Tricks**
   - Gradient clipping для стабильности обучения
   - Learning rate schedule с мультистеп декеем
   - Разные learning rates для bias и weight параметров
   - Hard negative mining для balance классов

3. **Inference Optimization**
   - Non-Maximum Suppression для удаления дубликатов
   - Confidence threshold filtering
   - Top-K selection

## Требования к системе

- Python 3.7+
- CUDA-compatible GPU (рекомендуется)
- 8GB+ RAM
- 5GB+ свободного места на диске

## Ссылки

- **BCCD Dataset**: https://github.com/Shenggan/BCCD_Dataset
- **SSD Paper**: https://arxiv.org/abs/1512.02325
- **PyTorch Tutorial**: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
- **D2L SSD Chapter**: https://d2l.ai/chapter_computer-vision/ssd.html

## Автор

Реализация SSD для детекции клеток крови на датасете BCCD.

## Лицензия

MIT License - датасет BCCD и код открыты для использования.
