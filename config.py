# config.py - Конфигурация и гиперпараметры для SSD детекции клеток крови

import torch

# Устройство для обучения
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Пути к данным
DATA_DIR = 'BCCD_Dataset/BCCD'  # Корневая директория датасета
TRAIN_IMAGES_DIR = 'BCCD_Dataset/BCCD/JPEGImages'
TRAIN_ANNOTATIONS_DIR = 'BCCD_Dataset/BCCD/Annotations'

# Параметры изображений
IMAGE_SIZE = 300  # SSD300
IMAGE_CHANNELS = 3

# Классы датасета BCCD (+ background)
CLASSES = ['__background__', 'WBC', 'RBC', 'Platelets']
NUM_CLASSES = len(CLASSES)

# Маппинг классов
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}

# Гиперпараметры обучения
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Learning rate schedule
LR_DECAY_EPOCHS = [60, 80]  # Эпохи для уменьшения LR
LR_DECAY_FACTOR = 0.1  # Коэффициент уменьшения

# Loss parameters
ALPHA = 1.0  # Вес для localization loss

# Prior boxes configuration
FEATURE_MAPS = [38, 19, 10, 5, 3, 1]  # Размеры feature maps для SSD300
ASPECT_RATIOS = {
    38: [1, 2, 0.5],
    19: [1, 2, 0.5, 3, 1/3],
    10: [1, 2, 0.5, 3, 1/3],
    5: [1, 2, 0.5, 3, 1/3],
    3: [1, 2, 0.5],
    1: [1, 2, 0.5]
}
SCALES = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]

# Matching and NMS parameters
IOU_THRESHOLD = 0.5  # Для matching priors к ground truth
NMS_THRESHOLD = 0.45  # Для Non-Maximum Suppression
CONFIDENCE_THRESHOLD = 0.01  # Минимальный confidence для детекции
TOP_K = 200  # Максимальное количество детекций перед NMS
MAX_DETECTIONS = 100  # Максимальное количество финальных детекций

# Data augmentation parameters
MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
STD = [0.229, 0.224, 0.225]   # ImageNet std

# Checkpoint parameters
CHECKPOINT_DIR = 'checkpoints'
SAVE_EVERY = 5  # Сохранять checkpoint каждые N эпох
BEST_MODEL_PATH = 'best_model.pth'

# Visualization
VIS_SAMPLES = 5  # Количество примеров для визуализации

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
