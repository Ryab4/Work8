# dataset.py - Dataset класс для загрузки BCCD данных

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import config
from utils import parse_voc_annotation
import random


class BCCDDataset(Dataset):
    """
    Dataset класс для BCCD (Blood Cell Count and Detection).
    """
    def __init__(self, images_dir, annotations_dir, split='train', transform=None):
        """
        Args:
            images_dir: путь к директории с изображениями
            annotations_dir: путь к директории с аннотациями
            split: 'train' или 'test'
            transform: трансформации для изображений
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.split = split
        self.transform = transform
        
        # Получаем список файлов
        self.image_files = []
        self.annotation_files = []
        
        for img_file in sorted(os.listdir(images_dir)):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(images_dir, img_file)
                ann_file = img_file.replace('.jpg', '.xml').replace('.jpeg', '.xml').replace('.png', '.xml')
                ann_path = os.path.join(annotations_dir, ann_file)
                
                if os.path.exists(ann_path):
                    self.image_files.append(img_path)
                    self.annotation_files.append(ann_path)
        
        print(f"Загружено {len(self.image_files)} изображений для {split}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Args:
            idx: индекс элемента
        Returns:
            image: Tensor размера (3, 300, 300)
            boxes: Tensor размера (n_objects, 4) в формате (xmin, ymin, xmax, ymax)
            labels: Tensor размера (n_objects,) с индексами классов
        """
        # Загружаем изображение
        image = Image.open(self.image_files[idx]).convert('RGB')
        
        # Парсим аннотацию
        boxes, label_names, orig_size = parse_voc_annotation(self.annotation_files[idx])
        
        # Конвертируем имена классов в индексы
        labels = [config.CLASS_TO_IDX[name] for name in label_names]
        
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        
        # Применяем трансформации
        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)
        
        return image, boxes, labels
    
    def collate_fn(self, batch):
        """
        Функция для collate в DataLoader.
        Args:
            batch: список из (image, boxes, labels)
        Returns:
            images: Tensor (batch_size, 3, 300, 300)
            boxes: List из Tensors
            labels: List из Tensors
        """
        images = []
        boxes = []
        labels = []
        
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
        
        images = torch.stack(images, dim=0)
        
        return images, boxes, labels


class TrainTransform:
    """
    Трансформации для обучающего датасета с аугментацией.
    """
    def __init__(self, size=300):
        self.size = size
        self.mean = config.MEAN
        self.std = config.STD
    
    def __call__(self, image, boxes, labels):
        """
        Args:
            image: PIL Image
            boxes: Tensor (n_objects, 4) в нормализованных координатах
            labels: Tensor (n_objects,)
        Returns:
            image: Tensor (3, 300, 300)
            boxes: Tensor (n_objects, 4)
            labels: Tensor (n_objects,)
        """
        # Конвертируем в Tensor
        image = T.ToTensor()(image)
        
        # Data augmentation
        # 1. Random brightness, contrast, saturation
        if random.random() < 0.5:
            image = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)(image)
        
        # 2. Random horizontal flip
        if random.random() < 0.5:
            image = T.functional.hflip(image)
            boxes_np = boxes.numpy()
            boxes_np[:, [0, 2]] = 1 - boxes_np[:, [2, 0]]
            boxes = torch.FloatTensor(boxes_np)
        
        # Resize
        image = T.Resize((self.size, self.size))(image)
        
        # Normalize
        image = T.Normalize(mean=self.mean, std=self.std)(image)
        
        return image, boxes, labels


class TestTransform:
    """
    Трансформации для тестового датасета без аугментации.
    """
    def __init__(self, size=300):
        self.size = size
        self.mean = config.MEAN
        self.std = config.STD
    
    def __call__(self, image, boxes, labels):
        """
        Args:
            image: PIL Image
            boxes: Tensor (n_objects, 4)
            labels: Tensor (n_objects,)
        Returns:
            image: Tensor (3, 300, 300)
            boxes: Tensor (n_objects, 4)
            labels: Tensor (n_objects,)
        """
        # Конвертируем в Tensor
        image = T.ToTensor()(image)
        
        # Resize
        image = T.Resize((self.size, self.size))(image)
        
        # Normalize
        image = T.Normalize(mean=self.mean, std=self.std)(image)
        
        return image, boxes, labels


def create_dataloaders(train_split=0.8):
    """
    Создание DataLoader'ов для обучения и валидации.
    Args:
        train_split: доля данных для обучения
    Returns:
        train_loader: DataLoader для обучения
        val_loader: DataLoader для валидации
    """
    from torch.utils.data import DataLoader, random_split
    
    # Создаем полный датасет
    full_dataset = BCCDDataset(
        images_dir=config.TRAIN_IMAGES_DIR,
        annotations_dir=config.TRAIN_ANNOTATIONS_DIR,
        split='train',
        transform=None
    )
    
    # Разделяем на train/val
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    # Устанавливаем трансформации
    train_dataset.dataset.transform = TrainTransform()
    val_dataset.dataset.transform = TestTransform()
    
    # Создаем DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=full_dataset.collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=full_dataset.collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
