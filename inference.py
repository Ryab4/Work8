# inference.py - Скрипт для inference на новых изображениях

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import config
from model import SSD300
from dataset import TestTransform
import torchvision.transforms as T


def load_model(checkpoint_path, device):
    """
    Загрузка обученной модели.
    Args:
        checkpoint_path: путь к checkpoint
        device: устройство для inference
    Returns:
        model: загруженная модель
    """
    model = SSD300(n_classes=config.NUM_CLASSES)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    return model


def detect(image_path, model, device, min_score=0.2, max_overlap=0.45, top_k=200):
    """
    Детекция объектов на изображении.
    Args:
        image_path: путь к изображению
        model: обученная модель
        device: устройство
        min_score: минимальный confidence threshold
        max_overlap: максимальный IoU для NMS
        top_k: максимальное количество детекций
    Returns:
        boxes: список детектированных boxes
        labels: список меток
        scores: список confidence scores
        original_image: оригинальное изображение
    """
    # Загружаем изображение
    original_image = Image.open(image_path).convert('RGB')
    orig_width, orig_height = original_image.size
    
    # Трансформация для модели
    transform = TestTransform(size=config.IMAGE_SIZE)
    
    # Dummy boxes и labels для трансформации
    dummy_boxes = torch.FloatTensor([[0, 0, 1, 1]])
    dummy_labels = torch.LongTensor([0])
    
    image, _, _ = transform(original_image, dummy_boxes, dummy_labels)
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Forward pass
    with torch.no_grad():
        predicted_locs, predicted_scores = model(image)
        
        # Detect objects
        det_boxes, det_labels, det_scores = model.detect_objects(
            predicted_locs, predicted_scores,
            min_score=min_score,
            max_overlap=max_overlap,
            top_k=top_k
        )
    
    # Результаты для первого (единственного) изображения в batch
    boxes = det_boxes[0].cpu()
    labels = det_labels[0].cpu()
    scores = det_scores[0].cpu()
    
    # Конвертируем нормализованные координаты в пиксели
    boxes[:, [0, 2]] *= orig_width
    boxes[:, [1, 3]] *= orig_height
    
    return boxes, labels, scores, original_image


def visualize_detection(image, boxes, labels, scores, save_path=None, show=True):
    """
    Визуализация детекций.
    Args:
        image: PIL Image
        boxes: Tensor (n_detections, 4) с координатами в пикселях
        labels: Tensor (n_detections,) с индексами классов
        scores: Tensor (n_detections,) с confidence scores
        save_path: путь для сохранения изображения
        show: показать изображение
    """
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(np.array(image))
    
    # Цвета для разных классов
    colors = {
        1: 'red',      # WBC
        2: 'blue',     # RBC
        3: 'green'     # Platelets
    }
    
    # Рисуем boxes
    for i in range(boxes.size(0)):
        box = boxes[i]
        label = labels[i].item()
        score = scores[i].item()
        
        if label == 0:  # Skip background
            continue
        
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        
        color = colors.get(label, 'yellow')
        
        # Рисуем прямоугольник
        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Добавляем текст
        class_name = config.IDX_TO_CLASS[label]
        text = f'{class_name}: {score:.2f}'
        ax.text(
            xmin, ymin - 5,
            text,
            bbox=dict(facecolor=color, alpha=0.5),
            fontsize=10,
            color='white'
        )
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f'Результат сохранен: {save_path}')
    
    if show:
        plt.show()
    
    plt.close()


def detect_and_visualize(image_path, model_path, save_path=None, 
                         min_score=0.2, max_overlap=0.45):
    """
    Детекция и визуализация объектов на изображении.
    Args:
        image_path: путь к изображению
        model_path: путь к модели
        save_path: путь для сохранения результата
        min_score: минимальный confidence threshold
        max_overlap: максимальный IoU для NMS
    """
    device = config.DEVICE
    
    # Загружаем модель
    print('Загрузка модели...')
    model = load_model(model_path, device)
    
    # Детекция
    print('Детекция объектов...')
    boxes, labels, scores, original_image = detect(
        image_path, model, device,
        min_score=min_score,
        max_overlap=max_overlap
    )
    
    # Визуализация
    print(f'Найдено объектов: {boxes.size(0)}')
    visualize_detection(original_image, boxes, labels, scores, save_path)
    
    # Вывод детекций
    for i in range(boxes.size(0)):
        label = labels[i].item()
        if label == 0:
            continue
        class_name = config.IDX_TO_CLASS[label]
        score = scores[i].item()
        print(f'  {class_name}: {score:.3f}')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SSD Inference на изображении')
    parser.add_argument('--image', type=str, required=True, help='Путь к изображению')
    parser.add_argument('--model', type=str, required=True, help='Путь к модели')
    parser.add_argument('--output', type=str, default='output.jpg', help='Путь для сохранения')
    parser.add_argument('--min_score', type=float, default=0.2, help='Минимальный confidence')
    parser.add_argument('--max_overlap', type=float, default=0.45, help='IoU threshold для NMS')
    
    args = parser.parse_args()
    
    detect_and_visualize(
        args.image,
        args.model,
        args.output,
        args.min_score,
        args.max_overlap
    )
