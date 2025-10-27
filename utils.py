# utils.py - Вспомогательные функции для SSD

import torch
import torch.nn.functional as F
import numpy as np
import math
from itertools import product
import xml.etree.ElementTree as ET
import config


def create_prior_boxes():
    """
    Создание prior boxes (anchor boxes) для всех feature maps.
    Returns:
        prior_boxes: Tensor размера (8732, 4) в формате (cx, cy, w, h)
    """
    prior_boxes = []
    
    for k, fmap_size in enumerate(config.FEATURE_MAPS):
        scale = config.SCALES[k]
        if k < len(config.SCALES) - 1:
            scale_next = config.SCALES[k + 1]
        else:
            scale_next = 1.0
        
        # Дополнительный prior с scale = sqrt(scale * scale_next)
        scale_extra = math.sqrt(scale * scale_next)
        
        aspect_ratios = config.ASPECT_RATIOS[fmap_size]
        
        for i, j in product(range(fmap_size), repeat=2):
            # Центр prior box
            cx = (j + 0.5) / fmap_size
            cy = (i + 0.5) / fmap_size
            
            # Prior boxes для разных aspect ratios
            for ar in aspect_ratios:
                w = scale * math.sqrt(ar)
                h = scale / math.sqrt(ar)
                prior_boxes.append([cx, cy, w, h])
            
            # Дополнительный prior с aspect ratio 1:1
            w = scale_extra
            h = scale_extra
            prior_boxes.append([cx, cy, w, h])
    
    prior_boxes = torch.FloatTensor(prior_boxes)
    prior_boxes.clamp_(0, 1)  # Ограничиваем значения в [0, 1]
    
    return prior_boxes


def cxcy_to_xy(boxes):
    """
    Конвертация из (cx, cy, w, h) в (xmin, ymin, xmax, ymax).
    Args:
        boxes: Tensor размера (n_boxes, 4)
    Returns:
        boxes_xy: Tensor размера (n_boxes, 4)
    """
    return torch.cat([boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:] / 2], 1)  # xmax, ymax


def xy_to_cxcy(boxes):
    """
    Конвертация из (xmin, ymin, xmax, ymax) в (cx, cy, w, h).
    Args:
        boxes: Tensor размера (n_boxes, 4)
    Returns:
        boxes_cxcy: Tensor размера (n_boxes, 4)
    """
    return torch.cat([(boxes[:, 2:] + boxes[:, :2]) / 2,  # cx, cy
                      boxes[:, 2:] - boxes[:, :2]], 1)  # w, h


def calculate_iou(boxes1, boxes2):
    """
    Расчет IoU между двумя наборами boxes.
    Args:
        boxes1: Tensor размера (n1, 4) в формате (xmin, ymin, xmax, ymax)
        boxes2: Tensor размера (n2, 4) в формате (xmin, ymin, xmax, ymax)
    Returns:
        iou: Tensor размера (n1, n2)
    """
    # Площади boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom
    
    wh = (rb - lt).clamp(min=0)  # w, h
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    # Union
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    
    return iou


def encode_boxes(gt_boxes, priors):
    """
    Кодирование ground truth boxes относительно prior boxes.
    Args:
        gt_boxes: Tensor размера (n_objects, 4) в формате (cx, cy, w, h)
        priors: Tensor размера (n_priors, 4) в формате (cx, cy, w, h)
    Returns:
        encoded: Tensor размера (n_priors, 4) с encoded offsets
    """
    # gcx = (gt_cx - prior_cx) / prior_w * 10
    # gcy = (gt_cy - prior_cy) / prior_h * 10
    # gw = log(gt_w / prior_w) * 5
    # gh = log(gt_h / prior_h) * 5
    
    gcxcy = (gt_boxes[:, :2] - priors[:, :2]) / priors[:, 2:] * 10
    gwh = torch.log(gt_boxes[:, 2:] / priors[:, 2:]) * 5
    
    return torch.cat([gcxcy, gwh], dim=1)


def decode_boxes(pred_offsets, priors):
    """
    Декодирование предсказанных offsets в boxes.
    Args:
        pred_offsets: Tensor размера (n_priors, 4) с predicted offsets
        priors: Tensor размера (n_priors, 4) в формате (cx, cy, w, h)
    Returns:
        boxes: Tensor размера (n_priors, 4) в формате (cx, cy, w, h)
    """
    # cx = prior_cx + gcx * prior_w / 10
    # cy = prior_cy + gcy * prior_h / 10
    # w = prior_w * exp(gw / 5)
    # h = prior_h * exp(gh / 5)
    
    cxcy = pred_offsets[:, :2] * priors[:, 2:] / 10 + priors[:, :2]
    wh = torch.exp(pred_offsets[:, 2:] / 5) * priors[:, 2:]
    
    return torch.cat([cxcy, wh], dim=1)


def match_priors_to_gt(gt_boxes, gt_labels, priors, iou_threshold=0.5):
    """
    Сопоставление prior boxes с ground truth для обучения.
    Args:
        gt_boxes: Tensor размера (n_objects, 4) в формате (xmin, ymin, xmax, ymax)
        gt_labels: Tensor размера (n_objects,) с метками классов
        priors: Tensor размера (n_priors, 4) в формате (cx, cy, w, h)
        iou_threshold: порог IoU для positive matches
    Returns:
        matched_boxes: Tensor размера (n_priors, 4) - matched GT boxes в формате (cx, cy, w, h)
        matched_labels: Tensor размера (n_priors,) - matched labels
    """
    n_priors = priors.size(0)
    n_objects = gt_boxes.size(0)
    
    # Конвертируем priors в xy формат
    priors_xy = cxcy_to_xy(priors)
    
    # Расчет IoU между priors и GT boxes
    iou = calculate_iou(priors_xy, gt_boxes)  # (n_priors, n_objects)
    
    # Для каждого prior находим GT с максимальным IoU
    best_iou, best_gt_idx = iou.max(dim=1)  # (n_priors,)
    
    # Также для каждого GT находим prior с максимальным IoU
    _, best_prior_idx = iou.max(dim=0)  # (n_objects,)
    
    # Гарантируем, что каждый GT объект имеет хотя бы один matched prior
    best_iou.index_fill_(0, best_prior_idx, 2.0)  # Значение > 1 для гарантированного match
    
    for j in range(n_objects):
        best_gt_idx[best_prior_idx[j]] = j
    
    # Получаем matched boxes и labels
    matched_boxes = gt_boxes[best_gt_idx]  # (n_priors, 4) в формате xy
    matched_boxes = xy_to_cxcy(matched_boxes)  # Конвертируем в cxcy формат
    
    matched_labels = gt_labels[best_gt_idx]  # (n_priors,)
    matched_labels[best_iou < iou_threshold] = 0  # Background для низкого IoU
    
    return matched_boxes, matched_labels


def non_max_suppression(boxes, scores, iou_threshold=0.45):
    """
    Non-Maximum Suppression для удаления дублирующихся детекций.
    Args:
        boxes: Tensor размера (n_boxes, 4) в формате (xmin, ymin, xmax, ymax)
        scores: Tensor размера (n_boxes,) с confidence scores
        iou_threshold: порог IoU для suppression
    Returns:
        keep_indices: List индексов boxes для сохранения
    """
    if boxes.size(0) == 0:
        return []
    
    # Сортируем по scores
    _, order = scores.sort(descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        
        i = order[0].item()
        keep.append(i)
        
        # Расчет IoU текущего box со всеми остальными
        iou = calculate_iou(boxes[i:i+1], boxes[order[1:]])[0]
        
        # Оставляем только boxes с IoU < threshold
        mask = iou < iou_threshold
        order = order[1:][mask]
    
    return keep


def parse_voc_annotation(xml_path):
    """
    Парсинг Pascal VOC XML аннотации.
    Args:
        xml_path: путь к XML файлу
    Returns:
        boxes: list of [xmin, ymin, xmax, ymax]
        labels: list of class names
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    labels = []
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    for obj in root.iter('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # Нормализация координат
        xmin = xmin / width
        ymin = ymin / height
        xmax = xmax / width
        ymax = ymax / height
        
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
    
    return boxes, labels, (width, height)


class MultiBoxLoss(torch.nn.Module):
    """
    Multibox Loss для SSD.
    Комбинирует classification loss и localization loss.
    """
    def __init__(self, priors, alpha=1.0, neg_pos_ratio=3):
        super(MultiBoxLoss, self).__init__()
        self.priors = priors
        self.priors_xy = cxcy_to_xy(priors)
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
    
    def forward(self, pred_locs, pred_scores, gt_boxes, gt_labels):
        """
        Args:
            pred_locs: Tensor (batch_size, n_priors, 4) - предсказанные offsets
            pred_scores: Tensor (batch_size, n_priors, n_classes) - предсказанные scores
            gt_boxes: List of Tensors размера (n_objects, 4) для каждого изображения
            gt_labels: List of Tensors размера (n_objects,) для каждого изображения
        Returns:
            loss: скалярное значение loss
        """
        batch_size = pred_locs.size(0)
        n_priors = self.priors.size(0)
        n_classes = pred_scores.size(2)
        
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(pred_locs.device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(pred_locs.device)
        
        # Match priors для каждого изображения в batch
        for i in range(batch_size):
            matched_boxes, matched_labels = match_priors_to_gt(
                gt_boxes[i], gt_labels[i], self.priors.to(pred_locs.device)
            )
            
            # Encode matched boxes
            true_locs[i] = encode_boxes(matched_boxes, self.priors.to(pred_locs.device))
            true_classes[i] = matched_labels
        
        # Positive priors (не background)
        positive_priors = true_classes > 0  # (batch_size, n_priors)
        
        # LOCALIZATION LOSS (Smooth L1 Loss)
        # Только для positive priors
        loc_loss = F.smooth_l1_loss(
            pred_locs[positive_priors],
            true_locs[positive_priors],
            reduction='sum'
        )
        
        # CONFIDENCE LOSS (Cross Entropy Loss)
        # Hard Negative Mining
        n_positives = positive_priors.sum(dim=1)  # (batch_size,)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (batch_size,)
        
        # Confidence loss для всех priors
        conf_loss_all = F.cross_entropy(
            pred_scores.view(-1, n_classes),
            true_classes.view(-1),
            reduction='none'
        ).view(batch_size, n_priors)  # (batch_size, n_priors)
        
        # Positive loss
        conf_loss_pos = conf_loss_all[positive_priors].sum()
        
        # Hard negative mining
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0.  # Исключаем positive priors
        
        # Сортируем negative losses и берем top-k
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        
        hardness_ranks = torch.arange(n_priors, dtype=torch.long).unsqueeze(0).expand_as(conf_loss_neg).to(pred_locs.device)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (batch_size, n_priors)
        
        conf_loss_hard_neg = conf_loss_neg[hard_negatives].sum()
        
        conf_loss = conf_loss_pos + conf_loss_hard_neg
        
        # TOTAL LOSS
        n_positives_total = n_positives.sum().float()
        
        if n_positives_total == 0:
            return torch.tensor(0., requires_grad=True).to(pred_locs.device)
        
        loss = (conf_loss + self.alpha * loc_loss) / n_positives_total
        
        return loss
