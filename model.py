# model.py - Архитектура SSD300 для детекции клеток крови

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import config


class VGGBase(nn.Module):
    """
    VGG-16 базовая сеть для извлечения признаков.
    """
    def __init__(self):
        super(VGGBase, self).__init__()
        
        # Загружаем предобученную VGG-16
        vgg16 = models.vgg16(pretrained=True)
        
        # Базовые слои VGG (до Pool5)
        self.conv1_1 = vgg16.features[0]
        self.conv1_2 = vgg16.features[2]
        self.pool1 = vgg16.features[4]
        
        self.conv2_1 = vgg16.features[5]
        self.conv2_2 = vgg16.features[7]
        self.pool2 = vgg16.features[9]
        
        self.conv3_1 = vgg16.features[10]
        self.conv3_2 = vgg16.features[12]
        self.conv3_3 = vgg16.features[14]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # Используем ceil_mode
        
        self.conv4_1 = vgg16.features[17]
        self.conv4_2 = vgg16.features[19]
        self.conv4_3 = vgg16.features[21]
        self.pool4 = vgg16.features[23]
        
        self.conv5_1 = vgg16.features[24]
        self.conv5_2 = vgg16.features[26]
        self.conv5_3 = vgg16.features[28]
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # Модифицированный pool5
        
        # Conv6 и Conv7 (заменяют FC слои)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        
        # L2 нормализация для conv4_3
        self.l2_norm = L2Norm(512, 20)
    
    def forward(self, x):
        """
        Args:
            x: Tensor размера (batch_size, 3, 300, 300)
        Returns:
            conv4_3_feats: Tensor размера (batch_size, 512, 38, 38)
            conv7_feats: Tensor размера (batch_size, 1024, 19, 19)
        """
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        conv4_3_feats = self.l2_norm(x)  # L2 нормализация
        x = self.pool4(conv4_3_feats)
        
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool5(x)
        
        x = F.relu(self.conv6(x))
        conv7_feats = F.relu(self.conv7(x))
        
        return conv4_3_feats, conv7_feats


class AuxiliaryConvolutions(nn.Module):
    """
    Дополнительные сверточные слои для создания feature maps на разных масштабах.
    """
    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()
        
        # Conv8
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Conv9
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Conv10
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)
        
        # Conv11
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)
    
    def forward(self, conv7_feats):
        """
        Args:
            conv7_feats: Tensor размера (batch_size, 1024, 19, 19)
        Returns:
            conv8_2_feats: Tensor размера (batch_size, 512, 10, 10)
            conv9_2_feats: Tensor размера (batch_size, 256, 5, 5)
            conv10_2_feats: Tensor размера (batch_size, 256, 3, 3)
            conv11_2_feats: Tensor размера (batch_size, 256, 1, 1)
        """
        x = F.relu(self.conv8_1(conv7_feats))
        conv8_2_feats = F.relu(self.conv8_2(x))
        
        x = F.relu(self.conv9_1(conv8_2_feats))
        conv9_2_feats = F.relu(self.conv9_2(x))
        
        x = F.relu(self.conv10_1(conv9_2_feats))
        conv10_2_feats = F.relu(self.conv10_2(x))
        
        x = F.relu(self.conv11_1(conv10_2_feats))
        conv11_2_feats = F.relu(self.conv11_2(x))
        
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


class PredictionConvolutions(nn.Module):
    """
    Prediction слои для localization и classification.
    """
    def __init__(self, n_classes):
        super(PredictionConvolutions, self).__init__()
        
        self.n_classes = n_classes
        
        # Количество prior boxes на каждой feature map
        n_boxes = {
            'conv4_3': 4,  # 3 aspect ratios + 1 extra
            'conv7': 6,    # 5 aspect ratios + 1 extra
            'conv8_2': 6,
            'conv9_2': 6,
            'conv10_2': 4,
            'conv11_2': 4
        }
        
        # Localization prediction convolutions
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)
        
        # Class prediction convolutions
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)
    
    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, 
                conv10_2_feats, conv11_2_feats):
        """
        Args:
            conv4_3_feats: (batch_size, 512, 38, 38)
            conv7_feats: (batch_size, 1024, 19, 19)
            conv8_2_feats: (batch_size, 512, 10, 10)
            conv9_2_feats: (batch_size, 256, 5, 5)
            conv10_2_feats: (batch_size, 256, 3, 3)
            conv11_2_feats: (batch_size, 256, 1, 1)
        Returns:
            locs: Tensor (batch_size, total_priors, 4) с localization predictions
            class_scores: Tensor (batch_size, total_priors, n_classes) с class predictions
        """
        batch_size = conv4_3_feats.size(0)
        
        # Localization predictions
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)
        
        l_conv7 = self.loc_conv7(conv7_feats)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4)
        
        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)
        
        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)
        
        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)
        
        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)
        
        # Class predictions
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)
        
        c_conv7 = self.cl_conv7(conv7_feats)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)
        
        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)
        
        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)
        
        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)
        
        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)
        
        # Concatenate predictions
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)
        class_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim=1)
        
        return locs, class_scores


class L2Norm(nn.Module):
    """
    L2 нормализация для conv4_3 feature map.
    """
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)
    
    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class SSD300(nn.Module):
    """
    SSD300 модель для детекции объектов.
    """
    def __init__(self, n_classes):
        super(SSD300, self).__init__()
        
        self.n_classes = n_classes
        
        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)
        
        # Создаем prior boxes
        from utils import create_prior_boxes
        self.priors = create_prior_boxes()
    
    def forward(self, images):
        """
        Args:
            images: Tensor размера (batch_size, 3, 300, 300)
        Returns:
            locs: Tensor (batch_size, n_priors, 4) с предсказанными offsets
            class_scores: Tensor (batch_size, n_priors, n_classes) с предсказанными scores
        """
        # Feature extraction
        conv4_3_feats, conv7_feats = self.base(images)
        
        # Auxiliary convolutions
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)
        
        # Predictions
        locs, class_scores = self.pred_convs(
            conv4_3_feats, conv7_feats, conv8_2_feats, 
            conv9_2_feats, conv10_2_feats, conv11_2_feats
        )
        
        return locs, class_scores
    
    def detect_objects(self, predicted_locs, predicted_scores, min_score=0.01, 
                       max_overlap=0.45, top_k=200):
        """
        Детекция объектов на изображении с применением NMS.
        Args:
            predicted_locs: (batch_size, n_priors, 4) - predicted offsets
            predicted_scores: (batch_size, n_priors, n_classes) - predicted scores
            min_score: минимальный confidence threshold
            max_overlap: максимальный IoU для NMS
            top_k: максимальное количество детекций
        Returns:
            detections: List списков детекций для каждого изображения
        """
        from utils import decode_boxes, cxcy_to_xy, non_max_suppression
        
        batch_size = predicted_locs.size(0)
        n_priors = self.priors.size(0)
        
        predicted_scores = F.softmax(predicted_scores, dim=2)
        
        all_images_boxes = []
        all_images_labels = []
        all_images_scores = []
        
        for i in range(batch_size):
            # Decode predictions
            decoded_locs = decode_boxes(predicted_locs[i], self.priors.to(predicted_locs.device))
            decoded_locs = cxcy_to_xy(decoded_locs)
            decoded_locs.clamp_(0, 1)
            
            image_boxes = []
            image_labels = []
            image_scores = []
            
            # Для каждого класса (кроме background)
            for c in range(1, self.n_classes):
                class_scores = predicted_scores[i][:, c]
                
                # Фильтрация по min_score
                score_above_min = class_scores > min_score
                n_above_min = score_above_min.sum().item()
                
                if n_above_min == 0:
                    continue
                
                class_scores = class_scores[score_above_min]
                class_decoded_locs = decoded_locs[score_above_min]
                
                # Сортировка по score
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
                class_decoded_locs = class_decoded_locs[sort_ind]
                
                # NMS
                keep = non_max_suppression(class_decoded_locs, class_scores, max_overlap)
                
                image_boxes.append(class_decoded_locs[keep])
                image_labels.append(torch.LongTensor([c] * len(keep)).to(predicted_locs.device))
                image_scores.append(class_scores[keep])
            
            # Объединяем детекции всех классов
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(predicted_locs.device))
                image_labels.append(torch.LongTensor([0]).to(predicted_locs.device))
                image_scores.append(torch.FloatTensor([0.]).to(predicted_locs.device))
            
            image_boxes = torch.cat(image_boxes, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            
            # Top-k
            if image_scores.size(0) > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]
                image_boxes = image_boxes[sort_ind][:top_k]
                image_labels = image_labels[sort_ind][:top_k]
            
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)
        
        return all_images_boxes, all_images_labels, all_images_scores
