# train.py - Скрипт обучения SSD модели

import torch
import torch.optim as optim
from tqdm import tqdm
import os
import config
from model import SSD300
from dataset import create_dataloaders
from utils import MultiBoxLoss, create_prior_boxes


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Обучение модели на одной эпохе.
    """
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
    
    for batch_idx, (images, boxes, labels) in enumerate(pbar):
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        
        # Forward pass
        predicted_locs, predicted_scores = model(images)
        
        # Compute loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'avg_loss': running_loss / (batch_idx + 1)})
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Валидация модели.
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, boxes, labels in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            
            # Forward pass
            predicted_locs, predicted_scores = model(images)
            
            # Compute loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            
            running_loss += loss.item()
    
    val_loss = running_loss / len(dataloader)
    return val_loss


def train():
    """
    Основная функция обучения.
    """
    # Создаем директорию для checkpoints
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Device
    device = config.DEVICE
    print(f'Используется устройство: {device}')
    
    # Создаем DataLoaders
    print('Загрузка данных...')
    train_loader, val_loader = create_dataloaders(train_split=0.8)
    
    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
    
    # Создаем модель
    print('Создание модели...')
    model = SSD300(n_classes=config.NUM_CLASSES)
    model = model.to(device)
    
    # Prior boxes
    priors = create_prior_boxes().to(device)
    
    # Loss function
    criterion = MultiBoxLoss(priors, alpha=config.ALPHA)
    
    # Optimizer
    biases = []
    not_biases = []
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)
    
    optimizer = optim.SGD(
        [
            {'params': biases, 'lr': 2 * config.LEARNING_RATE},
            {'params': not_biases}
        ],
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.LR_DECAY_EPOCHS,
        gamma=config.LR_DECAY_FACTOR
    )
    
    # Training loop
    print('Начало обучения...')
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Learning rate scheduler step
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'ssd300_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f'Checkpoint сохранен: {checkpoint_path}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_PATH)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)
            print(f'Лучшая модель сохранена: {best_model_path}')
    
    print('Обучение завершено!')
    
    # Save training history
    import json
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    history_path = os.path.join(config.CHECKPOINT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f'История обучения сохранена: {history_path}')


if __name__ == '__main__':
    train()
