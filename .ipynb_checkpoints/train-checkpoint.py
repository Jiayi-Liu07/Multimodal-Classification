import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import plot_training_history

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10):
    """
    Train and validate the model, tracking metrics and using learning rate scheduling.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run on (cuda/cpu)
        num_epochs: Number of epochs to train
    
    Returns:
        dict: Dictionary containing training history
    """
    model.to(device)
    
    # Initialize metrics tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': [],
        'epoch_times': []
    }
    
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            categorical = batch['categorical'].to(device)
            continuous = batch['continuous'].to(device)
            
            # Handle image if available
            if 'image' in batch:
                images = batch['image'].to(device)
            else:
                # Create a dummy tensor if no images
                images = torch.zeros((categorical.size(0), 3, 224, 224), device=device)
            
            labels = batch['label'].to(device)
            
            outputs = model(images, categorical, continuous)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # Record metrics
            train_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        # Calculate epoch metrics
        train_loss = np.mean(train_losses)
        train_acc = accuracy_score(train_targets, train_preds)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                categorical = batch['categorical'].to(device)
                continuous = batch['continuous'].to(device)
                
                # Handle image if available
                if 'image' in batch:
                    images = batch['image'].to(device)
                else:
                    # Create a dummy tensor if no images
                    images = torch.zeros((categorical.size(0), 3, 224, 224), device=device)
                
                labels = batch['label'].to(device)
                
                outputs = model(images, categorical, continuous)
                loss = criterion(outputs, labels)
                
                # Record metrics
                val_losses.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_targets, val_preds)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record epoch time
        epoch_time = time.time() - start_time
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        history['epoch_times'].append(epoch_time)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s - lr: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
    
    total_time = sum(history['epoch_times'])
    print(f"Training completed in {total_time:.2f}s")
    
    return history

def train_model(args, model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': [], 'epoch_times': []
    }
    
    print(f"Starting training {args.model_type} model for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            if args.model_type == 'multimodal':
                categorical = batch['categorical'].to(device)
                continuous = batch['continuous'].to(device)
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images, categorical, continuous)
            
            elif args.model_type == 'image_only':
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
            
            elif args.model_type == 'tabular_only':
                categorical = batch['categorical'].to(device)
                continuous = batch['continuous'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(categorical, continuous)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        train_loss = np.mean(train_losses)
        train_acc = accuracy_score(train_targets, train_preds)
        
        # Validation phase
        val_loss, val_acc = evaluate(args, model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record epoch time
        epoch_time = time.time() - start_time
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        history['epoch_times'].append(epoch_time)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{args.epochs} - {epoch_time:.2f}s - lr: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, f"final_{args.model_type}_model.pt")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, final_model_path)
    
    total_time = sum(history['epoch_times'])
    print(f"Training completed in {total_time:.2f}s")
    print(f"Final validation accuracy: {val_acc:.4f}")
    
    # Plot and save training history
    plot_training_history(history, args.output_dir, args.model_type)
    
    return history, final_model_path

def evaluate(args, model, data_loader, criterion, device):
    model.eval()
    losses = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            if args.model_type == 'multimodal':
                categorical = batch['categorical'].to(device)
                continuous = batch['continuous'].to(device)
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images, categorical, continuous)
            
            elif args.model_type == 'image_only':
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
            
            elif args.model_type == 'tabular_only':
                categorical = batch['categorical'].to(device)
                continuous = batch['continuous'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(categorical, continuous)
            
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    avg_loss = np.mean(losses)
    accuracy = accuracy_score(all_targets, all_preds)
    
    return avg_loss, accuracy

def test_model(args, model, test_loader, device, dataset):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    test_loss, test_acc = evaluate(args, model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Detailed evaluation
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if args.model_type == 'multimodal':
                categorical = batch['categorical'].to(device)
                continuous = batch['continuous'].to(device)
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images, categorical, continuous)
            
            elif args.model_type == 'image_only':
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
            
            elif args.model_type == 'tabular_only':
                categorical = batch['categorical'].to(device)
                continuous = batch['continuous'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(categorical, continuous)
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Get class names if available
    class_names = dataset.get_label_map() if hasattr(dataset, 'get_label_map') else None
    
    # Calculate and print classification report
    print("\nClassification Report:")
    if class_names is not None:
        report = classification_report(all_targets, all_preds, target_names=class_names)
    else:
        report = classification_report(all_targets, all_preds)
    print(report)
    
    # Calculate and plot confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    plt.title(f'Confusion Matrix - {args.model_type.replace("_", " ").title()} Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix
    cm_path = os.path.join(args.output_dir, f'{args.model_type}_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    # Save predictions to file
    results_df = pd.DataFrame({
        'True': all_targets,
        'Predicted': all_preds
    })
    if class_names is not None:
        results_df['True_Label'] = [class_names[i] for i in all_targets]
        results_df['Predicted_Label'] = [class_names[i] for i in all_preds]
    
    results_path = os.path.join(args.output_dir, f'{args.model_type}_test_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\nTest results saved to {args.output_dir}")
    return test_loss, test_acc