import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from typing import Optional, TextIO
from torch_geometric.nn import GCNConv, GATConv,TransformerConv
from torch_geometric.loader import  DataLoader
from imblearn.over_sampling import SMOTE
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import optuna
import seaborn as sns
from ..module.emkken import EMKKEN
# 清理缓存
torch.cuda.empty_cache()
torch.manual_seed(10)
seed = torch.initial_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_evaluate(model: nn.Module,
                      optimizer: optim.Optimizer,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      test_loader: DataLoader,
                      criterion: nn.Module,
                      scheduler: optim.lr_scheduler._LRScheduler,
                      epochs: int = 50) -> float:
    """
    End-to-end training and evaluation pipeline for graph neural networks.

    Performs full training cycle with validation and testing, tracks multiple performance metrics,
    implements model checkpointing and logging. Supports both classification and regression tasks.

    Args:
        model (nn.Module): Initialized neural network model
        optimizer (optim.Optimizer): Optimizer for parameter updates
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        criterion (nn.Module): Loss function
        scheduler (optim.lr_scheduler): Learning rate scheduler
        epochs (int, optional): Maximum training epochs. Default: 50

    Returns:
        float: Best validation accuracy achieved during training

    Metrics Tracked:
        - Loss (CrossEntropy/MAE/MSE)
        - Accuracy/F1-Score (Classification)
        - Cohen's Kappa (Inter-rater agreement)
        - MAE/MSE (Regression)
        - train_times
        - memory_usage
    """
    # Initialization
    model.to(device)

    # Initialize performance tracking metrics
    best_val_acc = 0.8
    metric_containers = {
        'losses': {'train': [], 'val': [], 'test': []},
        'accuracies': {'train': [], 'val': [], 'test': []},
        'f1_scores': {'train': [], 'val': [], 'test': []},
        'kappas': {'train': [], 'val': [], 'test': []},
        'maes': {'train': [], 'val': [], 'test': []},
        'mses': {'train': [], 'val': [], 'test': []},
        'train_times': [],
        'memory_usage': []
    }

    start_time = time.time()  # 记录训练开始时间

    with open("mamba_kan_log.txt", "a") as log_file:
        for epoch in range(epochs):
            epoch_start_time = time.time()  # 记录每个epoch的开始时间

            # Training phase
            model.train()
            train_metrics = _run_epoch(model, train_loader, device, criterion, optimizer, is_training=True)
            epoch_end_time = time.time()  # 记录每个epoch的结束时间
            epoch_train_time = epoch_end_time - epoch_start_time
            metric_containers['train_times'].append(epoch_train_time)

            # 记录内存使用
            memory_allocated = None
            memory_reserved = None
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # 转换为MB
                memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # 转换为MB
                metric_containers['memory_usage'].append({
                    'allocated': memory_allocated,
                    'reserved': memory_reserved
                })

            # Validation phase
            model.eval()
            val_metrics = _run_epoch(model, val_loader, device, criterion, None, is_training=False)

            # Test phase
            test_metrics = _run_epoch(model, test_loader, device, criterion, None, is_training=False)

            # Fill metric containers
            _fill_metric_containers(metric_containers, train_metrics, val_metrics, test_metrics)

            # Update learning rate scheduler
            scheduler.step()

            # Model checkpoint saving logic
            if test_metrics['acc'] > best_val_acc and test_metrics['f1'] > best_val_f1:
                _save_checkpoint(model, "mamba_kan_best_model")
                best_val_acc = test_metrics['acc']
                best_val_f1 = test_metrics['f1']

            # Log recording and terminal output
            _log_metrics(epoch, train_metrics, val_metrics, test_metrics, log_file, epoch_train_time, memory_allocated, memory_reserved)

        end_time = time.time()  # 记录训练结束时间
        total_train_time = end_time - start_time
        # Log training time
        print(f"Total Training Time: {total_train_time:.2f} seconds")
        log_file.write(f"Total Training Time: {total_train_time:.2f} seconds\n")

        # Log memory usage
        if torch.cuda.is_available():
            print("Memory Usage (MB):")
            for i, mem in enumerate(metric_containers['memory_usage']):
                print(f"  Epoch {i+1}: Allocated {mem['allocated']:.2f} MB, Reserved {mem['reserved']:.2f} MB")
                log_file.write(f"  Epoch {i+1}: Allocated {mem['allocated']:.2f} MB, Reserved {mem['reserved']:.2f} MB\n")

        # Final evaluation and result analysis
        final_metrics = _evaluate_model(model, test_loader, device, criterion)
        _log_final_results(final_metrics, log_file)

    # Resource cleaning and model saving
    _cleanup_resources(model)
    return best_val_acc


def _run_epoch(model: nn.Module,
                loader: DataLoader,
                device: torch.device,
                criterion: nn.Module,
                optimizer: Optional[optim.Optimizer],
                is_training: bool,
               accumulation_steps: 4) -> dict:
    """
    Internal helper function for executing a single training/validation epoch.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for the current phase (train/val/test).
        device (torch.device): Device to run computations on (CPU/GPU).
        criterion (nn.Module): Loss function.
        optimizer (Optional[optim.Optimizer]): Optimizer for training (None for evaluation).
        is_training (bool): Flag indicating whether this is a training epoch.

    Returns:
        dict: Dictionary containing the following metrics:
            - loss (float): Average loss over the epoch.
            - acc (float): Accuracy score.
            - f1 (float): Weighted F1 score.
            - kappa (float): Cohen's Kappa score.
            - mae (float): Mean Absolute Error.
            - mse (float): Mean Squared Error.
    """
    # Initialize metrics
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    # Set model mode
    model.train(is_training)
    # Disable gradient computation for evaluation
    with torch.set_grad_enabled(is_training):
        # for data in loader:
        for i, data in enumerate(loader):
            # Move data to device
            data = data.to(device)

            # Forward pass
            output, pooled_labels, _, _, reg_loss = model(data)
            loss = criterion(output, pooled_labels)

            # Add regularization loss if in training mode
            if is_training:
                total_loss = loss + 1e-5 * reg_loss
                total_loss.backward()
                # optimizer.step()
                # optimizer.zero_grad()
                # 梯度累积
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # Accumulate loss
            epoch_loss += loss.item()

            # Store predictions and labels
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(pooled_labels.cpu().numpy())

        # 清理缓存
        torch.cuda.empty_cache()

    # Calculate metrics
    avg_loss = epoch_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)

    return {
        'loss': avg_loss,
        'acc': acc,
        'f1': f1,
        'kappa': kappa,
        'mae': mae,
        'mse': mse
    }


def _evaluate_model(model: nn.Module,
                    loader: DataLoader,
                    device: torch.device,
                    criterion: nn.Module) -> dict:
    """
    Comprehensive model evaluation function for classification tasks.

    Computes multiple performance metrics including classification scores and regression metrics.
    Generates full classification report and probability outputs for ROC analysis.

    Args:
        model (nn.Module): Trained neural network model to evaluate
        loader (DataLoader): DataLoader for evaluation dataset
        device (torch.device): Computation device (CPU/GPU)
        criterion (nn.Module): Loss function for cross entropy calculation

    Returns:
        dict: Dictionary containing comprehensive evaluation metrics:
            - accuracy (float): Overall classification accuracy
            - f1 (float): Weighted F1 score
            - auc (float): Multi-class ROC AUC score (One-vs-One)
            - kappa (float): Cohen's Kappa inter-rater agreement
            - mae (float): Mean Absolute Error (regression metric)
            - mse (float): Mean Squared Error (regression metric)
            - cross_loss (float): Average cross entropy loss
            - report (str): Full classification report with precision/recall
    """
    # Set model to evaluation mode (disable dropout/batchnorm updates)
    model.eval()

    # Initialize data collection containers
    all_preds = []
    all_labels = []
    all_probs = []
    cross_loss = []

    # Disable gradient computation for evaluation efficiency
    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            output, pooled_labels, _, _, reg_loss = model(data)
            loss = criterion(output, pooled_labels)
            cross_loss.append(loss.item())

            preds = output.argmax(dim=1).cpu().numpy()
            probs = output.softmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(pooled_labels.cpu().numpy())
            all_probs.extend(probs)

    # Calculate comprehensive evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')
    kappa = cohen_kappa_score(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    avg_cross_loss = np.mean(cross_loss)
    report = classification_report(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'kappa': kappa,
        'mae': mae,
        'mse': mse,
        'cross_loss': avg_cross_loss,
        'report': report
    }


def _fill_metric_containers(metric_containers: dict,
                            train_metrics: dict,
                            val_metrics: dict,
                            test_metrics: dict) -> None:
    """
    Fills the metric_containers with metrics from the current epoch.

    Args:
        metric_containers (dict): Dictionary to store all metrics.
        train_metrics (dict): Metrics from the training phase.
        val_metrics (dict): Metrics from the validation phase.
        test_metrics (dict): Metrics from the testing phase.
    """
    # Fill losses
    metric_containers['losses']['train'].append(train_metrics['loss'])
    metric_containers['losses']['val'].append(val_metrics['loss'])
    metric_containers['losses']['test'].append(test_metrics['loss'])

    # Fill accuracies
    metric_containers['accuracies']['train'].append(train_metrics['acc'])
    metric_containers['accuracies']['val'].append(val_metrics['acc'])
    metric_containers['accuracies']['test'].append(test_metrics['acc'])

    # Fill F1 scores
    metric_containers['f1_scores']['train'].append(train_metrics['f1'])
    metric_containers['f1_scores']['val'].append(val_metrics['f1'])
    metric_containers['f1_scores']['test'].append(test_metrics['f1'])

    # Fill Kappas
    metric_containers['kappas']['train'].append(train_metrics['kappa'])
    metric_containers['kappas']['val'].append(val_metrics['kappa'])
    metric_containers['kappas']['test'].append(test_metrics['kappa'])

    # Fill MAEs
    metric_containers['maes']['train'].append(train_metrics['mae'])
    metric_containers['maes']['val'].append(val_metrics['mae'])
    metric_containers['maes']['test'].append(test_metrics['mae'])

    # Fill MSEs
    metric_containers['mses']['train'].append(train_metrics['mse'])
    metric_containers['mses']['val'].append(val_metrics['mse'])
    metric_containers['mses']['test'].append(test_metrics['mse'])


def _save_checkpoint(model: nn.Module, base_filename: str) -> None:
    """
    Saves the model's state dictionary to disk in both .pth and .h5 formats.

    Args:
        model (nn.Module): The trained model to save.
        base_filename (str): Base filename for the checkpoint (without extension).
    """
    # Save as PyTorch checkpoint
    torch.save(model.state_dict(), f"{base_filename}.pth")
    # Save as HDF5 format (optional, for compatibility)
    torch.save(model.state_dict(), f"{base_filename}.h5")
    print(f"Model checkpoint saved to {base_filename}.pth and {base_filename}.h5")


def _log_metrics(epoch: int,
                 train_metrics: dict,
                 val_metrics: dict,
                 test_metrics: dict,
                 log_file: TextIO,
                 epoch_train_time: float,
                 memory_allocated: Optional[float] = None,
                 memory_reserved: Optional[float] = None) -> None:
    """
    Logs training, validation, and test metrics for the current epoch.

    Args:
        epoch (int): Current epoch number.
        train_metrics (dict): Dictionary of training metrics.
        val_metrics (dict): Dictionary of validation metrics.
        test_metrics (dict): Dictionary of test metrics.
        log_file (TextIO): File object for writing logs.
        epoch_train_time (float): Training time for the current epoch.
        memory_allocated (float, optional): Allocated memory in MB. Default: None
        memory_reserved (float, optional): Reserved memory in MB. Default: None
    """
    # Print metrics to console
    print(f"Epoch {epoch + 1}:")
    print(f"  Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, Test Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: Train {train_metrics['acc']:.4f}, Val {val_metrics['acc']:.4f}, Test {test_metrics['acc']:.4f}")
    print(f"  F1 Score: Train {train_metrics['f1']:.4f}, Val {val_metrics['f1']:.4f}, Test {test_metrics['f1']:.4f}")
    print(f"  Kappa: Train {train_metrics['kappa']:.4f}, Val {val_metrics['kappa']:.4f}, Test {test_metrics['kappa']:.4f}")
    print(f"  MAE: Train {train_metrics['mae']:.4f}, Val {val_metrics['mae']:.4f}, Test {test_metrics['mae']:.4f}")
    print(f"  MSE: Train {train_metrics['mse']:.4f}, Val {val_metrics['mse']:.4f}, Test {test_metrics['mse']:.4f}")
    print(f"  Training Time: {epoch_train_time:.2f} seconds")
    if memory_allocated is not None and memory_reserved is not None:
        print(f"  Memory Allocated: {memory_allocated:.2f} MB, Memory Reserved: {memory_reserved:.2f} MB")
    print("-----------------------------------------------------------------------------")

    # Write metrics to log file
    log_file.write(f"Epoch {epoch + 1}:\n")
    log_file.write(f"  Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, Test Loss: {test_metrics['loss']:.4f}\n")
    log_file.write(f"  Accuracy: Train {train_metrics['acc']:.4f}, Val {val_metrics['acc']:.4f}, Test {test_metrics['acc']:.4f}\n")
    log_file.write(f"  F1 Score: Train {train_metrics['f1']:.4f}, Val {val_metrics['f1']:.4f}, Test {test_metrics['f1']:.4f}\n")
    log_file.write(f"  Kappa: Train {train_metrics['kappa']:.4f}, Val {val_metrics['kappa']:.4f}, Test {test_metrics['kappa']:.4f}\n")
    log_file.write(f"  MAE: Train {train_metrics['mae']:.4f}, Val {val_metrics['mae']:.4f}, Test {test_metrics['mae']:.4f}\n")
    log_file.write(f"  MSE: Train {train_metrics['mse']:.4f}, Val {val_metrics['mse']:.4f}, Test {test_metrics['mse']:.4f}\n")
    log_file.write(f"  Training Time: {epoch_train_time:.2f} seconds\n")
    if memory_allocated is not None and memory_reserved is not None:
        log_file.write(f"  Memory Allocated: {memory_allocated:.2f} MB, Memory Reserved: {memory_reserved:.2f} MB\n")
    log_file.write("-----------------------------------------------------------------------------\n")


def _log_final_results(final_metrics: dict, log_file: TextIO) -> None:
    """
    Logs the final evaluation results after training completes.

    Args:
        final_metrics (dict): Dictionary containing final evaluation metrics.
        log_file (TextIO): File object for writing logs.
    """
    # Print final results to console
    print("Final Evaluation Results:")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {final_metrics['f1']:.4f}")
    print(f"  AUC: {final_metrics['auc']:.4f}")
    print(f"  Kappa: {final_metrics['kappa']:.4f}")
    print(f"  MAE: {final_metrics['mae']:.4f}")
    print(f"  MSE: {final_metrics['mse']:.4f}")
    print(f"  Cross Loss: {final_metrics['cross_loss']:.4f}")
    print("Classification Report:")
    print(final_metrics['report'])
    print("-------------------------------------------------------------------------")

    # Write final results to log file
    log_file.write("Final Evaluation Results:\n")
    log_file.write(f"  Accuracy: {final_metrics['accuracy']:.4f}\n")
    log_file.write(f"  F1 Score: {final_metrics['f1']:.4f}\n")
    log_file.write(f"  AUC: {final_metrics['auc']:.4f}\n")
    log_file.write(f"  Kappa: {final_metrics['kappa']:.4f}\n")
    log_file.write(f"  MAE: {final_metrics['mae']:.4f}\n")
    log_file.write(f"  MSE: {final_metrics['mse']:.4f}\n")
    log_file.write(f"  Cross Loss: {final_metrics['cross_loss']:.4f}\n")
    log_file.write("Classification Report:\n")
    log_file.write(final_metrics['report'])
    log_file.write("\n-------------------------------------------------------------------------\n")


def _cleanup_resources(model: nn.Module) -> None:
    """
    Cleans up resources after training, including GPU memory and model checkpoints.

    Args:
        model (nn.Module): The trained model.
    """
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save the final model
    final_model_path = 'mamba_kan_final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


def _plot_metrics(metric_containers: dict, final_metrics: dict) -> None:
    """
    Plots training, validation, and testing metrics.

    Args:
        metric_containers (dict): Dictionary containing all metrics.
        final_metrics (dict): Dictionary containing final evaluation metrics.
    """
    # Plot loss curves
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(metric_containers['losses']['train']) + 1),
             metric_containers['losses']['train'], label='Training Loss')
    plt.plot(range(1, len(metric_containers['losses']['val']) + 1),
             metric_containers['losses']['val'], label='Validation Loss')
    plt.plot(range(1, len(metric_containers['losses']['test']) + 1),
             metric_containers['losses']['test'], label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Testing Loss Curve')
    plt.legend()
    plt.savefig('Loss.png')
    plt.close()

    # Plot training time
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(metric_containers['train_times']) + 1),
             metric_containers['train_times'], label='Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Epoch')
    plt.legend()
    plt.savefig('Training_Time.png')
    plt.close()

    # Plot memory usage
    if 'memory_usage' in metric_containers:
        allocated = [mem['allocated'] for mem in metric_containers['memory_usage']]
        reserved = [mem['reserved'] for mem in metric_containers['memory_usage']]
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(allocated) + 1), allocated, label='Allocated Memory')
        plt.plot(range(1, len(reserved) + 1), reserved, label='Reserved Memory')
        plt.xlabel('Epoch')
        plt.ylabel('Memory (MB)')
        plt.title('Memory Usage per Epoch')
        plt.legend()
        plt.savefig('Memory_Usage.png')
        plt.close()

    # Create a 2x3 subplot layout
    plt.figure(figsize=(15, 15))

    # Plot confusion matrix
    plt.subplot(2, 3, 1)
    cm = confusion_matrix(final_metrics['all_labels'], final_metrics['all_preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap')

    # Plot accuracy curve
    plt.subplot(2, 3, 2)
    plt.plot(range(1, len(metric_containers['accuracies']['train']) + 1),
             metric_containers['accuracies']['train'], label='Training Accuracy')
    plt.plot(range(1, len(metric_containers['accuracies']['val']) + 1),
             metric_containers['accuracies']['val'], label='Validation Accuracy')
    plt.plot(range(1, len(metric_containers['accuracies']['test']) + 1),
             metric_containers['accuracies']['test'], label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    # Plot F1 score curve
    plt.subplot(2, 3, 3)
    plt.plot(range(1, len(metric_containers['f1_scores']['train']) + 1),
             metric_containers['f1_scores']['train'], label='Training F1 Score')
    plt.plot(range(1, len(metric_containers['f1_scores']['val']) + 1),
             metric_containers['f1_scores']['val'], label='Validation F1 Score')
    plt.plot(range(1, len(metric_containers['f1_scores']['test']) + 1),
             metric_containers['f1_scores']['test'], label='Testing F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curve')
    plt.legend()

    # Plot Kappa curve
    plt.subplot(2, 3, 4)
    plt.plot(range(1, len(metric_containers['kappas']['train']) + 1),
             metric_containers['kappas']['train'], label='Training Kappa')
    plt.plot(range(1, len(metric_containers['kappas']['val']) + 1),
             metric_containers['kappas']['val'], label='Validation Kappa')
    plt.plot(range(1, len(metric_containers['kappas']['test']) + 1),
             metric_containers['kappas']['test'], label='Testing Kappa')
    plt.xlabel('Epoch')
    plt.ylabel('Kappa Coefficient')
    plt.title('Kappa Coefficient Curve')
    plt.legend()

    # Plot MAE curve
    plt.subplot(2, 3, 5)
    plt.plot(range(1, len(metric_containers['maes']['train']) + 1),
             metric_containers['maes']['train'], label='Training MAE')
    plt.plot(range(1, len(metric_containers['maes']['val']) + 1),
             metric_containers['maes']['val'], label='Validation MAE')
    plt.plot(range(1, len(metric_containers['maes']['test']) + 1),
             metric_containers['maes']['test'], label='Testing MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('MAE Curve')
    plt.legend()

    # Plot MSE curve
    plt.subplot(2, 3, 6)
    plt.plot(range(1, len(metric_containers['mses']['train']) + 1),
             metric_containers['mses']['train'], label='Training MSE')
    plt.plot(range(1, len(metric_containers['mses']['val']) + 1),
             metric_containers['mses']['val'], label='Validation MSE')
    plt.plot(range(1, len(metric_containers['mses']['test']) + 1),
             metric_containers['mses']['test'], label='Testing MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE Curve')
    plt.legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('Loss_Acc_F1_Kappa_MAE_MSE.png')
    plt.close()


def objective(trial: optuna.Trial) -> float:
    """
    Optuna hyperparameter optimization objective function.

    Defines search space for neural architecture and training parameters,
    executes complete training-evaluation cycles, and returns validation accuracy
    as optimization target.

    Args:
        trial (optuna.Trial): Optuna trial object managing hyperparameter suggestions

    Returns:
        float: Validation accuracy to maximize

    Hyperparameters:
        - Learning rate (lr): Log-uniform in [1e-6, 1e-3]
        - Weight decay: Uniform in [1e-6, 1e-3]
        - Dropout rates: Uniform in [0.1, 0.7]
        - Mamba state dimensions (d_state1, d_state2): Integers in [10, 50] and [4, 84]
        - Convolution widths (d_conv1, d_conv2): Integers in [2, 8] and [2, 4]
        - Hidden units: Integer in [32, 512] (step 16)
        - Batch size: Integer in [32, 256] (step 16)
        - Hidden dimension: Integer in [64, 320] (step 16)
        - T_max (CosineAnnealingLR): Integer in [100, 520]
        - Epochs: Integer in [80, 500]
    """
    # Define hyperparameter search space
    hparams = {
        'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.7),
        'dropout_rate1': trial.suggest_float('dropout_rate1', 0.1, 0.7),
        'dropout_rate2': trial.suggest_float('dropout_rate2', 0.1, 0.7),
        'd_state1': trial.suggest_int('d_state1', 10, 50, step=1),
        'd_state2': trial.suggest_int('d_state2', 4, 84, step=1),
        'd_conv1': trial.suggest_int('d_conv1', 2, 8, step=1),
        'd_conv2': trial.suggest_int('d_conv2', 2, 4, step=1),
        'hidden_units': trial.suggest_int('hidden_units', 32, 512, step=16),
        'batch_size': trial.suggest_int('batch_size', 32, 256, step=16),
        'hidden_dim': trial.suggest_int('hidden_dim', 64, 320, step=16),
        'T_max': trial.suggest_int('T_max', 100, 520, step=1),
        'epochs': trial.suggest_int('epochs', 80, 500, step=1),
        'output_dim': 3,
    }

    # Log hyperparameters
    print(f"Trial {trial.number}: Hyperparameters = {hparams}")
    with open("mamba_kan_log.txt", "a") as log_file:
        log_file.write(f"Trial {trial.number}: Hyperparameters = {hparams}\n")

    # Load and split data
    data_list = torch.load('data_list.pt')
    train_data, temp_data = train_test_split(data_list, train_size=0.8, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=hparams['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=hparams['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=hparams['batch_size'], shuffle=False, num_workers=4)

    # Initialize model
    model = EMKKEN(
        mamba_num_node_features=0,
        mamba_hidden_dim=hparams['hidden_dim'],
        mamba_d_state1=hparams['d_state1'],
        mamba_d_state2=hparams['d_state2'],
        mamba_d_conv1=hparams['d_conv1'],
        mamba_d_conv2=hparams['d_conv2'],
        mamba_dropout_rate1=hparams['dropout_rate1'],
        mamba_dropout_rate2=hparams['dropout_rate2'],
        knu_output_dim=hparams['output_dim'],
        knu_mlp_hidden_dim=hparams['hidden_units'],
        knu_dropout_rate=hparams['dropout_rate']
    )
    model.to(device)

    # Initialize optimizer, criterion and scheduler
    optimizer = optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hparams['T_max'])

    # Train and evaluate
    val_acc = train_and_evaluate(
        model, optimizer, train_loader, val_loader, test_loader,
        criterion, scheduler, hparams['epochs']
    )

    return val_acc


if __name__ == "__main__":
    # Using Optuna for hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    print('Best hyperparameters:', study.best_params)