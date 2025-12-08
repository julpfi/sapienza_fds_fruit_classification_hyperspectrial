import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import wandb
from tqdm import tqdm
from datetime import datetime
from src.config import set_seed

from src.models.models import get_model
from src.data_loader.data_loader import get_data_loader
from src.data_loader.utils.enums import FruitType, CameraType, DatasetSplit

#-------------------- Configuration -------------------------

CONFIG = {
    # Names for wandd 
    "project_name": "sapienza_fds_fruit_ripeness",
    "run_name": "Attention CNN Model - all bands wiht first conv",
    
    # Model and data 
    "model_type": "attention_combined_cnn",
    "fruit": FruitType.KIWI,
    "camera": CameraType.FX10,
    "bands": 224,
    "band_selection": None,
    "band_reduction": all,
    "img_size": (224, 224),
    
    # Hyperparameters
    "batch_size": 16,
    "epochs": 30,
    "lr": 1e-4,
    "num_workers": 2,
    
    # Paths (mounted drive)
    "data_root": "/content/drive/MyDrive/sapienza_fds_fruit_classification/data",
    "json_root": "/content/drive/MyDrive/sapienza_fds_fruit_classification/data/dataset",
    "save_dir": "/content/drive/MyDrive/sapienza_fds_fruit_classification/checkpoints"
}


#-------------------- Helpers -------------------------
def get_unique_filename(config, wandb_run_id=0):
    timestamp = datetime.now().strftime("%m%d")
    fruit = str(config['fruit']).split('.')[-1].lower()
    return f"{config['model_type']}_{fruit}_{timestamp}_{wandb_run_id}.pth"


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(loader), 100 * correct / total


#-------------------- Train Method (Called as Main) -------------------------
def train():
    # Setup of w&b, device 
    wandb.init(project=CONFIG['project_name'], name=CONFIG['run_name'], config=CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    print(f"Configuration:: {CONFIG}")

    # Get data loaders
    train_loader = get_data_loader(CONFIG, DatasetSplit.TRAIN, shuffle=True)
    val_loader   = get_data_loader(CONFIG, DatasetSplit.VAL,   shuffle=False)
    test_loader  = get_data_loader(CONFIG, DatasetSplit.TEST,  shuffle=False)
    
    # Get model 
    # TODO Work on wrapper and passsing of arguemtns (somewhere more advanced logic for handling model requ.?)
    model = get_model(
        CONFIG['model_type'], 
        pretrained=False, 
        num_classes=3, 
        in_channels=CONFIG['bands'])

    model = model.to(device)
    
    # TODO Try out different optimizers and maybe losses? 
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()


    # Training excecution 
    best_val_acc = 0.0
    save_path = os.path.join(CONFIG['save_dir'], get_unique_filename(CONFIG, wandb.run.id))
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        pbar = tqdm(train_loader, desc="Training")
        
        # Training loop
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device) #load to GPU
            
            # Training step
            optimizer.zero_grad()                           #reset gradients
            outputs = model(images)                         #forward pass
            loss = criterion(outputs, labels)               #compute loss
            loss.backward()                                 #backpropagation
            optimizer.step()                                #update weights
            
            train_loss += loss.item()                               #Track total loss
            _, predicted = torch.max(outputs.data, 1)               #Get prediction
            train_total += labels.size(0)                           #Update total samples
            train_correct += (predicted == labels).sum().item()     #Update correct prediction
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})        #Update progress bar

        # Metrics
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        avg_val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Results: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        wandb.log({
            "epoch": epoch + 1, "train_loss": avg_train_loss, "train_acc": train_acc, 
            "val_loss": avg_val_loss, "val_acc": val_acc
        })

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(model.state_dict(), save_path)
            wandb.run.summary["best_val_acc"] = best_val_acc


    # Test evaluation
    print(f"\nEvaluating model via test dataset")
    model.load_state_dict(torch.load(save_path, map_location=device)) # loading best model again 
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"Test Accuracy: {test_acc:.2f}%")
    wandb.log({"test_accuracy": test_acc, "test_loss": test_loss})
    wandb.finish()


if __name__ == "__main__":
    set_seed()
    train()