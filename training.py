import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import wandb
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import json
from unet import UNet

# Config
IMG_SIZE = 128

# Training data paths
TRAIN_DATA_DIR = "/content/dataset/training"
TRAIN_MANIFEST_FILE = "/content/dataset/training/data.json"

# Validation data paths
VAL_DATA_DIR = "/content/dataset/validation"
VAL_MANIFEST_FILE = "/content/dataset/validation/data.json"

COLOR_MAP = {
    "red": 0, "green": 1, "blue": 2, "yellow": 3, "purple": 4,
    "orange": 5, "cyan": 6, "magenta": 7
}
NUM_COLORS = len(COLOR_MAP)

# Grid of hyperparameters to try 
HYPERPARAMETER_GRID = {
    'learning_rate': [1e-3, 1e-4],
    'batch_size': [64],
    'loss_function': ['MSE', 'L1'],
    'optimizer': ['Adam'],
    'num_epochs': [50,100]  
}

# Dataset Class 
class PolygonDatasetGrid(Dataset):
    def __init__(self, manifest_path, input_dir, target_dir, color_map):
        print(f"Initializing PolygonDatasetGrid...")
        print(f"  manifest_path: {manifest_path}")
        print(f"  input_dir: {input_dir}")
        print(f"  target_dir: {target_dir}")
        print(f"  color_map: {color_map}")
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        print(f"  Loaded {len(self.manifest)} records from manifest")

        self.input_dir = input_dir
        self.target_dir = target_dir
        self.color_map = color_map

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])
        
        print(f"PolygonDatasetGrid initialization complete")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        try:
            record = self.manifest[idx]

            color_name = record['colour']
            input_filename = record['input_polygon']
            output_filename = record['output_image']

            color_idx = self.color_map[color_name]

            input_path = os.path.join(self.input_dir, input_filename)
            output_path = os.path.join(self.target_dir, output_filename)

            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input image not found: {input_path}")
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Output image not found: {output_path}")

            input_img = Image.open(input_path).convert("L")
            target_img = Image.open(output_path).convert("RGB")

            # Apply the transform to both images
            input_tensor = self.transform(input_img)
            target_tensor = self.transform(target_img)

            color_tensor = torch.tensor([color_idx], dtype=torch.long)

            return input_tensor, color_tensor, target_tensor
        
        except Exception as e:
            print(f"Error in _getitem_ for index {idx}: {e}")
            raise e

# Helper Functions
def get_loss_function(loss_name):
    """Get loss function by name"""
    if loss_name == 'MSE':
        return nn.MSELoss()
    elif loss_name == 'L1':
        return nn.L1Loss()
    elif loss_name == 'SmoothL1':
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

def get_optimizer(optimizer_name, model_params, lr):
    """Get optimizer by name"""
    if optimizer_name == 'Adam':
        return optim.Adam(model_params, lr=lr)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model_params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def create_data_loaders(batch_size):
    """Create training and validation data loaders"""
    print(f"Creating data loaders with batch_size={batch_size}")
    
    # Check paths exist
    train_input_dir = os.path.join(TRAIN_DATA_DIR, "inputs")
    train_output_dir = os.path.join(TRAIN_DATA_DIR, "outputs")
    val_input_dir = os.path.join(VAL_DATA_DIR, "inputs")
    val_output_dir = os.path.join(VAL_DATA_DIR, "outputs")
    
    print(f"Train input dir: {train_input_dir} (exists: {os.path.exists(train_input_dir)})")
    print(f"Train output dir: {train_output_dir} (exists: {os.path.exists(train_output_dir)})")
    print(f"Val input dir: {val_input_dir} (exists: {os.path.exists(val_input_dir)})")
    print(f"Val output dir: {val_output_dir} (exists: {os.path.exists(val_output_dir)})")
    
    # Training DataLoader
    print("Creating training dataset...")
    train_dataset = PolygonDatasetGrid(
        manifest_path=TRAIN_MANIFEST_FILE,
        input_dir=train_input_dir,
        target_dir=train_output_dir,
        color_map=COLOR_MAP
    )
    print(f"Training dataset created with {len(train_dataset)} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Training dataloader created with {len(train_loader)} batches")

    # Validation DataLoader
    print("Creating validation dataset...")
    val_dataset = PolygonDatasetGrid(
        manifest_path=VAL_MANIFEST_FILE,
        input_dir=val_input_dir,
        target_dir=val_output_dir,
        color_map=COLOR_MAP
    )
    print(f"Validation dataset created with {len(val_dataset)} samples")
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Validation dataloader created with {len(val_loader)} batches")
    
    return train_loader, val_loader

# Single Training Function
def train_single_config(config, device, model_class):
    """Train model with a single hyperparameter configuration"""
    
    # Extract hyperparameters
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    loss_function = config['loss_function']
    optimizer_name = config['optimizer']
    num_epochs = config['num_epochs']
    
    print(f"\nTraining with config: {config}")
    
    # Create unique run name
    run_name = f"lr_{learning_rate}bs{batch_size}{loss_function}{optimizer_name}ep{num_epochs}"
    
    # Initialize wandb
    wandb.init(
        project="unet-polygon-colorizer-grid-search",
        name=run_name,
        config=config,
        reinit=True
    )
    
    try:
        # Create data loaders
        train_loader, val_loader = create_data_loaders(batch_size)
        
        # Check if data loaders are working
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Get a fixed batch from validation set for consistent visualization
        try:
            val_inputs, val_colors, val_targets = next(iter(val_loader))
            val_inputs = val_inputs.to(device)
            val_colors = val_colors.to(device)
            val_targets = val_targets.to(device)
            print(f"Validation batch shapes: {val_inputs.shape}, {val_colors.shape}, {val_targets.shape}")
        except Exception as e:
            print(f"Error loading validation batch: {e}")
            raise e
        
        # Initialize model
        print("Initializing model...")
        model = model_class(n_channels=1, n_classes=3, num_colors=NUM_COLORS).to(device)
        print(f"Model created and moved to {device}")
        
        # Initialize optimizer and loss function
        optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
        criterion = get_loss_function(loss_function)
        print(f"Optimizer: {optimizer_name}, Loss: {loss_function}")
        
        # Watch model
        wandb.watch(model, criterion, log="all", log_freq=10)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        print(f"Starting training for {num_epochs} epochs...")
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            model.train()
            train_loss = 0
            batch_count = 0
            
            for batch_idx, (inputs, colors, targets) in enumerate(train_loader):
                inputs, colors, targets = inputs.to(device), colors.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs, colors)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                batch_count += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, colors, targets in val_loader:
                    inputs, colors, targets = inputs.to(device), colors.to(device), targets.to(device)
                    outputs = model(inputs, colors)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Track best validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            
            print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Best Val: {best_val_loss:.4f}")
            
            # Log to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss,
            })
            
            # Log images every 5 epochs or last epoch
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(val_inputs, val_colors)
                    
                    input_grid = val_inputs[:4].repeat(1, 3, 1, 1)
                    output_grid = val_outputs[:4]
                    target_grid = val_targets[:4]
                    
                    comparison_stack = torch.cat([input_grid, output_grid, target_grid], dim=0)
                    image_grid = make_grid(comparison_stack, nrow=4)
                    
                    wandb.log({
                        "predictions": wandb.Image(image_grid, caption="Top: Input, Middle: Prediction, Bottom: Ground Truth")
                    })
        
        # Save model with unique name
        model_save_path = f"polygon_unet_{run_name}.pth"
        print(f"Saving model to {model_save_path}")
        torch.save(model.state_dict(), model_save_path)
        
        # Verify model was saved
        if os.path.exists(model_save_path):
            print(f"Model successfully saved to {model_save_path}")
            file_size = os.path.getsize(model_save_path)
            print(f"Model file size: {file_size / 1024 / 1024:.2f} MB")
        else:
            print(f"WARNING: Model file {model_save_path} was not created!")
        
        # Log model artifact
        try:
            artifact = wandb.Artifact(f'polygon-model-{run_name}', type='model')
            artifact.add_file(model_save_path)
            wandb.log_artifact(artifact)
            print("Model artifact logged to wandb")
        except Exception as e:
            print(f"Warning: Could not log model artifact: {e}")
        
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        
        return {
            'config': config,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_losses[-1],
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model_path': model_save_path
        }
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'config': config,
            'best_val_loss': float('inf'),
            'error': str(e)
        }
    
    finally:
        wandb.finish()

# Grid Search Function
def grid_search(model_class):
    """Perform grid search over hyperparameters"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test data loading first
    print("Testing data loading...")
    try:
        # Check if data files exist
        if not os.path.exists(TRAIN_MANIFEST_FILE):
            print(f"Error: Training manifest file not found: {TRAIN_MANIFEST_FILE}")
            return None
        if not os.path.exists(VAL_MANIFEST_FILE):
            print(f"Error: Validation manifest file not found: {VAL_MANIFEST_FILE}")
            return None
        
        test_loader, _ = create_data_loaders(32)
        test_batch = next(iter(test_loader))
        print(f"Data loading successful. Batch shapes: {[x.shape for x in test_batch]}")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Generate all combinations of hyperparameters
    param_names = list(HYPERPARAMETER_GRID.keys())
    param_values = list(HYPERPARAMETER_GRID.values())
    
    all_combinations = list(product(*param_values))
    total_combinations = len(all_combinations)
    
    print(f"Total combinations to try: {total_combinations}")
    print("Hyperparameter grid:")
    for key, values in HYPERPARAMETER_GRID.items():
        print(f"  {key}: {values}")
    
    results = []
    
    # Train model for each combination
    for i, combination in enumerate(all_combinations):
        config = dict(zip(param_names, combination))
        print(f"\n{'='*50}")
        print(f"Configuration {i+1}/{total_combinations}")
        print(f"{'='*50}")
        
        result = train_single_config(config, device, model_class)
        results.append(result)
        
        # Print intermediate results
        if 'error' not in result:
            print(f"✓ Completed config {i+1}: Val Loss = {result['best_val_loss']:.4f}")
        else:
            print(f"✗ Failed config {i+1}: {result['error']}")
    
    # Find best configuration
    valid_results = [r for r in results if 'error' not in r]
    
    if valid_results:
        best_result = min(valid_results, key=lambda x: x['best_val_loss'])
        
        print(f"\n{'='*60}")
        print("GRID SEARCH RESULTS")
        print(f"{'='*60}")
        print(f"Best configuration:")
        for key, value in best_result['config'].items():
            print(f"  {key}: {value}")
        print(f"Best validation loss: {best_result['best_val_loss']:.4f}")
        print(f"Model saved at: {best_result['model_path']}")
        
        # Sort results by validation loss
        valid_results.sort(key=lambda x: x['best_val_loss'])
        
        print(f"\nTop {min(5, len(valid_results))} configurations:")
        for i, result in enumerate(valid_results[:5]):
            print(f"{i+1}. Val Loss: {result['best_val_loss']:.4f} - {result['config']}")
        
        # Save detailed results
        results_file = "grid_search_results.json"
        try:
            with open(results_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                results_for_json = []
                for r in results:
                    r_copy = r.copy()
                    if 'train_losses' in r_copy:
                        r_copy['train_losses'] = [float(x) for x in r_copy['train_losses']]
                    if 'val_losses' in r_copy:
                        r_copy['val_losses'] = [float(x) for x in r_copy['val_losses']]
                    results_for_json.append(r_copy)
                
                json.dump(results_for_json, f, indent=2)
            print(f"\nDetailed results saved to: {results_file}")
        except Exception as e:
            print(f"Warning: Could not save results file: {e}")
        
        return best_result
    else:
        print("No valid configurations found!")
        return None





   

# Main Execution 
def main_train():
    print("Starting Grid Search for Hyperparameter Tuning...")
    print(f"Total combinations to try: {len(list(product(*HYPERPARAMETER_GRID.values())))}")
    
    # Check file paths first
    print("\nChecking data paths...")
    print(f"TRAIN_DATA_DIR: {TRAIN_DATA_DIR}")
    print(f"TRAIN_MANIFEST_FILE: {TRAIN_MANIFEST_FILE}")
    print(f"VAL_DATA_DIR: {VAL_DATA_DIR}")
    print(f"VAL_MANIFEST_FILE: {VAL_MANIFEST_FILE}")
    
    print(f"Train data dir exists: {os.path.exists(TRAIN_DATA_DIR)}")
    print(f"Train manifest exists: {os.path.exists(TRAIN_MANIFEST_FILE)}")
    print(f"Val data dir exists: {os.path.exists(VAL_DATA_DIR)}")
    print(f"Val manifest exists: {os.path.exists(VAL_MANIFEST_FILE)}")
    
    if os.path.exists(TRAIN_DATA_DIR):
        print(f"Train data dir contents: {os.listdir(TRAIN_DATA_DIR)}")
    if os.path.exists(VAL_DATA_DIR):
        print(f"Val data dir contents: {os.listdir(VAL_DATA_DIR)}")
    
    # Run grid search
    best_config = grid_search(model_class=UNet)
    
    if best_config:
        print("\n" + "="*60)
        print("GRID SEARCH COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Best configuration found:")
        for key, value in best_config['config'].items():
            print(f"  {key}: {value}")
        print(f"Best validation loss: {best_config['best_val_loss']:.6f}")
        print(f"Model saved at: {best_config['model_path']}")
    else:
        print("Grid search failed - no valid configurations found")

if __name__ == "__main__":
    main_train()