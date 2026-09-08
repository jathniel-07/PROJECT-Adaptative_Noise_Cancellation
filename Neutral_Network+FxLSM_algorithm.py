import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import time

# ==============================================================================
# 1. CONFIGURATION (Easy to modify for future use)
# ==============================================================================

class Config:
    # Path to your noise dataset file(s). 
    # You can change this to load multiple files by modifying the load_data function later.
    DATA_PATH = "noise_samples.npy" 
    
    # Hyperparameters
    INPUT_DIM = 8           # Number of consecutive parameters (time steps) looking back
    HIDDEN_SIZE_1 = 64      # Size of first hidden layer
    HIDDEN_SIZE_2 = 32      # Size of second hidden layer
    OUTPUT_DIM = 1          # Scalar output (Cancellation signal)
    
    BATCH_SIZE = 256        # Larger batches are better for GPU efficiency
    EPOCHS = 50             # Number of training passes
    LEARNING_RATE = 0.001   # Adam LR
    
    # Device settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. CUSTOM DATASET CLASS (Handles loading and sliding windows)
# ==============================================================================

class NoiseDataset(Dataset):
    def __init__(self, file_path, input_window=8):
        self.data = None
        self.input_window = input_window
        
        # Load Data Logic
        if os.path.exists(file_path):
            try:
                raw_data = np.load(file_path)
                print(f"[INFO] Successfully loaded data from {file_path}")
                
                # Reshape handling: Ensure it's a 1D time series for windowing
                if raw_data.ndim ==1:
                    self.raw_data = raw_data.astype(np.float32)
                elif raw_data.ndim > 1:
                # Flatten multi-dimensional data or take first channel if needed for 1D signal
                    self.raw_data = raw_data.flatten().astype(np.float32)
                    print("[WARNING] Data was reshaped to 1D time-series.")
                else:
                    raise ValueError("Unexpected data dimensionality")

            # Create Sliding Window (Input Parameters)
            # X shape: (N - window, window), y shape: (N - window, 1)
            total_samples = len(self.raw_data)
            num_windows = total_samples - self.input_window + 1
            
            if num_windows <= 0:
                raise ValueError("Dataset too small for specified input window size.")

            # Efficient vectorized window creation
            X = np.lib.stride_tricks.as_strided(
                self.raw_data[:num_windows+self.input_window-1], 
                shape=(num_windows, self.input_window), 
                strides=(self.raw_data.strides[0], self.raw_data.strides[0])
            ).copy() # Copy to ensure contiguous memory for PyTorch

            # Target: Ideally for ANC we want to generate anti-noise. 
            # Here we assume target is the negative of the next sample or current sample depending on implementation.
            # Standard regression for cancellation: Predict signal that sums to zero.
            # y = -raw_data[window_start + input_window:] (Anti-noise signal)
            y = -self.raw_data[self.input_window:num_windows + self.input_window] 
            
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y).unsqueeze(1) # Add feature dimension
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==============================================================================
# 3. MODEL ARCHITECTURE (The Adaptive Filter Approximation)
# ==============================================================================

class AdaptiveFilterNet(nn.Module):
    def __init__(self, config: Config):
        super(AdaptiveFilterNet, self).__init__()
        
        # Input Layer: Matches the sliding window size (e.g., 8 consecutive params)
        input_dim = config.INPUT_DIM 
        
        # Define Sequential Architecture with ReLU (Non-linear activation as requested)
        self.model = nn.Sequential(
            nn.Linear(input_dim, config.HIDDEN_SIZE_1),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_SIZE_1, config.HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_SIZE_2, config.OUTPUT_DIM)
        )

    def forward(self, x):
        return self.model(x)


# ==============================================================================
# 4. TRAINING LOOP & LOSS CRITERION
# ==============================================================================

def train_model(config: Config, dataset: NoiseDataset):
    # Initialize Model and move to GPU immediately
    model = AdaptiveFilterNet(config).to(config.DEVICE)
    
    # Loss Function: Mean Squared Error (Standard for Signal Estimation/ANC)
    criterion = nn.MSELoss()
    
    # Optimizer: ADAM as requested
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Data Loading: Use DataLoader for batching and shuffling to reduce memory/time complexity
    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    
    print(f"\n[INFO] Starting Training on {config.DEVICE}")
    print(f"[INFO] Model Parameters: {sum(p.numel() for p in model.parameters())}")
    print("-" * 60)

    for epoch in range(config.EPOCHS):
        model.train() # Set model to training mode
        total_loss = 0.0
        num_batches = len(train_loader)
        
        start_time = time.time()
        
        # Progress tracking per batch is optional, here we print per epoch
        for inputs, targets in train_loader:
            # Move tensors to GPU
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            
            # Forward Pass
            outputs = model(inputs)
            
            # Calculate Loss
            loss = criterion(outputs, targets)
            
            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        
        # Check progress output
        print(f"[Epoch {epoch+1}/{config.EPOCHS}] Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s")

    print("-" * 60)
    print("[INFO] Training Complete!")
    
    return model


# ==============================================================================
# 5. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # Initialize Configuration
    cfg = Config()
    
    try:
        # Load Dataset (Ensure your .npy file is in the directory or provide full path)
        print("[INFO] Loading Defense Vehicle Noise Dataset...")
        train_dataset = NoiseDataset(cfg.DATA_PATH, input_window=cfg.INPUT_DIM)
        
        # Train Model
        trained_model = train_model(cfg, train_dataset)
        
        # Save Model for Future Deployment (Edge/Embedded Systems)
        torch.save(trained_model.state_dict(), "anc_adaptive_filter.pth")
        print("[INFO] Model weights saved to 'anc_adaptive_filter.pth'")
        
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found at {cfg.DATA_PATH}")
        print("[HINT] Please ensure your noise_samples.npy is in the correct directory.")
    except Exception as e:
        print(f"[CRITICAL ERROR] {str(e)}")
