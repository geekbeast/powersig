import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def create_k_layer_mlp(n, k):
    """
    Creates a k-layer MLP with the following architecture:
    - Input size: n
    - First layer: full input size n (non-linear)
    - Middle layers: log2(n) neurons (narrowing)
    - Output layer: single value
    
    Args:
        n (int): Input size
        k (int): Number of layers
    
    Returns:
        nn.Module: PyTorch neural network
    """
    
    class SwooshActivation(nn.Module):
        """Swoosh activation function - smooth approximation of ReLU"""
        def __init__(self):
            super(SwooshActivation, self).__init__()
            
        def forward(self, x):
            return x * torch.sigmoid(x)
    
    class KLayerMLP(nn.Module):
        def __init__(self, input_size, num_layers):
            super(KLayerMLP, self).__init__()
            
            self.input_size = input_size
            self.num_layers = num_layers
            
            # Calculate log2 of input size for middle layers
            self.middle_size = max(5, int(math.log2(input_size)))
            
            # Create layers
            layers = []
            
            # First layer: full input size
            layers.append(nn.Linear(input_size, input_size))
            layers.append(SwooshActivation())
            
            # Middle layers: log2(n) neurons
            for i in range(num_layers - 2):
                if i == 0:
                    # First middle layer: from input_size to middle_size
                    layers.append(nn.Linear(input_size, self.middle_size))
                else:
                    # Subsequent middle layers: middle_size to middle_size
                    layers.append(nn.Linear(self.middle_size, self.middle_size))
                layers.append(SwooshActivation())
            
            # Output layer: single value
            if num_layers > 1:
                layers.append(nn.Linear(self.middle_size, 1))
            else:
                # If only one layer, go directly from input to output
                layers.append(nn.Linear(input_size, 1))
            
            self.network = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.network(x)
    
    return KLayerMLP(n, k)


def create_k_layer_mlp_classification(n, k, num_classes):
    """
    Creates a k-layer MLP for multi-class classification with the following architecture:
    - Input size: n
    - First layer: full input size n (non-linear)
    - Middle layers: log2(n) neurons (narrowing)
    - Output layer: num_classes (for classification)
    
    Args:
        n (int): Input size
        k (int): Number of layers
        num_classes (int): Number of output classes
    
    Returns:
        nn.Module: PyTorch neural network
    """
    
    class SwooshActivation(nn.Module):
        """Swoosh activation function - smooth approximation of ReLU"""
        def __init__(self):
            super(SwooshActivation, self).__init__()
            
        def forward(self, x):
            return x * torch.sigmoid(x)
    
    class KLayerMLPClassification(nn.Module):
        def __init__(self, input_size, num_layers, num_classes):
            super(KLayerMLPClassification, self).__init__()
            
            self.input_size = input_size
            self.num_layers = num_layers
            self.num_classes = num_classes
            
            # Calculate log2 of input size for middle layers
            self.middle_size = max(5, int(math.log2(input_size)))
            
            # Create layers
            layers = []
            
            # First layer: full input size
            layers.append(nn.Linear(input_size, input_size))
            layers.append(SwooshActivation())
            
            # Middle layers: log2(n) neurons
            for i in range(num_layers - 2):
                if i == 0:
                    # First middle layer: from input_size to middle_size
                    layers.append(nn.Linear(input_size, self.middle_size))
                else:
                    # Subsequent middle layers: middle_size to middle_size
                    layers.append(nn.Linear(self.middle_size, self.middle_size))
                layers.append(SwooshActivation())
            
            # Output layer: num_classes (no softmax - will be applied in loss)
            if num_layers > 1:
                layers.append(nn.Linear(self.middle_size, num_classes))
            else:
                # If only one layer, go directly from input to output
                layers.append(nn.Linear(input_size, num_classes))
            
            self.network = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.network(x)
    
    return KLayerMLPClassification(n, k, num_classes)

# Example usage and training function
def train_mlp_model_regression(model, kernel, y_train, epochs=100, lr=0.001, optimizer_type='adam'):
    """
    Train the MLP model for regression using PyTorch.
    Uses MSE loss for regression tasks.
    
    Args:
        model: PyTorch model
        kernel: Kernel matrix (torch.Tensor) - shape (batch_size, input_size)
        y_train: Training targets (torch.Tensor) - shape (batch_size,) or (batch_size, 1) for regression
        epochs: Number of training epochs
        lr: Learning rate
        optimizer_type: Type of optimizer ('adam' or 'lbfgs')
    
    Returns:
        tuple: (trained_model, training_losses)
    """
    # Ensure y_train is 2D for regression (batch_size, 1)
    if y_train.dim() == 1:
        y_train = y_train.unsqueeze(1)
    
    # Use MSE loss for regression
    criterion = nn.MSELoss()
    
    if optimizer_type.lower() == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=100000, tolerance_grad=np.finfo(float).eps, tolerance_change=np.finfo(float).eps, lr=lr, line_search_fn='strong_wolfe')
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    if optimizer_type.lower() == 'lbfgs':
        # L-BFGS training loop
        def closure():
            optimizer.zero_grad()
            # Process entire batch at once
            outputs = model(kernel)
            loss = criterion(outputs, y_train)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        losses.append(loss.item())
        
        print(f'LBFGS Loss: {loss.item():.4f}')
    else:
        # Adam training loop
        for epoch in range(epochs):
            # Forward pass - process entire batch
            outputs = model(kernel)
            loss = criterion(outputs, y_train)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
    
    return model, losses


def train_mlp_model_classification(model, kernel, y_train, epochs=100, lr=0.001, optimizer_type='adam'):
    """
    Train the MLP model for classification using PyTorch with batch processing.
    Handles integer labels internally with one-hot encoding.
    
    Args:
        model: PyTorch model
        kernel: Kernel matrix (torch.Tensor) - shape (batch_size, input_size)
        y_train: Training targets (torch.Tensor) - shape (batch_size,) with integer labels
        epochs: Number of training epochs
        lr: Learning rate
        optimizer_type: Type of optimizer ('adam' or 'lbfgs')
    
    Returns:
        tuple: (trained_model, training_losses)
    """
    # Get number of classes from model output size
    num_classes = model.network[-1].out_features
    
    # Convert integer labels to one-hot encoding
    y_train_onehot = torch.zeros(len(y_train), num_classes, dtype=torch.float32)
    y_train_onehot.scatter_(1, y_train.unsqueeze(1), 1)
    
    # Use cross-entropy loss
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type.lower() == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=100000, tolerance_grad=np.finfo(float).eps, tolerance_change=np.finfo(float).eps, lr=lr, line_search_fn='strong_wolfe')
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    if optimizer_type.lower() == 'lbfgs':
        # L-BFGS training loop
        def closure():
            optimizer.zero_grad()
            # Process entire batch at once
            outputs = model(kernel)
            loss = criterion(outputs, y_train_onehot)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        losses.append(loss.item())
        
        print(f'LBFGS Loss: {loss.item():.4f}')
    else:
        # Adam training loop
        for epoch in range(epochs):
            # Forward pass - process entire batch
            outputs = model(kernel)
            loss = criterion(outputs, y_train_onehot)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
    
    return model, losses


