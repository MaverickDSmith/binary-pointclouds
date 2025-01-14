from difflogic import LogicLayer, GroupSum
import torch
from torch.utils.data import DataLoader
from dataloaders.DiffArrayDataset import BitArrayDataset
import torch.nn.functional as F

# Dataset and DataLoader setup
train_dataset = BitArrayDataset("/home/hi5lab/pointcloud_data/storage_test_two/slice64", "train")
val_dataset = BitArrayDataset("/home/hi5lab/pointcloud_data/storage_test_two/slice64", "val")
test_dataset = BitArrayDataset("/home/hi5lab/pointcloud_data/storage_test_two/slice64", "test")

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

# Define the model
class CustomLogicModel(torch.nn.Module):
    def __init__(self, device):
        super(CustomLogicModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            LogicLayer(274625, 512_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(512_000, 512_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(512_000, 512_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(512_000, 512_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(512_000, 1_024_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(1_024_000, 1_024_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(1_024_000, 1_024_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(1_024_000, 1_024_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(1_024_000, 2_048_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            GroupSum(k=40, tau=120)
        )

    def forward(self, x):
        return self.model(x)
    



# Device and model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomLogicModel(device).to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
class_criterion = torch.nn.CrossEntropyLoss()  

# Validation function
def validate(model, val_loader, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # No gradient computation during evaluation
        for x in val_loader:
            # anchor, pos, neg, num_slices = x
            anchor, _ = x
            anchor_input, anchor_label, _ = anchor
            anchor_input, anchor_label = anchor_input.to(device), anchor_label.to(device)

            outputs = model(anchor_input)
            loss = class_criterion(outputs, anchor_label)
            total_loss += loss.item()

            # Optional: Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == anchor_label).sum().item()
            total += anchor_label.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total * 100
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# Test function
def test(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for x in test_loader:
            anchor, _ = x
            anchor_input, anchor_label, _ = anchor
            outputs = model(anchor_input)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == anchor_label).sum().item()
            total += anchor_label.size(0)
    
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0.0
    for batch_idx, x in enumerate(train_loader):
        anchor, _ = x
        anchor_input, anchor_label, _ = anchor
        anchor_input, anchor_label = anchor_input.to(device), anchor_label.to(device)
        
        # Forward pass
        outputs = model(anchor_input)
    
        # Loss computation
        loss = class_criterion(outputs, anchor_label)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if batch_idx % 50 == 0:  # Log every 10 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    
    # Print epoch loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Total Training Loss: {total_loss:.4f}")

    # Validation
    val_loss, val_accuracy = validate(model, val_loader, device)

# Final test evaluation
test_accuracy = test(model, test_loader, device)
