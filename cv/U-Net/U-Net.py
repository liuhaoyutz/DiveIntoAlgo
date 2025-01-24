import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define U-Net Model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # Middle layer
        self.middle = self.conv_block(512, 1024)
        
        # Decoder
        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)
        
        # Output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            self.conv_block(out_channels, out_channels)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # Output shape: [batch_size, 64, H, W]
        e2 = self.encoder2(nn.MaxPool2d(2)(e1))  # Output shape: [batch_size, 128, H/2, W/2]
        e3 = self.encoder3(nn.MaxPool2d(2)(e2))  # Output shape: [batch_size, 256, H/4, W/4]
        e4 = self.encoder4(nn.MaxPool2d(2)(e3))  # Output shape: [batch_size, 512, H/8, W/8]
        
        # Middle layer
        m = self.middle(nn.MaxPool2d(2)(e4))  # Output shape: [batch_size, 1024, H/16, W/16]
        
        # Decoder
        d4 = self.decoder4(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(m), e4], dim=1))  # Output shape: [batch_size, 512, H/8, W/8]
        d3 = self.decoder3(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(d4), e3], dim=1))  # Output shape: [batch_size, 256, H/4, W/4]
        d2 = self.decoder2(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(d3), e2], dim=1))  # Output shape: [batch_size, 128, H/2, W/2]
        d1 = self.decoder1(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(d2), e1], dim=1))  # Output shape: [batch_size, 64, H, W]
        
        # Output layer
        out = self.final(d1)  # Output shape: [batch_size, out_channels, H, W]
        return out

# Custom Dataset for CIFAR-10 Segmentation
class CIFAR10Segmentation(Dataset):
    def __init__(self, root, train=True, transform=None, target_class=0):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        self.target_class = target_class  # Class to treat as foreground

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Generate binary mask: 1 for target class, 0 for others
        mask = np.array(label == self.target_class, dtype=np.float32)  # Convert to NumPy array
        return image, torch.tensor(mask, dtype=torch.float32)

# Data Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 Dataset
train_dataset = CIFAR10Segmentation(root='./data', train=True, transform=transform, target_class=0)
test_dataset = CIFAR10Segmentation(root='./data', train=False, transform=transform, target_class=0)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize Model, Loss Function, and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# Validation Function
def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), masks)
            running_loss += loss.item()
    return running_loss / len(test_loader)

# Training and Validation Loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, test_loader, criterion, device)
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Inference Function
def inference(model, image):
    model.eval()
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)
        output = model(image)
        output = torch.sigmoid(output).squeeze().cpu().numpy()
    return output

# Example Inference
sample_image, _ = test_dataset[0]
predicted_mask = inference(model, sample_image)
print("Predicted Mask Shape:", predicted_mask.shape)