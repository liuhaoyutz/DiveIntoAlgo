import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define the AlexNet model class
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        # Convolutional layers with ReLU activation and max pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # Input: 3x32x32, Output: 64x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # Output: 64x8x8
            nn.Conv2d(64, 192, kernel_size=3, padding=1),  # Output: 192x8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # Output: 192x4x4
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # Output: 384x4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # Output: 256x4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Output: 256x4x4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # Output: 256x2x2
        )
        # Fully connected layers with Dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),  # Adjusted for CIFAR-10 input size
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)  # Extract features from the input image
        x = x.view(x.size(0), 256 * 2 * 2)  # Flatten tensor for fully connected layers
        x = self.classifier(x)  # Pass through the classifier
        return x

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters setup
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally for data augmentation
    transforms.RandomCrop(32, padding=4),  # Randomly crop images with padding for data augmentation
    transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray into tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and standard deviation
])

# Load datasets
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = AlexNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# Test the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')

# Save the model
torch.save(model.state_dict(), 'alexnet_cifar10.pth')
