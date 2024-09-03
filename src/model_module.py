
import torch.nn as nn

# Define a simple neural network model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # Changed in_channels to 3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=128)  # Updated in_features to match the output size after pooling
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Assuming 10 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64*7*7)  # Flatten the tensor
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
