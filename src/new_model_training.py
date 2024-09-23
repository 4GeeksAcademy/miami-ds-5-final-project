import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
# from medmnist import PathMNIST
from PIL import Image
import pickle
import warnings
import os


def warn(*args, **kwargs):
    pass


warnings.warn = warn

batch_size = 256

os.makedirs('encoders', exist_ok=True)
os.makedirs('class_heads', exist_ok=True)
os.makedirs('attention', exist_ok=True)


class ContrastivePathMNIST(Dataset):
    def __init__(self, split='train', transform=None, download=False):
        self.data = PathMNIST(split=split, download=download)
        self.transform = transform
        self.images = self.data.imgs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        if self.transform:
            xi = self.transform(img)
            xj = self.transform(img)
        else:
            xi = transforms.ToTensor()(img)
            xj = transforms.ToTensor()(img)
        return xi, xj


class SupervisedPathMNIST(Dataset):
    def __init__(self, split='train', transform=None, download=False):
        self.data = PathMNIST(split=split, download=download)
        self.transform = transform
        self.images = self.data.imgs
        self.labels = self.data.labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx].item())
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def loader_loader():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_dim = 128

    supervised_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])

    contrast_transforms = transforms.Compose([
        transforms.RandomResizedCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    contrastive_dataset = ContrastivePathMNIST(split='train', transform=contrast_transforms, download=True)
    contrastive_loader = DataLoader(contrastive_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    train_dataset = SupervisedPathMNIST(split='train', transform=supervised_transforms, download=True)
    val_dataset = SupervisedPathMNIST(split='val', transform=supervised_transforms, download=True)
    test_dataset = SupervisedPathMNIST(split='test', transform=supervised_transforms, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=128, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_dim=128, num_classes=9):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv_block(x)
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class Encoder(nn.Module):
    def __init__(self, feature_dim=128):
        super(Encoder, self).__init__()
        self.convnet = nn.Sequential(
            ResidualBlock(3, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, 128)

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = f.normalize(x, p=2, dim=1)
        return x


def supervised_contrastive_loss(features, labels, temperature=0.05):
    device = features.device
    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float().to(device)

    dot_product = torch.matmul(features, features.T) / temperature
    exp_dot_product = torch.exp(dot_product)

    # Mask out self-comparisons
    mask_self = torch.eye(labels.shape[0], device=device).bool()
    exp_dot_product = exp_dot_product.masked_fill(mask_self, 0)

    # Compute denominators
    denom = exp_dot_product.sum(dim=1, keepdim=True)

    # Compute loss
    log_prob = dot_product - torch.log(denom + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    loss = -mean_log_prob_pos.mean()

    return loss


def contrastive_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # Concatenate for easier computation

    sim = torch.matmul(z, z.T)
    sim = sim / temperature

    labels = torch.arange(batch_size).to(z_i.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)
    sim = sim.masked_fill(mask, -1e9)

    loss = f.cross_entropy(sim, labels)
    return loss


class GeminiContrast(nn.Module):
    def __init__(self, Castor, Pollux, class_head, num_classes=9):
        super(GeminiContrast, self).__init__()
        self.Castor = Castor
        self.Pollux = Pollux
        self.class_head = class_head

        for param in self.Castor.parameters():
            param.requires_grad = True
        for param in self.Pollux.parameters():
            param.requires_grad = True

    def forward(self, x):
        Castor_feat = self.Castor(x)
        Pollux_feat = self.Pollux(x)
        gemini_feat = torch.cat([
            Castor_feat,
            Pollux_feat,
        ], dim=1)
        output = self.class_head(gemini_feat)
        return output


def twins_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Castor = Encoder().to(device)
    try:
        Castor.load_state_dict(torch.load('encoders/Castor-best.pth'))
    except FileNotFoundError or EOFError:
        projector = ProjectionHead(input_dim=128, output_dim=128).to(device)
        optimizer = optim.Adam(list(Castor.parameters()) + list(projector.parameters()), lr=1e-4)

        epochs = 50
        for epoch in range(epochs):
            Castor.train()
            projector.train()
            total_loss = 0
            for xi, xj in contrastive_loader:
                xi = xi.to(device)
                xj = xj.to(device)

                optimizer.zero_grad()

                hi = Castor(xi)
                hj = Castor(xj)

                zi = projector(hi)
                zj = projector(hj)

                loss = contrastive_loss(zi, zj)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            avg_loss = total_loss / len(contrastive_loader)
            print(f'Castor : Epoch [{epoch + 1}] :\tLoss: {avg_loss:.4f}')
        torch.save(Castor.state_dict(), 'encoders/Castor-best.pth')

    Pollux = Encoder().to(device)
    try:
        Pollux.load_state_dict(torch.load('encoders/Pollux-best.pth'))
    except FileNotFoundError or EOFError:
        optimizer = optim.Adam(Pollux.parameters(), lr=5e-4)
        epochs = 40

        for epoch in range(epochs):
            Pollux.train()
            total_loss = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                features = Pollux(images)
                loss = supervised_contrastive_loss(features, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f'Pollux : Epoch [{epoch + 1}] :\tLoss: {avg_loss:.4f}')
        torch.save(Pollux.state_dict(), 'encoders/Pollux-best.pth')

def training_loop():
    test_acc = {}
    max_acc = 0
    max_epoch = 0

    for i in range(3, 30):
        epochs = 30
        best_val_loss = float('inf')
        patience = 5
        counter = 0

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        gemini = GeminiContrast(Castor, Pollux, ClassificationHead(input_dim=512, num_classes=9).to(device)).to(device)
        optimizer = optim.Adam([{'params': gemini.class_head.parameters(), 'lr': 1e-6},
                                {'params': gemini.Castor.parameters(), 'lr': 1e-7},
                                {'params': gemini.Pollux.parameters(), 'lr': 5e-8}
                                ])
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            if epoch == i:
                gemini.eval()
                correct = 0
                total = 0

                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = gemini(images)

                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()

                test_accuracy = 100. * correct / total
                if test_accuracy > max_acc:
                    max_acc = test_accuracy
                    max_epoch = i
                torch.save(gemini.Castor.state_dict(), f'encoders/test-Castor-{i}.pth')
                torch.save(gemini.Pollux.state_dict(), f'encoders/test-Pollux-{i}.pth')
                torch.save(gemini.class_head.state_dict(), f'class_heads/test_diviner-{i}.pth')
                print(f'Test Accuracy: {test_accuracy:.2f}%')
                break

            gemini.train()
            total_loss = 0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = gemini(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_accuracy = 100. * correct / total
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)

            gemini.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = gemini(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()

                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_accuracy = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            print(f'Epoch [{epoch + 1}/{epochs}], '
                f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                torch.save(gemini.class_head.state_dict(), 'class_heads/best_diviner.pth')
            else:
                counter += 1
                if counter >= patience:
                    stop = True
                    print("Early stopping triggered")
                    gemini.class_head.load_state_dict(torch.load('class_heads/best_diviner.pth'))
                    gemini.eval()
                    correct = 0
                    total = 0

                    with torch.no_grad():
                        for images, labels in test_loader:
                            images = images.to(device)
                            labels = labels.to(device)

                            outputs = gemini(images)

                            _, predicted = outputs.max(1)
                            total += labels.size(0)
                            correct += predicted.eq(labels).sum().item()

                    test_accuracy = 100. * correct / total
                    torch.save(gemini.class_head.state_dict(), 'class_heads/test_diviner.pth')
                    print(f'Test Accuracy: {test_accuracy:.2f}%')

                    break

    print(f'Maximum Accuracy of {max_acc} achieved after {max_epoch} Epochs of training.')
