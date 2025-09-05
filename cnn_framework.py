#!/usr/bin/env python3
"""
cnn_framework.py

A minimal PyTorch CNN for MNIST (CPU-friendly) to complement the math-based CNN.
Features:
- Small CNN model with Conv2d, ReLU, MaxPool2d, Linear
- MNIST data loading (train/test)
- Training loop and evaluation
- Reports parameter count, FLOPs (per-image, analytical), and runtime
- No advanced optimizations; plain PyTorch only

Usage:
  python cnn_framework.py --epochs 1 --batch-size 64 --lr 0.01
"""

import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True),  # 28x28 -> 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                             # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True), # 14x14 -> 14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                             # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_flops_per_image(model: nn.Module, input_size: Tuple[int, int, int] = (1, 28, 28)) -> int:
    """
    Compute approximate FLOPs per image analytically for common layers.
    Conventions:
      - Conv2d: Hout*Wout*Cout * (Cin*K*K*2) + Cout*Hout*Wout (bias adds)
      - Linear: Nout * (Nin*2) + Nout (bias adds)
      - ReLU, MaxPool: ignored for FLOPs (very small vs MACs)
    """
    flops = 0

    C, H, W = input_size
    xC, xH, xW = C, H, W
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            Cin = m.in_channels
            Cout = m.out_channels
            K = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
            S = m.stride[0] if isinstance(m.stride, tuple) else m.stride
            P = m.padding[0] if isinstance(m.padding, tuple) else m.padding
            # output dims
            Hout = (xH + 2 * P - K) // S + 1
            Wout = (xW + 2 * P - K) // S + 1
            # FLOPs
            macs = Hout * Wout * Cout * (Cin * K * K)
            flops += 2 * macs  # mul+add
            if m.bias is not None:
                flops += Hout * Wout * Cout  # bias adds
            # update current shape
            xH, xW, xC = Hout, Wout, Cout
        elif isinstance(m, nn.MaxPool2d):
            K = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            S = m.stride if isinstance(m.stride, int) or m.stride is None else m.stride[0]
            if S is None:
                S = K
            P = m.padding if isinstance(m.padding, int) else m.padding[0]
            Hout = (xH + 2 * P - K) // S + 1
            Wout = (xW + 2 * P - K) // S + 1
            xH, xW = Hout, Wout
        elif isinstance(m, nn.Flatten):
            # shape changes in classifier phase
            pass
        elif isinstance(m, nn.Linear):
            Nin = m.in_features
            Nout = m.out_features
            flops += (Nin * 2) * Nout
            if m.bias is not None:
                flops += Nout
        else:
            # ReLU/others: ignore
            pass

    return flops


def get_data_loaders(batch_size: int, num_workers: int = 2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    epoch_time = time.time() - start_time
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc, epoch_time


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    eval_time = time.time() - start_time
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc, eval_time


def main():
    parser = argparse.ArgumentParser(description="Minimal PyTorch CNN for MNIST")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cpu")  # CPU-only to keep compatibility

    # Data
    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size, num_workers=args.num_workers)

    # Model
    model = SmallCNN().to(device)
    params = count_parameters(model)
    flops_per_image = compute_flops_per_image(model, input_size=(1, 28, 28))

    print("Model Summary:")
    print(f"  - Parameters: {params:,}")
    print(f"  - Approx FLOPs/image: {flops_per_image:,}")

    # Loss/Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Train
    total_train_time = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_time = train_one_epoch(model, train_loader, criterion, optimizer, device)
        total_train_time += tr_time
        print(f"Epoch {epoch}/{args.epochs}: train_loss={tr_loss:.4f} acc={tr_acc*100:.2f}% time={tr_time:.2f}s")

    # Evaluate
    te_loss, te_acc, te_time = evaluate(model, test_loader, criterion, device)

    # Report runtime
    print("\nRuntime Report:")
    print(f"  - Total train time: {total_train_time:.2f}s")
    print(f"  - Eval time: {te_time:.2f}s")

    print("\nTest Metrics:")
    print(f"  - Test loss: {te_loss:.4f}")
    print(f"  - Test accuracy: {te_acc*100:.2f}%")


if __name__ == "__main__":
    main()
