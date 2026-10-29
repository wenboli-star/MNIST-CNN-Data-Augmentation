import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import SmallCNN

ROOT = Path(__file__).resolve().parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loaders(args):
    standard = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_transform = standard
    if args.variant == 'augmented':
        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.08, 0.08)),
            standard,
        ])
    train_data = datasets.MNIST(ROOT / 'data', train=True, download=True, transform=train_transform)
    validation_data = datasets.MNIST(ROOT / 'data', train=True, download=False, transform=standard)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(train_data))
    train_idx, val_idx = indices[:55000], indices[55000:]
    if args.limit_train:
        train_idx = train_idx[:args.limit_train]
    train_loader = DataLoader(Subset(train_data, train_idx.tolist()), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(Subset(validation_data, val_idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return train_loader, val_loader


def one_epoch(model, loader, loss_fn, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)
    count, correct, total_loss = 0, 0, 0.0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()
            count += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total_loss += loss.item() * y.size(0)
    return total_loss / count, correct / count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['baseline', 'augmented'], required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--limit-train', type=int, default=0, help='For quick smoke tests only')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=0.001)
    args = p.parse_args()
    if args.epochs < 1 or args.limit_train < 0:
        p.error('epochs must be >= 1, limit-train must be >= 0')
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = loaders(args)
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    out = ROOT / 'results' / args.variant
    out.mkdir(parents=True, exist_ok=True)
    best = -1
    records = []
    print(f'Variant={args.variant}, device={device}, train_samples={len(train_loader.dataset)}, val_samples={len(val_loader.dataset)}', flush=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = one_epoch(model, train_loader, loss_fn, opt, device)
        val_loss, val_acc = one_epoch(model, val_loader, loss_fn, None, device)
        records.append(dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc, val_loss=val_loss, val_accuracy=val_acc))
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), out / 'best.pth')
        print(f'epoch {epoch}/{args.epochs} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}', flush=True)
        with (out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding='utf-8')
    print('Saved:', out / 'best.pth')


if __name__ == '__main__':
    main()
