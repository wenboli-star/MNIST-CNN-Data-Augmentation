import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SmallCNN

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['baseline', 'augmented'], required=True)
    args = parser.parse_args()
    path = ROOT / 'results' / args.variant / 'best.pth'
    if not path.is_file():
        parser.error(f'No model weights at {path}; train the model first')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_data = datasets.MNIST(ROOT / 'data', train=False, download=True, transform=transform)
    loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            predictions = logits.argmax(1).cpu()
            total += y.size(0)
            correct += (predictions == y).sum().item()
            total_loss += loss_fn(logits, y.to(device)).item() * y.size(0)
            for true, pred in zip(y.tolist(), predictions.tolist()):
                confusion[true, pred] += 1
    result = {'variant': args.variant, 'test_samples': total, 'test_accuracy': correct / total, 'test_loss': total_loss / total, 'confusion_matrix': confusion.tolist()}
    destination = path.parent / 'test_results.json'
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'{args.variant}: accuracy={result["test_accuracy"]:.4f}, loss={result["test_loss"]:.4f}')
    print('Saved:', destination)


if __name__ == '__main__':
    main()
