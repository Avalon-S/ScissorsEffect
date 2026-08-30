#!/usr/bin/env python
"""
Train the naturally-trained surrogates this project uses.

    python scripts/train_cifar10_standard.py                      # 4 CIFAR-10 models
    python scripts/train_cifar10_standard.py --dataset cifar100   # the WRN-28-10

CIFAR-10 gives ResNet-18, ResNet-50, VGG-16 and DenseNet-121 as c10_*.pt under
cifar10/standard/. CIFAR-100 gives the WRN-28-10 as Standard_WRN28_10.pt under
cifar100/Linf/, saved unwrapped because models/loader.py rebuilds RobustBench's
WideResNet(28, 100, 10) and applies the CIFAR-100 statistics itself.

Recipe, the same for both:
SGD(lr=0.1, momentum=0.9, weight_decay=5e-4), MultiStepLR([100,150], 0.1),
200 epochs, batch 128, RandomCrop(32, padding=4) + RandomHorizontalFlip, no
label smoothing, no mixup/cutmix, no EMA.

Architectures are CIFAR variants (3x3 stem, no initial max-pool), not ImageNet
architectures fed upsampled 32x32 images. Either way the surrogate consumes
[0,1] images, like every other surrogate here: the CIFAR-10 models carry the
normalisation inside the checkpoint, the CIFAR-100 one gets it from the loader.

The run aborts if a model finishes below --min_acc, so a model that failed to
train cannot reach the experiments.

  python scripts/train_cifar10_standard.py --models resnet18 vgg16 resnet50 densenet121
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------
# CIFAR-variant architectures
# --------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout * self.expansion:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(cout * self.expansion))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + self.short(x))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, stride, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.c3 = nn.Conv2d(cout, cout * 4, 1, bias=False)
        self.b3 = nn.BatchNorm2d(cout * 4)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout * 4:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout * 4, 1, stride, bias=False),
                nn.BatchNorm2d(cout * 4))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = F.relu(self.b2(self.c2(o)))
        o = self.b3(self.c3(o))
        return F.relu(o + self.short(x))


class ResNetC(nn.Module):
    def __init__(self, block, nblocks, num_classes=10):
        super().__init__()
        self.cin = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)   # 3x3 stem, no maxpool
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make(block, 64, nblocks[0], 1)
        self.layer2 = self._make(block, 128, nblocks[1], 2)
        self.layer3 = self._make(block, 256, nblocks[2], 2)
        self.layer4 = self._make(block, 512, nblocks[3], 2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make(self, block, cout, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(block(self.cin, cout, s))
            self.cin = cout * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.layer4(self.layer3(self.layer2(self.layer1(o))))
        o = F.adaptive_avg_pool2d(o, 1).flatten(1)
        return self.fc(o)


VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
             512, 512, 512, 'M', 512, 512, 512, 'M']


class VGGC(nn.Module):
    def __init__(self, cfg=VGG16_CFG, num_classes=10):
        super().__init__()
        layers, cin = [], 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(cin, v, 3, padding=1, bias=False),
                           nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                cin = v
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class DenseLayer(nn.Module):
    def __init__(self, cin, growth, bn_size=4):
        super().__init__()
        self.b1 = nn.BatchNorm2d(cin)
        self.c1 = nn.Conv2d(cin, bn_size * growth, 1, bias=False)
        self.b2 = nn.BatchNorm2d(bn_size * growth)
        self.c2 = nn.Conv2d(bn_size * growth, growth, 3, padding=1, bias=False)

    def forward(self, x):
        o = self.c1(F.relu(self.b1(x)))
        o = self.c2(F.relu(self.b2(o)))
        return torch.cat([x, o], 1)


class Transition(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.bn = nn.BatchNorm2d(cin)
        self.conv = nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        return F.avg_pool2d(self.conv(F.relu(self.bn(x))), 2)


class DenseNetC(nn.Module):
    """DenseNet-121 topology (6,12,24,16), CIFAR stem."""

    def __init__(self, blocks=(6, 12, 24, 16), growth=32, num_classes=10):
        super().__init__()
        c = 2 * growth
        self.conv1 = nn.Conv2d(3, c, 3, padding=1, bias=False)
        layers = []
        for i, n in enumerate(blocks):
            for _ in range(n):
                layers.append(DenseLayer(c, growth))
                c += growth
            if i != len(blocks) - 1:
                layers.append(Transition(c, c // 2))
                c //= 2
        self.features = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(c)
        self.fc = nn.Linear(c, num_classes)

    def forward(self, x):
        o = self.features(self.conv1(x))
        o = F.adaptive_avg_pool2d(F.relu(self.bn(o)), 1).flatten(1)
        return self.fc(o)


BUILDERS = {
    "resnet18": lambda: ResNetC(BasicBlock, [2, 2, 2, 2]),
    "resnet50": lambda: ResNetC(Bottleneck, [3, 4, 6, 3]),
    "vgg16": lambda: VGGC(),
    "densenet121": lambda: DenseNetC(),
}

def _wrn2810():
    """CIFAR-100 WRN-28-10, taken from RobustBench rather than defined here:
    models/loader.py reconstructs the surrogate with exactly this class, so a
    private copy of the architecture would load with strict=False surprises."""
    from robustbench.model_zoo.architectures.wide_resnet import WideResNet
    return WideResNet(depth=28, num_classes=100, widen_factor=10)


BUILDERS_C100 = {"wrn2810": _wrn2810}

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2471, 0.2435, 0.2616)
MEAN_C100 = (0.5071, 0.4865, 0.4409)
STD_C100 = (0.2673, 0.2564, 0.2762)


class Normalized(nn.Module):
    """Wrap so the saved model consumes [0,1] images, exactly like every other
    surrogate in this project (the attack operates in [0,1] space)."""

    def __init__(self, net, mean=MEAN, std=STD):
        super().__init__()
        self.net = net
        self.register_buffer("m", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.m) / self.s)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--data_dir", type=str, default="/root/autodl-tmp/data")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--milestones", type=int, nargs="+", default=[100, 150])
    p.add_argument("--min_acc", type=float, default=None,
                   help="Abort if a finished model is below this. A silently "
                        "broken surrogate must never reach the experiments.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    a = p.parse_args()
    c100 = a.dataset == "cifar100"
    if a.models is None:
        a.models = list(BUILDERS_C100 if c100 else BUILDERS)
    if a.out_dir is None:
        a.out_dir = ("/root/autodl-tmp/models/cifar100/Linf" if c100
                     else "/root/autodl-tmp/models/cifar10/standard")
    if a.min_acc is None:
        # ours finished at 81.07% on CIFAR-100 and above 88% on CIFAR-10
        a.min_acc = 78.0 if c100 else 88.0
    return a


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)

    tr_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    te_tf = transforms.ToTensor()
    root = args.data_dir
    c100 = args.dataset == "cifar100"
    DS = datasets.CIFAR100 if c100 else datasets.CIFAR10
    builders = BUILDERS_C100 if c100 else BUILDERS
    stats = (MEAN_C100, STD_C100) if c100 else (MEAN, STD)
    tr = DS(root, train=True, download=True, transform=tr_tf)
    te = DS(root, train=False, download=True, transform=te_tf)
    trl = DataLoader(tr, args.batch_size, shuffle=True,
                     num_workers=args.workers, pin_memory=True, drop_last=False)
    tel = DataLoader(te, 500, shuffle=False, num_workers=args.workers)

    summary = {}
    for name in args.models:
        torch.manual_seed(0)
        net = Normalized(builders[name](), *stats).to(dev)
        opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=args.wd, nesterov=False)
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, args.milestones, 0.1)
        print(f"\n{'='*70}\n[{name}] params="
              f"{sum(p.numel() for p in net.parameters())/1e6:.2f}M\n{'='*70}",
              flush=True)
        t0 = time.time()
        best = 0.0
        for ep in range(args.epochs):
            net.train()
            tot = corr = 0
            run = 0.0
            for x, y in trl:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                out = net(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                opt.step()
                run += loss.item() * y.size(0)
                corr += (out.argmax(1) == y).sum().item()
                tot += y.size(0)
            sch.step()
            if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
                net.eval()
                c = t = 0
                with torch.no_grad():
                    for x, y in tel:
                        x, y = x.to(dev), y.to(dev)
                        c += (net(x).argmax(1) == y).sum().item()
                        t += y.size(0)
                acc = 100.0 * c / t
                best = max(best, acc)
                print(f"  ep {ep+1:3d}  train_loss={run/tot:.4f} "
                      f"train_acc={100*corr/tot:.2f}  test_acc={acc:.2f}  "
                      f"lr={sch.get_last_lr()[0]:.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

        net.eval()
        c = t = 0
        with torch.no_grad():
            for x, y in tel:
                x, y = x.to(dev), y.to(dev)
                c += (net(x).argmax(1) == y).sum().item()
                t += y.size(0)
        final = 100.0 * c / t
        if final < args.min_acc:
            raise RuntimeError(
                f"{name} finished at {final:.2f}% < --min_acc {args.min_acc}. "
                f"Refusing to save a surrogate that would silently corrupt the "
                f"experiments: a checkpoint that cannot classify the dataset "
                f"must never reach them.")
        if c100:
            # models/loader.py rebuilds WideResNet(28, 100, 10) and applies the
            # CIFAR-100 statistics itself, so what it needs is the unwrapped
            # state dict. It reads the "state_dict" key, so the accuracy and the
            # recipe can travel with it.
            path = os.path.join(args.out_dir, "Standard_WRN28_10.pt")
            payload = net.net.state_dict()
        else:
            path = os.path.join(args.out_dir, f"c10_{name}.pt")
            payload = net.state_dict()
        torch.save({"state_dict": payload, "arch": name,
                    "clean_acc": final, "recipe": vars(args)}, path)
        summary[name] = {"clean_acc": final, "best": best,
                         "minutes": (time.time() - t0) / 60, "path": path}
        print(f"[{name}] final test acc {final:.2f}%  -> {path}")

    with open(os.path.join(args.out_dir, "training_summary.json"), "w") as f:
        json.dump({"recipe": vars(args), "models": summary}, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
