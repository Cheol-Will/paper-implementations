import argparse, os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

from resnet import ResNet, ResBlock
from preactresnet import PreActResNet, PreActResBlock
from stochastic_depth import StDepth, StDepthBlock
from densenet import DenseNet, DenseBlock
from convmixer import ConvMixer, ConvMixerBlock
from mlpmixer import MLPMixer, MixerBlock
from vision_transformer import VisionTransformer, Encoder

import numpy as np
import matplotlib.pyplot as plt

NUM_CLASSES = 10

def train_loop(dataloader, model, loss_fn, optimizer, device, batch_size):
    cutmix = v2.CutMix(num_classes=NUM_CLASSES)
    mixup = v2.MixUp(num_classes=NUM_CLASSES)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    size = len(dataloader.dataset)
    running_loss = 0.
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = cutmix_or_mixup(X, y)
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        if batch % 100 == 99:
            avg_loss, current = running_loss / (batch + 1), batch * batch_size + len(X)
            print(f"loss: {avg_loss:>7f}  [{current:>5d}/{size:>5d}]")

    return running_loss/num_batches

def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    return correct, test_loss

def train_epochs(args, model, trainloader, testloader, loss_fn, device, optimizer, scheduler, EPOCHS):
    min_test_loss = 1_000_000
    writer = SummaryWriter(f'runs/models/{args.model}')
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")

        model.train(True)
        avg_loss = train_loop(trainloader, model, loss_fn, optimizer, device, args.batch_size)
        scheduler.step()
        model.eval()
        correct, test_loss = test_loop(testloader, model, loss_fn, device)

        print(f'Train Error: {avg_loss:>7f}')
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        # Track best performance, and save the model
        if test_loss < min_test_loss:
            min_test_loss = test_loss

            model_path = os.path.join(args.output_dir, f"{args.model}_weights_best_model.pth")
            print(f"save model in {model_path}")
            torch.save(model.state_dict(), model_path)

        # write Accuracy and Loss
        n_iter = epoch * len(trainloader)
        writer.add_scalar(f"Acc", correct, n_iter)
        writer.add_scalar(f"Loss", test_loss, n_iter)

    writer.flush()
    writer.close()

def build_loader(batch_size):
    # Aug: RandAug, mixup, CutMix, random erasing, gradient clipping, timm
    transform = transforms.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandAugment(),
        v2.ToTensor(),
        # v2.ToImage(),
        # v2.ToDtype(torch.float32, scale = True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        v2.RandomErasing(),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print('Training set has {} instances'.format(len(trainset)))
    print('Validation set has {} instances'.format(len(testset)))

    return trainloader, testloader

def main(args):
    if not os.path.isdir(args.output_dir):
        print(f'Output directory {args.output_dir} does not exist; creating it')

    trainloader, testloader = build_loader(args.batch_size)
        
    if args.model == "ResNet":         
        model = ResNet(ResBlock, channel_list=[16, 32, 64], num_blocks_list=[18, 18, 18], dataset="cifar-10")
    elif args.model == "PreActResNet":
        model = PreActResNet(PreActResBlock, channel_list=[16, 32, 64], num_blocks_list=[18, 18, 18], dataset="cifar-10")
    elif args.model == "DenseNet":
        model = DenseNet(DenseBlock, num_blocks_list=[13, 13, 13], growth_rate=12, in_channels=16)
    elif args.model == "StDepth":
        model = StDepth(StDepthBlock, channel_list=[16, 32, 64], num_blocks_list=[18, 18, 18])
    elif args.model == "MLPMixer":
        model = MLPMixer(hidden_dim=256, patch_size=4, c_hidden=1024, s_hidden=128, depth=8, height=32, width=32, num_classes=10)
    elif args.model == "ConvMixer":
        model = ConvMixer(hidden_dim=256, depth=16, patch_size=1, kernel_size=15, num_classes=10)
    elif args.model == "ViT":
        model = VisionTransformer(image_size=(32, 32), hidden_dim=128, num_heads=4, mlp_dim=512, patch_size=4, depth=12, num_classes=10)
    else: 
        print("Check if the model name is correct")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-08, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

    epochs = 100
    train_epochs(args, model, trainloader, testloader, loss_fn, device, optimizer, scheduler, epochs)
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model}_weights.pth"))

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default = "output")
    parser.add_argument("--model", default="ResNet", 
                        choices=["ResNet", "PreActResNet", "DenseNet", "StDepth", "ConvMixer", "MLPMixer", "ViT"])
    parser.add_argument("--batch_size", default=64, type=int)
    args = parser.parse_args()
    main(args)