import argparse, os
import datetime

import torch
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from dataloader.loader import build_loader
from models import build_model
from utils import tensorboard_write

def train_loop(dataloader, model, loss_fn, optimizer, device, batch_size):
    # extract num_classes from nn.Linear()
    num_classes = model.clf[-1].out_features

    cutmix = v2.CutMix(num_classes=num_classes) # modify to get output shape from model's output head 
    mixup = v2.MixUp(num_classes=num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    size = len(dataloader.dataset)
    running_loss = 0.
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # X, y = cutmix_or_mixup(X, y)
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

def train_epochs(args, model, trainloader, testloader, loss_fn, optimizer, scheduler, device, restart_epoch=None):
    min_test_loss = 1_000_000
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y%m%d')
    writer = SummaryWriter(f'runs/{args.dataset}/models/{args.model}_{args.epochs}_{timestamp}{args.exp_name}')
    model_path = os.path.join(args.output_dir, f"{args.dataset}_{args.model}_best_model_{timestamp}{args.exp_name}.pth")

    # restore checkpoint to keep training
    if args.checkpoint_start_from is not None:
        model_path = os.path.join(args.output_dir, args.checkpoint_start_from)
        print(f"\nRestore model from {model_path}")

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
    else:
        epoch = 0
    
    while epoch < args.epochs:
        epoch += 1
        print(f"Epoch {epoch}\n-------------------------------")

        model.train(True)
        avg_loss = train_loop(trainloader, model, loss_fn, optimizer, device, args.batch_size)
        scheduler.step()
        
        model.eval()
        correct, test_loss = test_loop(testloader, model, loss_fn, device)

        print(f'Train : \n Avg loss: {avg_loss:>8f}')
        print(f"Test : \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        # Track best performance and save the model
        if test_loss < min_test_loss:
            min_test_loss = test_loss
                        
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": args.epochs,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }

            print(f"save model in {model_path}")
            torch.save(checkpoint, model_path)

        # write Accuracy and Loss
        n_iter = epoch * len(trainloader)
        tensorboard_write(writer, n_iter, correct, test_loss, avg_loss)
    
        if epoch % 10 == 9:
            writer.flush()

    writer.close()

def main(args):
    if not os.path.isdir(args.output_dir):
        print(f'Output directory {args.output_dir} does not exist; creating it')
        os.makedirs(args.output_dir, exist_ok=True)
        
    trainloader, testloader = build_loader(args.batch_size)
    model = build_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-08, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    
    train_epochs(args, model, trainloader, testloader, loss_fn, optimizer, scheduler, device)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y%m%d')
    torch.save(checkpoint, os.path.join(args.output_dir, f"{args.dataset}_{args.model}_{args.epochs}_{timestamp}{args.exp_name}.pth"))

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default = "output")
    parser.add_argument("--dataset", default = "cifar-10")
    parser.add_argument("--model", default="ResNet", 
                        choices=["ResNet", "PreActResNet", "DenseNet", "StDepth", "ConvMixer", "MLPMixer", "ViT", "FractalNet"])
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    # parser.add_argument("--restore_from_checkpoint", default=True, type=bool_flag)
    parser.add_argument("--checkpoint_start_from", default = None)

    args = parser.parse_args()
    main(args)