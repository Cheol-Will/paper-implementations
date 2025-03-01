import argparse, os, time

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from DenseNet import DenseNet, DenseBlock
from ResNet import ResNet, ResBlock
from PreActResNet import PreActResNet, PreActResBlock

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

CUR_DIR = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default = os.path.join(CUR_DIR, "output"))
parser.add_argument("--model", default="ResNet", choices=["ResNet", "PreActResNet", "DenseNet"])
parser.add_argument("--batch_size", default=64, type=int)

def imshow(img, filename="output.png"):
    if not os.path.isdir(args.output_dir):
        print('Output directory "%s" does not exist; creating it' % args.output_dir)
        os.makedirs(args.output_dir)

    path = os.path.join(args.output_dir, filename)
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")  
    plt.savefig(path, bbox_inches="tight")  
    print(f"Image saved as {filename} in {args.output_dir}")  

def train_loop(dataloader, model, loss_fn, optimizer, device, batch_size):
    size = len(dataloader.dataset)
    running_loss = 0.
    last_loss = 0.
    
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

        if batch % 100 == 99:
            last_loss = running_loss / 100 # loss per bath
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {last_loss:>7f}  [{current:>5d}/{size:>5d}]")

    return last_loss

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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct, test_loss

def train_epochs(args, model, trainloader, testloader, loss_fn, device, optimizer, EPOCHS):
    best_tloss = 1_000_000
    start = time.time()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")

        model.train(True)
        avg_loss = train_loop(trainloader, model, loss_fn, optimizer, device, args.batch_size)
        model.eval()
        correct, test_loss = test_loop(testloader, model, loss_fn, device)
        print(f'Train Error: {avg_loss:>7f}')
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        # Track best performance, and save the model's state
        if test_loss < best_tloss and epoch > 30:
            best_tloss = test_loss
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(CUR_DIR, f"output/{args.model}_weights_{timestamp}_{epoch}.pth")
            print(f"save model in {model_path}")
            torch.save(model.state_dict(), model_path)

    end = time.time()
    print(f"{end - start:.5f} sec")

def build_loader(args):
    transform = transforms.Compose(
        [
            # Need to add shifting
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToTensor(),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = args.batch_size
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print('Training set has {} instances'.format(len(trainset)))
    print('Validation set has {} instances'.format(len(testset)))

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader

def main(args):

    # writer = SummaryWriter(f'runs/cifar-10-{parser.model}')
    trainloader, testloader = build_loader(args)

    if args.model == "ResNet":         
        model = ResNet(ResBlock, num_blocks_list = [18, 18, 18], dataset = "cifar-10")
    elif args.model == "PreActResNet":
        model = PreActResNet(PreActResBlock, num_blocks_list = [18, 18, 18], dataset = "cifar-10")
    elif args.model == "DenseNet":
        model = DenseNet(DenseBlock, num_blocks_list = [13, 13, 13], growth_rate = 12, in_channels = 16)
    else: 
        print("Check if the model name is correct")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov = True, weight_decay = 0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)

    epochs = 300
    train_epochs(args, model, trainloader, testloader, loss_fn, device, optimizer, epochs)

    torch.save(model.state_dict(), os.path.join(CUR_DIR, f"output/{args.model}_weights.pth"))

    return 

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)