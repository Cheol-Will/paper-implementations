import argparse, os

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

from ResNet import ResNet, ResBlock
from PreActResNet import PreActResNet, PreActResBlock
from StochasticDepth import StDepth, StDepthBlock
from DenseNet import DenseNet, DenseBlock
from ConvMixer import ConvMixer, ConvMixerBlock

import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
CUR_DIR = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default = os.path.join(CUR_DIR, "output"))
parser.add_argument("--model", default="ResNet", 
    choices=["ResNet", "PreActResNet", "DenseNet", "StDepth", "ConvMixer", "MLPMixer"])
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
    num_batches = len(dataloader)
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
    writer = SummaryWriter(f'runs/models')
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")

        model.train(True)
        avg_loss = train_loop(trainloader, model, loss_fn, optimizer, device, args.batch_size)

        model.eval()
        correct, test_loss = test_loop(testloader, model, loss_fn, device)

        print(f'Train Error: {avg_loss:>7f}')
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        # Track best performance, and save the model
        if test_loss < min_test_loss and epoch > 30:
            min_test_loss = test_loss

            model_path = os.path.join(args.output_dir, f"{args.model}_weights_best_model_{epoch}.pth")
            print(f"save model in {model_path}")
            torch.save(model.state_dict(), model_path)

        # write Accuracy and Loss
        n_iter = epoch * len(trainloader)
        writer.add_scalar(f"Acc/{args.model}", correct, n_iter)
        writer.add_scalar(f"Loss/{args.model}", test_loss, n_iter)

        scheduler.step()


    writer.flush()
    writer.close()

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
    if not os.path.isdir(args.output_dir):
        print(f'Output directory {args.output_dir} does not exist; creating it')

    trainloader, testloader = build_loader(args)
        
    if args.model == "ResNet":         
        model = ResNet(ResBlock, channel_list = [16, 32, 64], num_blocks_list = [18, 18, 18], dataset = "cifar-10")
    elif args.model == "PreActResNet":
        model = PreActResNet(PreActResBlock, channel_list = [16, 32, 64], num_blocks_list = [18, 18, 18], dataset = "cifar-10")
    elif args.model == "DenseNet":
        model = DenseNet(DenseBlock, num_blocks_list = [13, 13, 13], growth_rate = 12, in_channels = 16)
    elif args.model == "StDepth":
        model = StDepth(StDepthBlock, channel_list = [16, 32, 64], num_blocks_list = [18, 18, 18])
    elif args.model == "ConvMixer":
        model = ConvMixer(h=256, d=8, p=1, k=15, num_class=10)
    elif args.model == "MLPMixer":
        model = MLPMixer(c = 128, p = 4, c_hidden = 256, s_hidden = 64, h = 32, w = 32, num_class = 10)
    else: 
        print("Check if the model name is correct")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov = True, weight_decay = 0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)

    epochs = 300
    train_epochs(args, model, trainloader, testloader, loss_fn, device, optimizer, scheduler, epochs)
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"output/{args.model}_weights.pth"))

    return 

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)