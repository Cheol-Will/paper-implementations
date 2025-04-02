import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2

def build_loader(batch_size):
    # Aug: RandAug, mixup, CutMix, random erasing, gradient clipping, timm
    transform_train = transforms.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandAugment(),
        v2.ToTensor(),
        # v2.ToImage(),
        # v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        v2.RandomErasing(),
    ])
    transform_test = transforms.Compose([
        v2.ToTensor(),
        # v2.ToImage(),
        # v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print('Training set has {} instances'.format(len(trainset)))
    print('Validation set has {} instances'.format(len(testset)))

    return trainloader, testloader