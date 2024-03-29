import os
import argparse
import torch
from tqdm import tqdm

from dataloader import get_cifar10_data, get_cifar5m_data
from resnet import ResNet18
from train_utils import train_loop, eval_loop

import wandb

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--arch', default='resnet18', type=str, help='Model Architecture')
parser.add_argument('--dataset', default='cifar10', type=str, help='Choose from cifar10 or cifar5m')
parser.add_argument('--lr', default=0.1, type=float, help='Learning Rate')
parser.add_argument('--epochs', default=100, type=int, help='Number of Epochs')
parser.add_argument('--bs', default=128, type=int, help='Batch Size')
parser.add_argument('--device', default='cuda:0', type=str, help='Device for training')
args = parser.parse_args()

wandb.init(
    project="cifar-resnet-runs",
    config={
    "learning_rate": args.lr,
    "architecture": args.arch,
    "dataset": args.dataset,
    "epochs": args.epochs,
    "device": args.device,
    "batch_size": args.bs,
    },
    # mode="disabled"
)

if args.dataset=='cifar10':
    trainloader, testloader = get_cifar10_data(args.bs)
elif args.dataset=='cifar10-subset50':
    trainloader, testloader = get_cifar10_data(args.bs, subset=0.5)
elif args.dataset=='cifar5m-50k':
    trainloader, testloader = get_cifar5m_data(args.bs, subset=50000)
elif args.dataset=='cifar5m-100k':
    trainloader, testloader = get_cifar5m_data(args.bs, subset=100000)
elif args.dataset=='cifar5m-150k':
    trainloader, testloader = get_cifar5m_data(args.bs, subset=150000)
elif args.dataset=='cifar5m-200k':
    trainloader, testloader = get_cifar5m_data(args.bs, subset=200000)
elif args.dataset=='cifar5m-500k':
    trainloader, testloader = get_cifar5m_data(args.bs, subset=500000)
else:
    raise Exception("Dataset not supported yet")

if args.arch=='resnet18':
    model = ResNet18()
else:
    raise Exception("Model arch not supported yet")
model = model.to(args.device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

for epoch in tqdm(range(args.epochs)):
    train_loss = train_loop(model, trainloader, optimizer, criterion, args.device, scheduler)
    test_loss, test_acc = eval_loop(model, testloader, criterion, args.device)

    wandb.log({"train_loss": train_loss, "test_loss": test_loss, "test_acc": test_acc})

savefile = 'models/' + args.dataset + '-' + args.arch + '.pth'
torch.save(model.state_dict(), savefile)

wandb.save(savefile)
wandb.finish()
