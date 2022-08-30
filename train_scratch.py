from __future__ import print_function
import argparse  # Python 命令行解析工具
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from nets import resnet34, CNN, CNNCifar10, resnet18, resnet50, MLP, AlexNet, vgg8_bn
from utils import test, get_dataset
import warnings

warnings.filterwarnings('ignore')


def train(model, train_loader, optimizer, epoch):
    model.train()

    for idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def get_model(dataset, net):
#     if "mnist" in dataset:
#         if net == "mlp":
#             model = MLP().cuda()
#         elif net == "lenet":
#             model = CNN().cuda()
#         elif net == "alexnet":
#             model = AlexNet().cuda()
#     elif dataset == "svhn":
#         if net == "alexnet":
#             model = CNNCifar10().cuda()
#         elif net == "vgg":
#             model = CNNCifar10().cuda()
#         elif net == "resnet18":
#             model = resnet18(num_classes=10).cuda()
#     elif dataset == "cifar10":
#         # model = resnet18(num_classes=10).cuda()
#         model = CNNCifar10().cuda()
#     elif dataset == "cifar100":
#         model = resnet50(num_classes=100).cuda()
#     elif dataset == "imagenet":
#         model = resnet18(num_classes=12).cuda()
#     return model


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default="cifar10",
                        help='dataset')
    parser.add_argument('--net', type=str, default="cifar10",
                        help='dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--model', type=str, default='resnet34',
                        help='SGD momentum (default: 0.9)')
    args = parser.parse_args()

    train_loader, test_loader = get_dataset(args.dataset)
    # model = get_teacher_model(args.dataset, args.net)
    model = CNNCifar10().cuda()
    # model = vgg8_bn(num_classes=10).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    bst_acc = -1
    public = "pretrained_large/{}_{}".format(args.dataset, args.net)
    tf_writer = SummaryWriter(log_dir=public)
    for epoch in range(1, args.epochs + 1):
        # adjust_learning_rate(args.lr, optimizer, epoch)
        train(model, train_loader, optimizer, epoch)
        acc, loss = test(model, test_loader)
        if acc > bst_acc:
            bst_acc = acc
            torch.save(model.state_dict(), '{}/{}_{}.pkl'.format(public, args.dataset, args.net))

        tf_writer.add_scalar('test_acc', acc, epoch)
        bst_acc = max(bst_acc, acc)
        print("Epoch:{},\t test_acc:{}, best_acc:{}".format(epoch, acc, bst_acc))


if __name__ == '__main__':
    main()



