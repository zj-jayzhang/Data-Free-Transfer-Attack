import os
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from kornia import augmentation
from sklearn.manifold import TSNE
from torch.nn.functional import mse_loss
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
from advertorch.attacks import LinfPGDAttack, LinfBasicIterativeAttack
import math
import os
import random
from torchvision import datasets, transforms
from PIL import Image
from nets import CNN, CNNCifar10, resnet18, resnet50


def get_model(dataset, load):
    if "mnist" in dataset:
        model = CNN().cuda()
        # model = Net_m().cuda()
    elif dataset == "cifar10" or dataset == "svhn":
        # model = resnet18(num_classes=10).cuda()
        model = CNNCifar10().cuda()
    elif dataset == "cifar100":
        model = resnet50(num_classes=100).cuda()
    elif dataset == "tiny":
        model = resnet50(num_classes=200).cuda()
    # pretraind = 'public/attack/pretrained/'
    pretraind = 'pretrained_ckpt/cifar10_cnn'
    load_list = ['cnn_mnist.pth', 'cnn_fmnist.pth', 'cnncifar10.pkl', 'res18_svhn.pth',
                 'res50_cifar100.pth', 'res50_tiny_imagenet.pth']
    if load == 1:
        if "mnist" == dataset:
            state_dict = torch.load(pretraind + load_list[0])['state_dict']
        elif "fmnist" == dataset:
            state_dict = torch.load(pretraind + load_list[1])['state_dict']
        elif dataset == "cifar10":
            state_dict = torch.load(pretraind + load_list[2])['state_dict']
        elif dataset == "svhn":
            state_dict = torch.load(pretraind + load_list[3])['state_dict']
        elif dataset == "cifar100":
            state_dict = torch.load(pretraind + load_list[4])['state_dict']
        elif dataset == "tiny":
            state_dict = torch.load(pretraind + load_list[5])
    else:
        state_dict = None

    return model, state_dict


def cal_prob(black_net, data):
    with torch.no_grad():
        outputs = black_net(data.detach())
        score = F.softmax(outputs, dim=1)  # score-based
    score = score.detach().cpu().numpy()
    score = torch.from_numpy(score).cuda().float()
    return score


def cal_label(black_net, data):
    with torch.no_grad():
        outputs = black_net(data.detach())
        _, label = torch.max(outputs.data, 1)
    label = label.detach().cpu().numpy()
    label = torch.from_numpy(label).cuda().long()
    return label


def test_robust(loader, substitute_net, original_net, dataset):
    # cfgs = dict(random=True, test_num_steps=40, test_step_size=0.01, test_epsilon=0.3, num_classes=10)
    if dataset == "mnist":
        cfgs = dict(test_step_size=0.01, test_epsilon=0.3)
    elif dataset == "cifar10" or dataset == "cifar100":
        cfgs = dict(test_step_size=2.0 / 255, test_epsilon=8.0 / 255)
    elif dataset == "fmnist":
        cfgs = dict(test_step_size=0.01, test_epsilon=0.3)
    elif dataset == "svhn" or dataset == "tiny":
        cfgs = dict(test_step_size=0.01, test_epsilon=0.3)

    correct_ghost = 0.0
    correct = 0.0
    total = 0.0
    substitute_net.eval()
    adversary = LinfBasicIterativeAttack(
        substitute_net, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), eps=cfgs['test_epsilon'],
        nb_iter=120, eps_iter=cfgs['test_step_size'], clip_min=0.0, clip_max=1.0,
        targeted=False)

    for inputs, labels in loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        total += labels.size(0)
        t_label = cal_label(original_net, inputs)
        idx = torch.where(t_label == labels)[0]
        correct += idx.shape[0]
        adv_inputs_ghost = adversary.perturb(inputs[idx], labels[idx])
        predicted = cal_label(original_net, adv_inputs_ghost)
        correct_ghost += (predicted != labels[idx]).sum()
    # print('Attack success rate: {}, clean acc: {}'.format(100. * correct_ghost / correct, 100 * correct / total))
    return 100. * correct_ghost / correct, 100 * correct / total


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            total += data.shape[0]
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total
    acc = 100. * correct / total
    return acc, test_loss


def print_log(strs, log):
    print(strs)
    log.write(strs)


def get_dataset(dataset):
    data_dir = '/mnt/lustre/share_data/zhangjie/'
    if dataset == "mnist":
        train_dataset = datasets.MNIST(data_dir, train=True,
                                       transform=transforms.Compose(
                                           [transforms.ToTensor()]))
        test_dataset = datasets.MNIST(data_dir, train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                      ]))
    elif dataset == "fmnist":
        train_dataset = datasets.FashionMNIST(data_dir, train=True,
                                              transform=transforms.Compose(
                                                  [transforms.ToTensor()]))
        test_dataset = datasets.FashionMNIST(data_dir, train=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                             ]))
    elif dataset == "svhn":
        train_dataset = datasets.SVHN(data_dir, split="train",
                                      transform=transforms.Compose(
                                          [transforms.ToTensor()]))
        test_dataset = datasets.SVHN(data_dir, split="test",
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ]))
    elif dataset == "cifar10":
        train_dataset = datasets.CIFAR10(data_dir, train=True,
                                         transform=transforms.Compose(
                                             [
                                                 transforms.RandomCrop(32, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                             ]))
        test_dataset = datasets.CIFAR10(data_dir, train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                        ]))
    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(data_dir, train=True,
                                          transform=transforms.Compose(
                                              [
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                              ]))
        test_dataset = datasets.CIFAR100(data_dir, train=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                         ]))
    elif dataset == "tiny":
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
            ])
        }
        data_dir = "data/tiny-imagenet-200/"
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val', 'test']}
        train_dataset = image_datasets['train']
        test_dataset = image_datasets['val']
        # train_loader = data.DataLoader(image_datasets['train'], batch_size=128, shuffle=True, num_workers=4)
        # val_loader = data.DataLoader(image_datasets['val'], batch_size=128, shuffle=False, num_workers=4)

    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256,
                                               shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                              shuffle=False, num_workers=4)

    return train_loader, test_loader


class ScoreLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(ScoreLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target):
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)  # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))  # [NHW, C]
        target = target.view(-1, 1)  # [NHW，1]

        score = F.log_softmax(logits, 1)  # score-based
        score = score.gather(1, target)  # [NHW, 1]
        loss = -1 * score

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class MultiTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str(self.transform)


def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple)):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0, 3, 1, 2)  # make it channel first
    assert len(images.shape) == 4
    assert isinstance(images, np.ndarray)

    N, C, H, W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))

    pack = np.zeros((C, H * row + padding * (row - 1), W * col + padding * (col - 1)), dtype=images.dtype)
    for idx, img in enumerate(images):
        h = (idx // col) * (H + padding)
        w = (idx % col) * (W + padding)
        pack[:, h:h + H, w:w + W] = img
    return pack


def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir != '':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images(imgs, col=col).transpose(1, 2, 0).squeeze()
        imgs = Image.fromarray(imgs)
        if size is not None:
            if isinstance(size, (list, tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max(h, w)
                scale = float(size) / float(max_side)
                _w, _h = int(w * scale), int(h * scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            if img.shape[0] == 1:
                img = Image.fromarray(img[0])
            else:
                img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(output_filename + '-%d.png' % (idx))


def _collect_all_images(root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    images = []
    if isinstance(postfix, str):
        postfix = [postfix]
    for dirpath, dirnames, files in os.walk(root):  # '/dockerdata/cvpr/10-28/ft_local/run/svhn_4',[],files(all imgs)
        files.sort()
        files = files[-256 * 400:]
        # files = files[-2048 * 400:]
        for pos in postfix:
            for f in files:
                if f.endswith(pos):
                    images.append(os.path.join(dirpath, f))
    return images


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(self.root)  # [ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s' % (
            self.root, len(self), self.transform)


class ImagePool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs, targets=None):
        save_image_batch(imgs, os.path.join(self.root, "%d.png" % (self._idx)), pack=False)
        self._idx += 1

    def get_dataset(self, transform=None, labeled=True):
        return UnlabeledImageDataset(self.root, transform=transform)

