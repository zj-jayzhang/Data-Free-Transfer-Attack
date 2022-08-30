#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import os
from tensorboardX import SummaryWriter
import numpy as np
import torch.optim as optim
import warnings
from tqdm import tqdm
from torch.nn.functional import mse_loss
import random
from torchvision import transforms
from kornia import augmentation
import torch
import torch.nn.functional as F
import torch.utils.data.sampler as sp
import torch.backends.cudnn as cudnn

from nets import Generator_2
from utils import ScoreLoss, ImagePool, MultiTransform, reset_model, get_dataset, cal_prob, cal_label, setup_seed, \
    get_model, print_log, test, test_robust, save_checkpoint

warnings.filterwarnings('ignore')


class Synthesizer():
    def __init__(self, generator, nz, num_classes, img_size,
                 iterations, lr_g,
                 sample_batch_size, save_dir, dataset):
        super(Synthesizer, self).__init__()
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.nz = nz
        self.score_loss = ScoreLoss()
        self.num_classes = num_classes
        self.sample_batch_size = sample_batch_size
        self.save_dir = save_dir
        self.data_pool = ImagePool(root=self.save_dir)
        self.data_iter = None
        self.dataset = dataset

        self.generator = generator.cuda().train()

        self.aug = MultiTransform([
            # global view
            transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
            ]),
            # local view
            transforms.Compose([
                augmentation.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]], scale=[0.25, 1.0]),
                augmentation.RandomHorizontalFlip(),
            ]),
        ])
        # =======================
        if not ("cifar" in dataset):
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])

    def get_data(self):
        datasets = self.data_pool.get_dataset(transform=self.transform)  # 获取程序运行到现在所有的图片
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=256, shuffle=True,
            num_workers=4, pin_memory=True, )
        return self.data_loader

    def gen_data(self, student):
        student.eval()
        best_cost = 1e6
        best_inputs = None
        z = torch.randn(size=(self.sample_batch_size, self.nz)).cuda()  #
        z.requires_grad = True
        targets = torch.randint(low=0, high=self.num_classes, size=(self.sample_batch_size,))
        targets = targets.sort()[0]
        targets = targets.cuda()
        reset_model(self.generator)
        optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g, betas=[0.5, 0.999])

        for it in range(self.iterations):
            optimizer.zero_grad()
            inputs = self.generator(z)  # bs,nz
            global_view, _ = self.aug(inputs)  # crop and normalize

            s_out = student(global_view)
            loss = self.score_loss(s_out, targets)  # ce_loss
            if best_cost > loss.item() or best_inputs is None:
                best_cost = loss.item()
                best_inputs = inputs.data

            loss.backward()
            optimizer.step()
        # with tqdm(total=self.iterations) as t:

                # optimizer_mlp.step()
                # t.set_description('iters:{}, loss:{}'.format(it, loss.item()))

        # save best inputs and reset data iter
        self.data_pool.add(best_inputs)  # 生成了一个batch的数据


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--score', type=float, default=0,
                        help="number of rounds of training")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    # Data Free

    parser.add_argument('--save_dir', default='run/mnist', type=str)

    # Basic
    parser.add_argument('--lr_g', default=1e-3, type=float,
                        help='initial learning rate for generation')
    parser.add_argument('--g_steps', default=30, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--nz', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--synthesis_batch_size', default=256, type=int)
    # Misc
    parser.add_argument('--seed', default=2021, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--type', default="score", type=str,
                        help='score or label')
    parser.add_argument('--model', default="", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--other', default="", type=str,
                        help='seed for initializing training.')
    args = parser.parse_args()
    return args


def kd_train(synthesizer, model, optimizer, score_val):
    sub_net, blackBox_net = model
    sub_net.train()
    blackBox_net.eval()

    # with tqdm(synthesizer.get_data()) as epochs:
    data = synthesizer.get_data()
    for idx, (images) in enumerate(data):
        optimizer.zero_grad()
        images = images.cuda()
        original_score = cal_prob(blackBox_net, images)  # prob
        substitute_outputs = sub_net(images.detach())
        substitute_score = F.softmax(substitute_outputs, dim=1)
        loss_mse = mse_loss(
            substitute_score, original_score, reduction='mean')
        label = cal_label(blackBox_net, images)  # label
        loss_ce = F.cross_entropy(substitute_outputs, label)
        # ==============================
        # idx = torch.where(substitute_outputs.max(1)[1] != label)[0]
        # loss_adv = F.cross_entropy(substitute_outputs[idx], label[idx])
        # ==============================
        loss = loss_ce + loss_mse * score_val

        loss.backward()
        optimizer.step()
    # return loss.item()



if __name__ == '__main__':
    dir = './saved/ours'
    if not os.path.exists(dir):
        os.mkdir(dir)

    args = args_parser()
    setup_seed(args.seed)
    train_loader, test_loader = get_dataset(args.dataset)

    public = dir + '/logs_{}_{}'.format(args.dataset, str(args.score))
    if not os.path.exists(public):
        os.mkdir(public)
    log = open('{}/log_ours.txt'.format(public), 'w')

    list = [i for i in range(0, len(test_loader.dataset))]
    data_list = random.sample(list, 1024)
    val_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=128,
                                             sampler=sp.SubsetRandomSampler(data_list), num_workers=4)

    tf_writer = SummaryWriter(log_dir=public)
    sub_net, _ = get_model(args.dataset, 0)
    blackBox_net, state_dict = get_model(args.dataset, 1)
    blackBox_net.load_state_dict(state_dict)

    print_log("===================================== \n", log)
    acc, _ = test(blackBox_net, val_loader)
    print_log("Accuracy of the black-box model:{:.3} % \n".format(acc), log)
    acc, _ = test(sub_net, val_loader)
    print_log("Accuracy of the substitute model:{:.3} % \n".format(acc), log)
    asr, val_acc = 0.0, 0.0 # test_robust(val_loader, sub_net, blackBox_net, args.dataset)
    print_log("ASR:{:.3} %, val acc:{:.3} % \n".format(asr, val_acc), log)
    print_log("===================================== \n", log)
    log.flush()

    ################################################
    # data generator
    ################################################
    nz = args.nz
    nc = 3 if "cifar" in args.dataset or args.dataset == "svhn" or args.dataset == "tiny" else 1
    # img_size = 32 if "cifar" in args.dataset or args.dataset == "svhn" else 28

    if "cifar" in args.dataset or args.dataset == "svhn":
        img_size = 32
    elif "mnist" in args.dataset:
        img_size = 28
    elif args.dataset == "tiny":
        img_size = 64

    if "cifar" in args.dataset or args.dataset == "svhn":
        img_size2 = (3, 32, 32)
    elif "mnist" in args.dataset:
        img_size2 = (1, 28, 28)
    elif args.dataset == "tiny":
        img_size2 = (3, 64, 64)
    generator = Generator_2(nz=nz, ngf=64, img_size=img_size, nc=nc).cuda()
    # ====================
    sub_net = torch.nn.DataParallel(sub_net)
    blackBox_net = torch.nn.DataParallel(blackBox_net)
    generator = torch.nn.DataParallel(generator)
    # ====================

    args.cur_ep = 0
    # img_size2 = (
    #     3, 32, 32) if "cifar" in args.dataset or args.dataset == "svhn" else (1, 28, 28)

    if args.dataset == "cifar100":
        num_class = 100
    elif args.dataset == "tiny":
        num_class = 200
    else:
        num_class = 10

    synthesizer = Synthesizer(generator,
                              nz=nz,
                              num_classes=num_class,
                              img_size=img_size2,
                              iterations=args.g_steps,
                              lr_g=args.lr_g,
                              sample_batch_size=args.batch_size,
                              save_dir=args.save_dir,
                              dataset=args.dataset)
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=args.momentum)
    sub_net.train()
    best_acc = -1
    best_asr = -1

    best_acc_ckpt = '{}/{}_ours_acc.pth'.format(public, args.dataset)
    best_asr_ckpt = '{}/{}_ours_asr.pth'.format(public, args.dataset)
    for epoch in tqdm(range(args.epochs)):
        # 1. Data synthesis
        synthesizer.gen_data(sub_net)  # g_steps
        kd_train(synthesizer, [sub_net, blackBox_net], optimizer, args.score)
        if epoch % 1 == 0:  # 250*40, 250*10=2.5k
            acc, test_loss = test(sub_net, val_loader)
            asr, val_acc = test_robust(val_loader, sub_net, blackBox_net, args.dataset)

        #     save_checkpoint({
        #     'state_dict': sub_net.state_dict(),
        #     'epoch': epoch,
        # }, acc > best_acc, best_acc_ckpt)
        #
        #     save_checkpoint({
        #     'state_dict': sub_net.state_dict(),
        #     'epoch': epoch,
        # }, asr > best_asr, best_asr_ckpt)

            best_asr = max(best_asr, asr)
            best_acc = max(best_acc, acc)

            print_log("Accuracy of the substitute model:{:.3} %, best accuracy:{:.3} % \n".format(acc, best_acc), log)
            print_log("ASR:{:.3} %, best asr:{:.3} %, val acc:{:.3} % \n".format(asr, best_asr, val_acc), log)
            log.flush()

"""
40*256=1w
CUDA_VISIBLE_DEVICES=2 python3 main.py --epochs=400  --save_dir=run/svhn_1 \
--dataset=svhn --score=1 --other=cnn_svhn --g_steps=5

CUDA_VISIBLE_DEVICES=3 python3 main.py --epochs=400  --save_dir=run/svhn_2 \
--dataset=svhn --score=1 --other=cnn_svhn --g_steps=30


CUDA_VISIBLE_DEVICES=2 python3 main.py --epochs=400  --save_dir=run/cifar10 --dataset=cifar10 --score=1 --other=cnn_cifar10 --g_steps=5

CUDA_VISIBLE_DEVICES=2 python3 main.py --epochs=400  --save_dir=run/mnist_1 --dataset=mnist --score=1 --other=cnn_mnsit --g_steps=10

CUDA_VISIBLE_DEVICES=1 python3 main.py --epochs=400  --save_dir=run/fmnist_1 --dataset=fmnist --score=1 --other=cnn_fmnsit --g_steps=10

CUDA_VISIBLE_DEVICES=1 python3 main.py --epochs=400  --save_dir=run/fmnist_2 --dataset=fmnist --score=1 --other=cnn_fmnsit --g_steps=30
"""
