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
import joblib
from advertorch.attacks import LinfBasicIterativeAttack


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
    # blackBox_net.eval()

    # with tqdm(synthesizer.get_data()) as epochs:
    data = synthesizer.get_data()
    for idx, (images) in enumerate(data):
        optimizer.zero_grad()
        images = images.cuda()
        with torch.no_grad():
            original_score = cal_azure_proba(blackBox_net, images)
            label = torch.max(original_score.data, 1)[1]

        # original_score = cal_prob(blackBox_net, images)  # prob
        substitute_outputs = sub_net(images.detach())
        substitute_score = F.softmax(substitute_outputs, dim=1)
        loss_mse = mse_loss(
            substitute_score, original_score, reduction='mean')
        # label = cal(blackBox_net, images)  # label
        loss_ce = F.cross_entropy(substitute_outputs, label)
        # ==============================
        # idx = torch.where(substitute_outputs.max(1)[1] != label)[0]
        # loss_adv = F.cross_entropy(substitute_outputs[idx], label[idx])
        # ==============================
        loss = loss_ce + loss_mse * score_val

        loss.backward()
        optimizer.step()


def cal_azure(model, data):
    data = data.view(data.size(0), 784).cpu().numpy()
    output = model.predict(data)
    output = torch.from_numpy(output).cuda().long()
    return output


def cal_azure_proba(model, data):
    data = data.view(data.size(0), 784).cpu().numpy()
    output = model.predict_proba(data)
    output = torch.from_numpy(output).cuda().float()
    return output

if __name__ == '__main__':
    dir = './saved/api'
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
    clf = joblib.load('pretrained/sklearn_mnist_model.pkl')

    with torch.no_grad():
        correct_netD = 0.0
        total = 0.0
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            predicted = cal_azure(clf, inputs)
            total += labels.size(0)
            correct_netD += (predicted == labels).sum()
        print('Accuracy of the black-box network : %.2f %%' %
              (100. * correct_netD.float() / total))
    ################################################
    # estimate the attack success rate of initial D:
    ################################################
    correct_ghost = 0.0
    total = 0.0
    sub_net.eval()
    adversary_ghost = LinfBasicIterativeAttack(
        sub_net, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        nb_iter=100, eps_iter=0.01, clip_min=0.0, clip_max=1.0,
        targeted=False)
    for inputs, labels in val_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        adv_inputs_ghost = adversary_ghost.perturb(inputs, labels)
        with torch.no_grad():
            predicted = cal_azure(clf, adv_inputs_ghost)
        total += labels.size(0)
        correct_ghost += (predicted == labels).sum()
    print('Attack success rate: %.2f %%' %
          (100 - 100. * correct_ghost.float() / total))

    ################################################
    # data generator
    ################################################
    nz = args.nz
    nc = 1

    img_size = 28
    img_size2 = (1, 28, 28)

    generator = Generator_2(nz=nz, ngf=64, img_size=img_size, nc=nc).cuda()

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
    for epoch in tqdm(range(args.epochs)):
        # 1. Data synthesis
        synthesizer.gen_data(sub_net)  # g_steps
        kd_train(synthesizer, [sub_net, clf], optimizer, args.score)
        if epoch % 1 == 0:  # 250*40, 250*10=2.5k
            acc, test_loss = test(sub_net, val_loader)
            ################################################
            # estimate the attack success rate of initial D:
            ################################################
            correct_ghost = 0.0
            total = 0.0
            sub_net.eval()
            adversary_ghost = LinfBasicIterativeAttack(
                sub_net, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                nb_iter=100, eps_iter=0.01, clip_min=0.0, clip_max=1.0,
                targeted=False)
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                adv_inputs_ghost = adversary_ghost.perturb(inputs, labels)
                with torch.no_grad():
                    predicted = cal_azure(clf, adv_inputs_ghost)
                total += labels.size(0)
                correct_ghost += (predicted == labels).sum()
            asr = (100 - 100. * correct_ghost.float() / total)
            print('Attack success rate: %.2f %%' % asr)
            save_checkpoint({
                'state_dict': sub_net.state_dict(),
                'epoch': epoch,
            }, acc > best_acc, best_acc_ckpt)

            best_asr = max(best_asr, asr)
            best_acc = max(best_acc, acc)

            print_log("Accuracy of the substitute model:{:.3} %, best accuracy:{:.3} % \n".format(acc, best_acc), log)
            print_log("ASR:{:.3} %, best asr:{:.3} %\n".format(asr, best_asr), log)
            log.flush()

"""

CUDA_VISIBLE_DEVICES=2 python3 attack_api.py --epochs=100  --save_dir=run/mnist_1 --dataset=mnist --score=1 --other=cnn_mnsit --g_steps=10

"""
