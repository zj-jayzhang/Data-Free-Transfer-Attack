from __future__ import print_function

import argparse
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.sampler as sp
from advertorch.attacks import GradientSignAttack, PGDAttack, LinfPGDAttack
from advertorch.attacks import LinfBasicIterativeAttack, CarliniWagnerL2Attack


from utils import get_model, test, setup_seed, get_dataset


def test_adver(net, tar_net, attack, target, testloader, dataset):
    if dataset == "mnist":
        cfgs = dict(test_step_size=0.01, test_epsilon=0.3)
    elif dataset == "cifar10" or dataset == "cifar100":
        cfgs = dict(test_step_size=2.0 / 255, test_epsilon=8.0 / 255)
    elif dataset == "fmnist":
        cfgs = dict(test_step_size=0.01, test_epsilon=0.3)
    elif dataset == "svhn" or dataset == "tiny":
        cfgs = dict(test_step_size=2.0 / 255, test_epsilon=8.0 / 255)


    net.eval()
    tar_net.eval()
    # BIM
    if attack == 'BIM':
        adversary = LinfBasicIterativeAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=cfgs['test_epsilon'],
            nb_iter=120, eps_iter=cfgs['test_step_size'], clip_min=0.0, clip_max=1.0,
            targeted=target)
    # PGD
    elif attack == 'PGD':
        if target:
            adversary = PGDAttack(
                net,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=cfgs['test_epsilon'],
                nb_iter=20, eps_iter=cfgs['test_step_size'], clip_min=0.0, clip_max=1.0,
                targeted=target)
        else:
            adversary = PGDAttack(
                net,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=cfgs['test_epsilon'],
                nb_iter=20, eps_iter=cfgs['test_step_size'], clip_min=0.0, clip_max=1.0,
                targeted=target)
    # FGSM
    elif attack == 'FGSM':
        adversary = GradientSignAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=cfgs['test_epsilon'],
            targeted=target)
    elif attack == 'CW':
        adversary = CarliniWagnerL2Attack(
            net,
            num_classes=10,
            learning_rate=0.45,
            binary_search_steps=10,
            max_iterations=20,
            targeted=target)

    # ----------------------------------
    # Obtain the attack success rate of the model
    # ----------------------------------

    correct = 0.0
    total = 0.0
    tar_net.eval()
    total_L2_distance = 0.0
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = tar_net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        if target:
            # randomly choose the specific label of targeted attack
            labels = torch.randint(0, 9, (inputs.shape[0],)).cuda()
            # test the images which are not classified as the specific label
            idx = torch.where(predicted != labels)[0]
            adv_inputs_ori = adversary.perturb(inputs[idx], labels[idx])
            L2_distance = (torch.norm(adv_inputs_ori - inputs[idx])).item()
            total_L2_distance += L2_distance
            with torch.no_grad():
                outputs = tar_net(adv_inputs_ori)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels[idx]).sum()
        else:
            # test the images which are classified correctly
            idx = torch.where(predicted == labels)[0]
            adv_inputs_ori = adversary.perturb(inputs[idx], labels[idx])
            L2_distance = (torch.norm(adv_inputs_ori - inputs[idx])).item()
            total_L2_distance += L2_distance
            with torch.no_grad():
                outputs = tar_net(adv_inputs_ori)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels[idx]).sum()

    asr = 100. * correct.float() / total if target else 100.0 - 100. * correct.float() / total
    return asr


if __name__ == '__main__':
    setup_seed(2021)
    parser = argparse.ArgumentParser()
    # parser.add_argument('--target', type=int, )
    parser.add_argument('--dataset', type=str, )
    parser.add_argument('--dir', type=str, )

    opt = parser.parse_args()
    cudnn.benchmark = True

    train_loader, test_loader = get_dataset(opt.dataset)
    list = [i for i in range(0, len(test_loader.dataset))]
    data_list = random.sample(list, 1024)
    val_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=64,
                                             sampler=sp.SubsetRandomSampler(data_list), num_workers=4)

    sub_net, _ = get_model(opt.dataset, 0)
    state_dict_1 = torch.load(opt.dir)['state_dict']
    sub_net.load_state_dict(state_dict_1)

    blackBox_net, state_dict = get_model(opt.dataset, 1)
    blackBox_net.load_state_dict(state_dict)

    acc, _ = test(sub_net, val_loader)
    print("Accuracy of the sub_net:{:.3} % \n".format(acc))

    # asr = test_adver(sub_net, blackBox_net, 'CW', 'Untarget' == 'Target', val_loader, opt.dataset)
    # print('Untarget' + " , " + "type: " + 'CW' + ", ASR:{:.2f} %, ".format(asr))
    for attack in ['Target', 'Untarget']:
        for adv in ['FGSM', 'BIM', 'PGD']:
            asr = test_adver(sub_net, blackBox_net, adv, attack == 'Target', val_loader, opt.dataset)
            print(attack + " , " + "type: " + adv + ", ASR:{:.2f} %, ".format(asr))

# python3 eval_rob.py --dataset=mnist --dir=