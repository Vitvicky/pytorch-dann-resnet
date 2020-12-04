"""Train dann."""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from core.test import test
from utils.utils import save_model
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def train_src(model, params, src_data_loader, tgt_data_loader, tgt_data_loader_eval, device, logger=None):
    """Train dann."""
    ####################
    # 1. setup network #
    ####################

    # setup criterion and optimizer

    if not params.finetune_flag:
        print("training non-office task")
        optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
    else:
        print("training office task")
        parameter_list = [{
            "params": model.features.parameters(),
            "lr": 0.001
        }, {
            "params": model.fc.parameters(),
            "lr": 0.001
        }, {
            "params": model.bottleneck.parameters()
        }, {
            "params": model.classifier.parameters()
        }, {
            "params": model.discriminator.parameters()
        }]
        optimizer = optim.SGD(parameter_list, lr=0.01, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################
    global_step = 0
    for epoch in range(params.num_epochs):
        # set train state for Dropout and BN layers
        model.train()
        # zip source and target data pair
        len_dataloader = min(len(src_data_loader), len(tgt_data_loader))
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, class_src), (images_tgt, _)) in data_zip:

            p = float(step + epoch * len_dataloader) / \
                params.num_epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            if params.lr_adjust_flag == 'simple':
                lr = adjust_learning_rate(optimizer, p)
            else:
                lr = adjust_learning_rate_office(optimizer, p)
            if not logger == None:
                logger.add_scalar('lr', lr, global_step)

            # prepare domain label
            size_src = len(images_src)
            size_tgt = len(images_tgt)

            # make images variable
            class_src = class_src.to(device)
            images_src = images_src.to(device)

            # zero gradients for optimizer
            model.zero_grad()

            # train on source domain
            src_class_output, src_domain_output = model(input_data=images_src, alpha=alpha)
            src_loss_class = criterion(src_class_output, class_src)

            loss = src_loss_class

            # optimize dann
            loss.backward()
            optimizer.step()

            global_step += 1

            # print step info
            if not logger == None:
                logger.add_scalar('loss', loss.item(), global_step)

            if ((step + 1) % params.log_step == 0):
                print(
                    "Epoch [{:4d}/{}] Step [{:2d}/{}]: loss={:.6f}".format(epoch + 1, params.num_epochs, step + 1, len_dataloader, loss.data.item()))

        # eval model
        if ((epoch + 1) % params.eval_step == 0):
            src_test_loss, src_acc, src_acc_domain = test(model, src_data_loader, device, flag='source')
            tgt_test_loss, tgt_acc, tgt_acc_domain = test(model, tgt_data_loader_eval, device, flag='target')
            if not logger == None:
                logger.add_scalar('src_test_loss', src_test_loss, global_step)
                logger.add_scalar('src_acc', src_acc, global_step)


        # save model parameters
        if ((epoch + 1) % params.save_step == 0):
            save_model(model, params.model_root,
                       params.src_dataset + '-' + params.tgt_dataset + "-dann-{}.pt".format(epoch + 1))

    # save final model
    save_model(model, params.model_root, params.src_dataset + '-' + params.tgt_dataset + "-dann-final.pt")

    return model

def train_dann(model, params, src_data_loader, tgt_data_loader, tgt_data_loader_eval, device, loggi, logger=None):
    """Train dann."""
    ####################
    # 1. setup network #
    ####################

    # setup criterion and optimizer

    if not params.finetune_flag:
        print("training non-office task")
        optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
    else:
        print("training office task")
        parameter_list = [{
            "params": model.features.parameters(),
            "lr": 0.001
        }, {
            "params": model.fc.parameters(),
            "lr": 0.001
        }, {
            "params": model.bottleneck.parameters()
        }, {
            "params": model.classifier.parameters()
        }, {
            "params": model.discriminator.parameters()
        }]
        optimizer = optim.SGD(parameter_list, lr=0.001, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################
    try:
        global_step = 0
        bestAcc = 0.0
        for epoch in range(params.num_epochs):
            # set train state for Dropout and BN layers
            model.train()
            # zip source and target data pair
            len_dataloader = min(len(src_data_loader), len(tgt_data_loader))
            data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
            for step, ((images_src, class_src), (images_tgt, _)) in data_zip:

                p = float(step + epoch * len_dataloader) / \
                    params.num_epochs / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                if params.lr_adjust_flag == 'simple':
                    lr = adjust_learning_rate(optimizer, p)
                else:
                    lr = adjust_learning_rate_office(optimizer, p)
                if not logger == None:
                    logger.add_scalar('lr', lr, global_step)

                # prepare domain label
                size_src = len(images_src)
                size_tgt = len(images_tgt)
                label_src = torch.zeros(size_src).long().to(device)  # source 0
                label_tgt = torch.ones(size_tgt).long().to(device)  # target 1

                # make images variable
                class_src = class_src.to(device)
                images_src = images_src.to(device)
                images_tgt = images_tgt.to(device)

                # zero gradients for optimizer
                optimizer.zero_grad()

                # train on source domain
                src_class_output, src_domain_output = model(input_data=images_src, alpha=alpha)
                src_loss_class = criterion(src_class_output, class_src)
                src_loss_domain = criterion(src_domain_output, label_src)

                # train on target domain
                _, tgt_domain_output = model(input_data=images_tgt, alpha=alpha)
                tgt_loss_domain = criterion(tgt_domain_output, label_tgt)

                loss = src_loss_class + src_loss_domain + tgt_loss_domain
                if params.src_only_flag:
                    loss = src_loss_class

                # optimize dann
                loss.backward()
                optimizer.step()

                global_step += 1

                # print step info
                if not logger == None:
                    logger.add_scalar('src_loss_class', src_loss_class.item(), global_step)
                    logger.add_scalar('src_loss_domain', src_loss_domain.item(), global_step)
                    logger.add_scalar('tgt_loss_domain', tgt_loss_domain.item(), global_step)
                    logger.add_scalar('loss', loss.item(), global_step)

                if ((step + 1) % params.log_step == 0):
                    print(
                        "Epoch [{:4d}/{}] Step [{:2d}/{}]: src_loss_class={:.6f}, src_loss_domain={:.6f}, tgt_loss_domain={:.6f}, loss={:.6f}"
                        .format(epoch + 1, params.num_epochs, step + 1, len_dataloader, src_loss_class.data.item(),
                                src_loss_domain.data.item(), tgt_loss_domain.data.item(), loss.data.item()))

            # eval model
            if ((epoch + 1) % params.eval_step == 0):
                tgt_test_loss, tgt_acc, tgt_acc_domain = test(model, tgt_data_loader_eval, device, loggi, flag='target')
                src_test_loss, src_acc, src_acc_domain = test(model, src_data_loader, device, loggi, flag='source')
                loggi.info('\n')
                if tgt_acc > bestAcc:
                    bestAcc = tgt_acc
                    bestAccS = src_acc
                    save_model(model, params.model_root,
                    params.src_dataset + '-' + params.tgt_dataset + "-dann-best.pt")
                if not logger == None:
                    logger.add_scalar('src_test_loss', src_test_loss, global_step)
                    logger.add_scalar('src_acc', src_acc, global_step)
                    logger.add_scalar('src_acc_domain', src_acc_domain, global_step)
                    logger.add_scalar('tgt_test_loss', tgt_test_loss, global_step)
                    logger.add_scalar('tgt_acc', tgt_acc, global_step)
                    logger.add_scalar('tgt_acc_domain', tgt_acc_domain, global_step)
    except KeyboardInterrupt as ke:
        loggi.info('Saving the final weights before quitting')
        # save final model
        save_model(model, params.model_root, params.src_dataset + '-' + params.tgt_dataset + "-dann-final.pt")
        loggi.info('\n============ Summary ============= \n')
        loggi.info('Accuracy of the %s dataset: %f' % (params.src_dataset, bestAccS))
        loggi.info('Accuracy of the %s dataset: %f' % (params.tgt_dataset, bestAcc))

    return model

def adjust_learning_rate(optimizer, p):
    lr_0 = 0.01
    alpha = 10
    beta = 0.75
    lr = lr_0 / (1 + alpha * p)**beta
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_office(optimizer, p):
    lr_0 = 0.001
    alpha = 10
    beta = 0.75
    lr = lr_0 / (1 + alpha * p)**beta
    for param_group in optimizer.param_groups[:2]:
        param_group['lr'] = lr
    for param_group in optimizer.param_groups[2:]:
        param_group['lr'] = 10 * lr
    return lr
