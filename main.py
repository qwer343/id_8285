# Task Transfer Learning
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import os

from opts import parse_opts
from model import StudentModel, SmartModel, LabelPerturbation
from function import train, validation
from utils import save_path, Logger
from dataload import dataLoadFunc


import warnings

warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    opt = parse_opts()
    
    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

    
    train_loader, valid_loader = dataLoadFunc(opt)

    StudentModel = StudentModel(opt)
    StudentModel = StudentModel.to(device)
    smartModel = None
    perturbation_model = None
    if opt.smart_model:
        opt.is_smart_model = True
        smartModel = SmartModel(opt)
        smartModel = smartModel.to(device)

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        StudentModel = nn.DataParallel(StudentModel)
        parms = list(StudentModel.module.parameters())
    
        if opt.isSource:
            smartModel = nn.DataParallel(smartModel)
     
    elif torch.cuda.device_count() == 1:
        print("Using", torch.cuda.device_count(), "GPU!")
        parms = list(StudentModel.parameters())

    optimizer = torch.optim.SGD(parms, opt.lr,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)


    if opt.optim == "sgd":
        optimizer = torch.optim.SGD(parms, opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
    elif opt.optim == "adam":
        optimizer = torch.optim.Adam(parms, lr=opt.lr)


    if opt.dataset in ['cifar10', 'cifar100', 'stl10']:
        milestones = [60,120,160,200]
        gammaValue = 0.1
        if opt.dataset == 'stl10':
            gammaValue = 0.2
    elif opt.dataset == 'imagenet':
        milestones = [30, 60, 90]
        gammaValue = 0.1
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gammaValue)

    # record training process
    savePath, date_method, save_model_path  = save_path(opt)
    train_logger = Logger(os.path.join(savePath, date_method, 'train.log'),
                                    ['epoch', 'loss', 'acc', 'lr'])
    val_logger = Logger(os.path.join(savePath, date_method, 'val.log'),
                                    ['epoch', 'loss', 'acc', 'best_acc', 'lr'])

    writer = SummaryWriter(os.path.join(savePath, date_method,'logfile'))
    # start training
    best_acc = 0
    for epoch in range(1, opt.epochs + 1):
        # train, test model
        train_losses, train_scores, = train([StudentModel, smartModel, perturbation_model], device, train_loader, optimizer, epoch, opt)
        
        test_losses, test_scores = validation([StudentModel, smartModel, perturbation_model], device, optimizer, valid_loader, opt)
        scheduler.step()
                
        # plot average of each epoch loss value
        train_logger.log({
                        'epoch': epoch,
                        'loss': train_losses.avg,
                        'acc': train_scores.avg,
                        'lr': optimizer.param_groups[0]['lr']
                    })
        if best_acc < test_scores.avg:
            best_acc = test_scores.avg
            torch.save({'state_dict': studentModel.state_dict()}, os.path.join(save_model_path, 'student_best.pth'))
        val_logger.log({
                        'epoch': epoch,
                        'loss': test_losses.avg,
                        'acc': test_scores.avg,
                        'best_acc' : best_acc,
                        'lr': optimizer.param_groups[0]['lr']
                    })
        writer.add_scalar('Loss/train', train_losses.avg, epoch)
        writer.add_scalar('Loss/test', test_losses.avg, epoch)
        
        writer.add_scalar('scores/train', train_scores.avg, epoch)
        writer.add_scalar('scores/test', test_scores.avg, epoch)
        torch.save({'state_dict': studentModel.state_dict()}, os.path.join(save_model_path, 'student_lastest.pth'))  # save spatial_encoder
       
