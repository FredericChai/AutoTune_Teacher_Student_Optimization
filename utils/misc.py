import shutil
import torch
import sys
import os
import sklearn 
import re
import math
import numpy
import torch.nn as nn
import torch.nn.functional as F

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accracy_k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def make_path(path):
    """make dirs for checkpoint"""
    if not os.path.isdir(path):
        os.makedirs(path)



def loss_fn_kd(outputs, labels, teacher_outputs,alpha,temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = alpha
    T = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def write_report(result_path,report,best_epoch,best_acc,predict,target):
    writer = open(os.path.join(result_path,'classification_report.txt'),'w')
    writer.write(report+'\n')
    writer.write('best_epoch:'+str(best_epoch)+'\n')
    writer.write('best_acc:'+str(best_acc))
    writer.close()
    # write down predict result
    predict_writer = open(os.path.join(result_path,'best_predict.txt'),'w')
    predict_writer.write(predict)
    predict_writer.close()
    #write down target result
    target_writer = open(os.path.join(result_path,'target.txt'),'w')
    target_writer.write(target)
    target_writer.close()

# end the iterative sparse train 
    return

def load_model(model_arch,numOfClasses):

    if model_arch.endswith('vgg16'):
        NN = VGG.vgg16(num_classes = numOfClasses)
    elif args.arch.endswith('vgg19'):
        NN = VGG.vgg19(num_classes = numOfClasses)
    elif model_arch.endswith('vgg16_pretrained'):
        NN = models.vgg16(pretrained = True)
        num_features = NN.classifier[6].in_features  
        NN.classifier[6] = nn.Linear(num_features,numOfClasses)  #change last fc layer and keep all other layer if used pretrained model
    elif model_arch.endswith('resnet101_pretrained') or model_arch.endswith('resnet101') :
        NN = models.resnet101(pretrained = True)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch.endswith('resnet18_pretrained') or model_arch.endswith('resnet18'):
        NN = models.resnet18(pretrained = True)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch.endswith('resnet34_pretrained') or model_arch.endswith('resnet34') :
        NN = models.resnet34(pretrained = True)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch.endswith('resnet50_pretrained') or model_arch.endswith('resnet50'):
        NN = models.resnet50(pretrained = True)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch.endswith('resnet50') or model_arch.endswith('resnet50'):
        NN = models.resnet50(pretrained = True)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch.endswith('resnet152_pretrained'):
        NN = models.resnet152(pretrained = True)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch.endswith('resnet152'):
        NN = models.resnet152(pretrained = False)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
