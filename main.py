from __future__ import print_function
import os
import argparse
import math
import random
import time

import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.metrics import recall_score,accuracy_score
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt import UtilityFunction

from model import VGG 
from utils import *

parser = argparse.ArgumentParser(description = 'Pytorch VGG on PET-CT image')
#path of dataset
parser.add_argument('--dataset',default = '',type = str,help = 'path of dataset')
#configuration 
parser.add_argument('--epochs',default = 100,type=int,help='epochs for each sparse model to run')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')                 
parser.add_argument('--schedule', type=int, default=300,help='Set epochs to decrease learning rate.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--arch',default='',type=str,help='VGG Architechture to load the checkpoint')

#Meter for measure()
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--evaluate', default = 0,type = int,help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--source_checkpoint', default = '', help = 'path to sav81e the pruned parameters as checkpoint')
parser.add_argument('--eval_model', default = '', help = 'path to save the pruned parameters as checkpoint')
parser.add_argument('--alpha', type=float, help='parameters to control KD')
parser.add_argument('--temperature', type=float, help='parameters to control KD')
# parser.add_argument('--teacher_arch', type=str, help='parameters to control KD')
# parser.add_argument('--evaluate_only',dest = 'evaluate_only',action = 'store_true',help='if in this mode. directly load checkpoint and ')

# parser.add_argument('--norm_activation',dest = 'norm_activation',action = 'store_true')
# parser.add_argument('--filter_prune',dest = 'filter_prune',action = 'store_true')


#load all args
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

#Define the id of device
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

#Set Random seed for 
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

def main():
    bounds = {'alpha':(0.3,0.9),'temperature':(3,15)}
    optimizer = BayesianOptimization(
        f = KD_train,
        pbounds = bounds,
        verbose = 2,
        random_state = 0)
    utility = UtilityFunction(kind = 'ei', kappa= 1,xi=0.0)

    for _ in range(5):
        next_p = optimizer.suggest(utility)
        print('suggest for next:',next_p)
        result = KD_train(**next_p)
        optimizer.register(params = next_p,target = result)

    for i,res in enumerate(optimizer.res):
        print("ite {} \t {}".format(i,res))

    logger = JSONLogger(path = './BO_logs.json')
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

def KD_train(alpha,temperature):
    """the normal training for kd train"""
    args.alpha = alpha
    args.temperature = temperature
    result_path = args.dataset + 'Basian_TA-Net_checkpoint/'+args.arch+'-checkpoint-epoch'+str(args.epochs)+'/alpha-'+\
                    str(args.alpha)+'-T-'+str(args.temperature)   
    make_path(result_path) 

    #load the data
    dataloaders,testDataloaders,dataset_sizes,class_names,numOfClasses = load_data(args.dataset+'Data_new')
    #load teacher model and checkpoint
    NN = load_model(args.arch,numOfClasses)
    NN = torch.nn.DataParallel(NN).cuda() 
    cudnn.benchmark = True
    # print('  Total parameters: %.2f' % (sum(p.numel() for p in NN.parameters())))

    if args.evaluate==1:
        args.eval_model = result_path+"/model_best.pth.tar"
        evaluate_checkpoint = torch.load(args.eval_model)
        NN.load_state_dict(evaluate_checkpoint['state_dict'])
        report,predict,target = evaluate(testDataloaders, NN, class_names, use_cuda)
        best_epoch = evaluate_checkpoint['epoch']
        best_acc = evaluate_checkpoint['best_acc']
        write_report(result_path,report,best_epoch,best_acc,predict,target)
        return

    teacher_NN = load_model(args.arch,numOfClasses)
    teacher_NN = torch.nn.DataParallel(teacher_NN).cuda() 
    teacher_NN.load_state_dict(torch.load(args.source_checkpoint)['state_dict'])
    #loss funcion and optimizer
    optimizer = optim.SGD(NN.parameters(), lr=args.lr, momentum=args.momentum)
    title = 'KD_train-'+args.arch
    #load checkpoint
    logger = Logger(os.path.join(result_path, 'log.txt'),title = title)
    logger.set_names(['Best Acc','Trainning Loss','Valid Loss','Train Acc','Valid Acc'])  
    best_acc = 0 
    # Train and validate model
    for epoch in range(0,args.epochs): 
        # adjust the learning rate when the epoch is in the schdule
        adjust_learning_rate(optimizer,epoch,args.lr,args.schedule,args.gamma)
        for parameter_group in optimizer.param_groups:
            current = parameter_group['lr']
        print('\nEpoch: [%d | %d] temperature: %f best_acc: %f' % (epoch + 1, args.epochs,args.temperature, best_acc ))
        train_loss, train_acc = train(dataloaders['train'], NN, optimizer, epoch,use_cuda,args.alpha,args.temperature,teacher_NN)
        val_loss, val_acc = validate(testDataloaders, NN, epoch,use_cuda,args.alpha,args.temperature,teacher_NN)

        # append logger file
        logger.append([best_acc, train_loss, val_loss, train_acc, val_acc])

        # save model
        is_best = val_acc > best_acc 
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': NN.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=result_path)

    logger.close()
    plot_result(logger,result_path)
    savefig(os.path.join(result_path, 'log.eps'))
    print('Best acc:',best_acc)
    #evaluate the model based on the best model on val set
    evaluate_checkpoint = torch.load(os.path.join(result_path,'model_best.pth.tar'))
    NN.load_state_dict(evaluate_checkpoint['state_dict'])
    optimizer.load_state_dict(evaluate_checkpoint['optimizer'])
    best_epoch = evaluate_checkpoint['epoch']
    report,target,predict = evaluate(testDataloaders, NN, class_names, use_cuda)
    write_report(result_path,report,best_epoch,best_acc)
    return best_acc
    #write down classfication report
def train(trainloader,model,optimizer,epoch,use_cuda,alpha,temperature,teacher_NN):
    #train mode
    model.train()
    teacher_NN.eval()
    #metrics of the model
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    with tqdm(total = len(trainloader)) as pbar:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # measure data loading time
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            teacher_output = teacher_NN(inputs)
            loss = loss_fn_kd(outputs, targets, teacher_output,alpha,temperature)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()   
            pbar.set_description('loss: %.4f top1: %.4f' % (loss.view(-1).data.tolist()[0],top1.avg))
            pbar.update(1)
    return (losses.avg, top1.avg)

def validate(testloader, model, epoch, use_cuda,alpha,temperature,teacher_NN):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            torch.no_grad()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            teacher_output = teacher_NN(inputs)
            loss = loss_fn_kd(outputs, targets,teacher_output,alpha,temperature)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return (losses.avg, top1.avg)



def evaluate(testloader,model,class_names,use_cuda):

    #evaluate mode
    model.eval()
    pred = []
    targ = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        torch.no_grad()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        # loss = loss_fn_kd(outputs, targets)
        
        # record prediction and target 
        _,output = torch.max(outputs.data,1)
        pred += output.tolist()
        targ += targets.tolist()
    # sensitivity, F1-score
    report = classification_report(targ,pred,digits = 4,target_names =class_names)

    return report,targ,pred
 


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']



def load_data(path):
    #Load data and augment train data
    data_transforms = {
        #Augment the trainning data
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224), #crop the given image
            transforms.RandomHorizontalFlip(),  #horizontally flip the image
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        #Scale and normalize the validation data
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        #Scale and normalize the validation data
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}

    data_dir = path
    # testData_dir = '/mnt/HDD1/Frederic/ensemble_baseline/TestImage/'

    image_datasets = {
            x : datasets.ImageFolder(os.path.join(data_dir,x),
                                 data_transforms[x])
            for x in ['train','val','test']
        }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],     
                                                        batch_size=36, 
                                                        shuffle=True,
                                                        num_workers=0) 
                                                    for x in ['train','val']}

    testImageLoaders = torch.utils.data.DataLoader(image_datasets['test'],batch_size=24,shuffle=False)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    class_names = image_datasets['train'].classes
    numOfClasses = len(class_names)

    return dataloaders,testImageLoaders,dataset_sizes,class_names,numOfClasses

def write_report(result_path,report,best_epoch,best_acc):
    writer = open(os.path.join(result_path,'classification_report.txt'),'w')
    writer.write(report+'\n')
    writer.write('best_epoch:'+str(best_epoch)+'\n')
    writer.write('best_acc:'+str(best_acc))
    writer.close()
# end the iterative sparse train 
    return

def adjust_learning_rate(optimizer,epoch,learning_rate,schedule,gamma):
    if epoch == schedule:
        new_learning_rate = gamma*learning_rate
        for parameter_group in optimizer.param_groups:
            parameter_group['lr'] = new_learning_rate

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
    elif model_arch.endswith('resnet101') or model_arch.endswith('resnet18'):
        NN = models.resnet18(pretrained = False)
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



    return NN
if __name__ == '__main__':
    main()
