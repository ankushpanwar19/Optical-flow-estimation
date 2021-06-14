# Based on https://github.com/ClementPinard/FlowNetPytorch

import argparse
from models.raft_full import BasicRAFT
from models.raft_small import RAFT
from models.raft import RAFTNet
import os
from random import Random
import shutil
import time
import json

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets.data_scheduler import CurriculumSampler

from spatial_correlation_sampler import spatial_correlation_sample

import flow_transforms
import models
import datasets
from multiscaleloss import multiscaleEPE, realEPE, sequence_loss
import datetime
from tensorboardX import SummaryWriter
import numpy as np

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))
model_names.append('raft')
dataset_names = sorted(name for name in datasets.__all__)


parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR',
                    default='/cluster/project/infk/hilliges/lectures/mp21/project6/dataset/', 
                    help='path to dataset')
parser.add_argument('--name', default='demo', help='name of the experiment')
parser.add_argument('--dataset', metavar='DATASET', default='humanflow',
                    choices=dataset_names,
                    help='dataset type : ' +
                    ' | '.join(dataset_names))
group = parser.add_mutually_exclusive_group()
group.add_argument('-s', '--split-file', default=None, type=str,
                   help='test-val split file')
group.add_argument('--split-value', default=0.8, type=float,
                   help='test-val split proportion (between 0 (only test) and 1 (only train))')
parser.add_argument('--arch', '-a', metavar='ARCH', default='flownets',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--batch_size',
                    default=1, type=int,
                    help='batch_size')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--multiscale-weights', '-w', default=[0.005,0.01,0.02,0.08,0.32], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
parser.add_argument('--sparse', action='store_true',
                    help='look for NaNs in target flow when computing EPE, avoid if flow is garantied to be dense,'
                    'automatically seleted when choosing a KITTIdataset')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--div-flow', default=20, type=float,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--milestones', default=[15,30,40], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')
parser.add_argument('--exp_desc', default="Test", type=str, help='helps to identify the experiment changes')
parser.add_argument('--curriculum_learn', type=bool, default=False)
parser.add_argument('--curriculum_idx_path', default="scheduling/sortidx_for_schedule.npy", help='file to schedule curriculum learning')

best_EPE = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def main():
    global args, best_EPE, save_path
    args = parser.parse_args()
    save_path = args.name
    save_path = os.path.join('checkpoints', save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # saving configuration to json file
    config_path= os.path.join(save_path,"config.json")
    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # saving code file to  zi[]
    zip_path=config_path= os.path.join(save_path,"code.zip")
    files_to_zip=" models datasets evaluate_humanflow.py flow_transforms.py generate_visuals.py main.py multiscaleloss.py run_inference.py test_humanflow.py"
    os.system('zip -r '+ zip_path+files_to_zip)
    
    train_writer = SummaryWriter(os.path.join(save_path,'train'))
    test_writer = SummaryWriter(os.path.join(save_path,'test'))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path,'test',str(i))))

    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255])
    ])

    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        # transforms.Normalize(mean=[0,0],std=[args.div_flow,args.div_flow])
    ])

    if 'KITTI' in args.dataset:
        args.sparse = True
    if args.sparse:
        co_transform = flow_transforms.Compose([
            flow_transforms.RandomCrop((320,448)),
            flow_transforms.RandomVerticalFlip(),
            flow_transforms.RandomHorizontalFlip()
        ])
    else:
        co_transform = flow_transforms.Compose([
            flow_transforms.RandomColorWarp(10,0.1),
            flow_transforms.RandomScale(0.8,1.2),
            flow_transforms.RandomTranslate(10),
            flow_transforms.RandomRotate(10,5),
            flow_transforms.RandomCrop((320,448)),
            flow_transforms.RandomVerticalFlip(),
            flow_transforms.RandomHorizontalFlip()
        ])

    print("=> fetching img pairs in '{}'".format(args.data))
    train_set, test_set = datasets.__dict__[args.dataset](
        args.data,
        transform=input_transform,
        target_transform=target_transform,
        co_transform=co_transform,
        split=args.split_file if args.split_file else args.split_value
    )
    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=True)

    # create model
    if args.pretrained:
        network_data = torch.load(args.pretrained,map_location=device)
        args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))
    
    if args.arch == 'raft':
        # model = torch.nn.DataParallel(RAFTNet())
        model = torch.nn.DataParallel(BasicRAFT())
        data = torch.load("./models/raft_models/raft-things.pth", map_location=device)
        model.load_state_dict(data)

        # model = torch.nn.DataParallel(RAFT())
        # check_point = torch.load(
        #     "./models/raft_models/raft-small.pth", map_location=device)
        # model.load_state_dict(check_point)
    
    else:
        model = models.__dict__[args.arch](data=network_data).to(device=device)
        model = torch.nn.DataParallel(model)
        
    model = model.to(device=device)
    cudnn.benchmark = True

    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    param_groups = model.parameters()

    if args.solver == 'adam':
        # optimizer = torch.optim.Adam(param_groups, args.lr,
                                    #  betas=(args.momentum, args.beta), weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, args.lr, weight_decay=args.weight_decay)
    
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)

    if args.evaluate:
        best_EPE = validate(val_loader, model, 0, output_writers)
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    ##### curriculium learning parameters ######
    sort_idx_path=args.curriculum_idx_path
    sort_idx = np.load(sort_idx_path)

    initial_ratio=0.5
    power=0.5
    scheduled_ratio = lambda ep: (1. - initial_ratio) / ((args.epochs - int(args.epochs/5)) ** power) * (ep ** power) + initial_ratio
    schedule_strategy='expand'
    ratio=initial_ratio

    is_best = False
    for epoch in range(args.start_epoch, args.epochs):

        if args.curriculum_learn==True and ratio<1:
            ratio = float(scheduled_ratio(epoch))
            # print('Epoch: {}, using sampler'.format(epoch))
            sampler = CurriculumSampler(train_set, sort_idx, ratio,initial_ratio, schedule_strategy, seed=epoch)

            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,num_workers=args.workers, pin_memory=True,sampler=sampler)
        
        print('Epoch: {}, using sampler and data batches:{}'.format(epoch,len(train_loader)))

        # train for one epoch
        train_loss, train_EPE = train(train_loader, model, optimizer, epoch, train_writer)
        train_writer.add_scalar('mean EPE', train_EPE, epoch)
        
        scheduler.step()
        # evaluate on validation set

        with torch.no_grad():
            EPE = validate(val_loader, model, epoch, output_writers)
        test_writer.add_scalar('mean EPE', EPE, epoch)

        if best_EPE < 0:
            best_EPE = EPE

        is_best = EPE < best_EPE
        best_EPE = min(EPE, best_EPE)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_EPE': best_EPE,
            'div_flow': args.div_flow
        }, is_best)


def train(train_loader, model, optimizer, epoch, train_writer,):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    model.train()

    end = time.time()
    #import ipdb; ipdb.set_trace()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.to(device)
        input = torch.cat(input,1).to(device) #concating left and right images
        
        # compute output
        
        if args.arch == 'raft':
            img1, img2 = torch.split(input, [3, 3], dim=1)
            output = model(img1, img2)
            loss, flow2_EPE = sequence_loss(output, target)
            # flow2_EPE = args.div_flow * realEPE(output[-1], target, sparse=args.sparse)
        else:
            output = model(input)
            if args.sparse:
                # Since Target pooling is not very precise when sparse,
                # take the highest resolution prediction and upsample it instead of downsampling target
                h, w = target.size()[-2:]
                output = [F.interpolate(output[0], (h,w)), *output[1:]]
            
            loss = multiscaleEPE(output, target, weights=args.multiscale_weights, sparse=args.sparse)
        
            flow2_EPE = args.div_flow * realEPE(output[0], target, sparse=args.sparse)
        # record loss and EPE
        losses.update(loss.item(), target.size(0))
        train_writer.add_scalar('train_loss', loss.item(), n_iter)
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            
            result_str = 'Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t EPE {6}'.format(
                epoch, i, epoch_size, batch_time, data_time, losses, flow2_EPEs)
            
            # with open('./train-stats/result.txt', 'a') as f:
            #     f.write(result_str + "\n")
            
            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t EPE {6}'
                  .format(epoch, i, epoch_size, batch_time,
                          data_time, losses, flow2_EPEs))
        n_iter += 1
        break
        if i >= epoch_size:
            break

    return losses.avg, flow2_EPEs.avg


def validate(val_loader, model, epoch, output_writers):
    global args

    batch_time = AverageMeter()
    flow2_EPEs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = torch.cat(input,1).to(device)

        # compute output
        if args.arch == 'raft':
            img1, img2 = torch.split(input, [3, 3], dim=1)
            output = model(img1, img2)
            # flow2_EPE = args.div_flow * realEPE(output[-1], target, sparse=args.sparse)
            flow2_EPE = torch.sum((output[-1] - target)**2, dim=1).sqrt().mean()
        else:
            output = model(input)
            flow2_EPE = args.div_flow*realEPE(output, target, sparse=args.sparse)
        # record EPE
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i < len(output_writers):  # log first output of first batches
        #     if epoch == 0:
        #         mean_values = torch.tensor([0.411,0.432,0.45], dtype=input.dtype).view(3,1,1)
        #         output_writers[i].add_image('GroundTruth', flow2rgb(args.div_flow * target[0], max_value=10), 0)
        #         output_writers[i].add_image('Inputs', (input[0,:3].cpu() + mean_values).clamp(0,1), 0)
        #         output_writers[i].add_image('Inputs', (input[0,3:].cpu() + mean_values).clamp(0,1), 1)
        #     output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output[0], max_value=10), epoch)

        if i % args.print_freq == 0:
            
            result_str = 'Test: [{0}/{1}]\t Time {2}\t EPE {3}'.format(
                i, len(val_loader), batch_time, flow2_EPEs)
            
            # with open('./train-stats/val_result.txt', 'a') as f:
            #     f.write(result_str + "\n")
                
            print('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                  .format(i, len(val_loader), batch_time, flow2_EPEs))

    print(' * EPE {:.3f}'.format(flow2_EPEs.avg))

    return flow2_EPEs.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

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

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)


if __name__ == '__main__':
    import sys
    with open("experiment_recorder.md", "a") as f:
        f.write('\n python3 ' + ' '.join(sys.argv))
    main()
