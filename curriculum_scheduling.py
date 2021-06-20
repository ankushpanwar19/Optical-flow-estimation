# Based on https://github.com/ClementPinard/FlowNetPytorch

import argparse
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
import flow_transforms
from models.raft_full import BasicRAFT
from models.raft_small import RAFT
from models.raft import RAFTNet
import models
import datasets
from multiscaleloss import multiscaleEPE, realEPE,individual_epe
import datetime
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))
model_names.append('raft')
dataset_names = sorted(name for name in datasets.__all__)


parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--name', default='demo', help='name of the experiment')
parser.add_argument('--dataset', metavar='DATASET', default='flying_chairs',
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
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--sparse', action='store_true',
                    help='look for NaNs in target flow when computing EPE, avoid if flow is garantied to be dense,'
                    'automatically seleted when choosing a KITTIdataset')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--div-flow', default=20, type=float,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--exp_desc', default="Test", type=str, help='helps to identify the experiment changes')
parser.add_argument('--schedule_out', default="scheduling",metavar='DIR', help='path to schedule index ouput')

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


    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255])
    ])

    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        # transforms.Normalize(mean=[0,0],std=[args.div_flow,args.div_flow])
    ])

    co_transform=None

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
        num_workers=args.workers, pin_memory=True, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False)

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
    else:
        model = models.__dict__[args.arch](data=network_data).to(device=device)
        model = torch.nn.DataParallel(model).to(device=device)
        cudnn.benchmark = True

    model = model.to(device=device)
   
    with torch.no_grad():
        sort_idx = validate(train_loader, model, 0)

    np.save(os.path.join(args.schedule_out, 'sortidx_for_schedule_full'),sort_idx)

    print("Finished")



def validate(val_loader, model, epoch):
    global args

    # switch to evaluate mode
    model.eval()
    epe_score_list=[]
    end = time.time()
    for i, (input, target) in enumerate(tqdm(val_loader)):
        target = target.to(device)
        input = torch.cat(input,1).to(device)

        # compute output
        if args.arch == 'raft':
            image1, image2 = torch.split(input, [3, 3], dim=1)
            output = model(image1, image2)
            flow2_EPE = individual_epe(output[-1], target)
        else:
            output = model(input)
            flow2_EPE = args.div_flow*individual_epe(output, target)

        epe_score_list+=flow2_EPE.tolist()
        # if len(epe_score_list)>8:
        #     break
        
    epe_score_list=np.array(epe_score_list)
    sort_idx=np.argsort(epe_score_list)

    return sort_idx


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
