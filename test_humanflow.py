import argparse
import glob
from models import RAFTNet
import os
import numpy as np
import cv2

# from scipy.ndimage import imread
from matplotlib.pyplot import imread
# from scipy.misc import imsave
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
import models
import datasets
from multiscaleloss import realEPE, motion_warping_error
import flow_transforms
from tqdm import tqdm



parser = argparse.ArgumentParser(description='Test Optical Flow',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', dest='arch', type=str, default='pwc', choices=['raft','pwc', 'spynet', 'flownet2'],
                    help='flow network architecture. Options: pwc | spynet')
parser.add_argument('--dataset', dest='dataset', default='KITTI', choices=['KITTI_occ', 'humanflow'],
                    help='test dataset')
parser.add_argument('--div-flow', default=1, type=float,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--no-norm', action='store_true',
                    help='don\'t normalize the image' )
parser.add_argument('--pretrained', metavar='PTH', default=None, help='path to pre-trained model')
parser.add_argument('--save-name', dest='save_name', type=str, default=None,
                    help='flow network architecture. Options: Name for saving results')
parser.add_argument('--output-dir', dest='output_dir', metavar='DIR', default=None, help='path to output flo')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

def main():
    global args
    args = parser.parse_args()
    # test_list = make_dataset(args.data)
    test_list = make_dataset_new(args.data, phase="test",flowmap_exist=False)
    # test_list = make_real_dataset(args.data)

    # if args.arch == 'pwc':
    #     model = models.pwc_dc_net('models/pwc_net_ft.pth.tar').cuda()
    # elif args.arch == 'spynet':
    #     model = models.spynet(nlevels=5, strmodel='F').cuda()
    # elif args.arch == 'flownet2':
    #     model = models.FlowNet2().cuda()
    #     print("=> using pre-trained weights for FlowNet2")
    #     weights = torch.load('models/FlowNet2_checkpoint.pth.tar')
    #     model.load_state_dict(weights['state_dict'])

    if args.pretrained is not None:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch == 'raft':
            checkpoint = torch.load(args.pretrained, map_location=device)
            model = models.BasicRAFT()
            model.load_state_dict(checkpoint['state_dict'])
        else:
            network_data = torch.load(args.pretrained,map_location=device)
            args.arch = network_data['arch']    
            model = models.__dict__[args.arch](data=network_data).to(device=device)
            if 'div_flow' in network_data.keys():
                args.div_flow = network_data['div_flow']

    else:
        model = models.pwc_dc_net('models/pwc_net.pth.tar').to(device=device)
    
    
    
    model.to(device)
    
    model.eval()
    flow_epe = AverageMeter()
    avg_mot_err = AverageMeter()

    avg_parts_epe = {}
    for bk in BODY_MAP.keys():
        avg_parts_epe[bk] = AverageMeter()

    if args.no_norm:
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0,0,0], std=[255,255,255])
        ])
    else:
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
            transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
        ])
        
    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        # transforms.Normalize(mean=[0,0],std=[args.div_flow, args.div_flow])
    ])


    for i, (img_paths, flow_path, seg_path) in enumerate(tqdm(test_list)):
        # import pdb
        # pdb.set_trace()
        raw_im1 = flow_transforms.ArrayToTensor()(255*imread(img_paths[0])[:,:,:3])
        raw_im2 = flow_transforms.ArrayToTensor()(255*imread(img_paths[1])[:,:,:3])

        img1 = input_transform(255*imread(img_paths[0])[:,:,:3])
        img2 = input_transform(255*imread(img_paths[1])[:,:,:3])

        if flow_path is None:
            _, h, w = img1.size()
        #     new_h = int(np.floor(h/256)*256)
        #     new_w = int(np.floor(w/448)*448)

        #     # if i>744:
        #     #     import ipdb; ipdb.set_trace()
        #     img1 = F.upsample(img1.unsqueeze(0), (new_h,new_w), mode='bilinear').squeeze()
        #     img2 = F.upsample(img2.unsqueeze(0), (new_h,new_w), mode='bilinear').squeeze()


        if flow_path is not None:
            gtflow = target_transform(load_flo(flow_path))
            segmask = flow_transforms.ArrayToTensor()(cv2.imread(seg_path))

        input_var = torch.cat([img1, img2]).unsqueeze(0)

        if flow_path is not None:
            gtflow_var = gtflow.unsqueeze(0)
            segmask_var = segmask.unsqueeze(0)

        input_var = input_var.to(device)

        if flow_path is not None:
            gtflow_var = gtflow_var.to(device)
            segmask_var = segmask_var.to(device)

        # compute output
        model.eval()
        image1, image2 = torch.split(input_var, [3, 3], dim=1)
        output = model(image1, image2, iters=16)[-1]
        output = output/20.0
        
        if flow_path is not None:
            epe = args.div_flow*realEPE(output, gtflow_var, sparse=True if 'KITTI' in args.dataset else False)
            epe_parts = partsEPE(output, gtflow_var, segmask_var)
            epe_parts.update((x, args.div_flow*y) for x, y in epe_parts.items() )

            # record EPE
            flow_epe.update(epe.item(), gtflow_var.size(0))
            for bk in avg_parts_epe:
                if epe_parts[bk].item() > 0:
                    avg_parts_epe[bk].update(epe_parts[bk].item(), gtflow_var.size(0))

        # record motion warping error
        raw_im1 = raw_im1.to(device=device).unsqueeze(0)
        raw_im2 = raw_im2.to(device=device).unsqueeze(0)
        mot_err = motion_warping_error(raw_im1, raw_im2, args.div_flow*output)
        avg_mot_err.update(mot_err.item(), raw_im1.size(0))

        if args.output_dir is not None:
            if flow_path is not None:
                _, h, w = gtflow.size()
                output_path = flow_path.replace(args.data, args.output_dir)
                output_path = output_path.replace('/test/','/')
                os.system('mkdir -p '+output_path[:-15])
            else:
                output_path = img_paths[0].replace(args.data, args.output_dir)
                output_path = output_path.replace('/test/','/')
                output_path = output_path.replace('/composition/','/')
                os.system('mkdir -p '+output_path[:-10])
                output_path = output_path.replace('.png', '.flo')
            upsampled_output = F.interpolate(output, (h//4,w//4), mode='bilinear', align_corners=False) # resize to 0.25 for storage
            flow_write(output_path,  upsampled_output.cpu()[0].data.numpy()[0],  upsampled_output.cpu()[0].data.numpy()[1])
            # flow_write(output_path,  output.cpu()[0].data.numpy()[0],  output.cpu()[0].data.numpy()[1])

    # if args.save_name is not None:
    #     epe_dict = {}
    #     for bk in BODY_MAP.keys():
    #         epe_dict[bk] = avg_parts_epe[bk].avg
    #     epe_dict['full_epe'] = flow_epe.avg
    #     np.save(os.path.join('results', args.save_name), epe_dict)

    # print("Averge EPE",flow_epe.avg )
    # print("Motion warping error", avg_mot_err.avg)

def partsEPE(output, gtflow, seg_mask):
    parts_epe_dict = {}
    for bk in BODY_MAP.keys():
        mask = seg_mask == BODY_MAP[bk]
        gt_partflow = mask.type_as(gtflow)[:,:2,:,:] * gtflow
        epe_part = realEPE(output, gt_partflow, sparse=True)
        parts_epe_dict[bk] = epe_part

    return parts_epe_dict


def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    return data2D

def make_dataset(dir, phase='test'):
    '''Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm    [name]_flow.flo' '''
    images = []
    for flow_map in sorted(glob.glob(os.path.join(dir, phase+'/*/flow/*.flo'))):
        #flow_map = os.path.relpath(flow_map, dir)
        img1 = flow_map.replace('/flow/', '/composition/')
        img1 = img1.replace('.flo', '.png')
        img2 = img1[:-9] + str(int(img1.split('/')[-1][:-4])+1).zfill(5) + '.png'

        seg_mask = flow_map.replace('/flow/', '/segm_EXR/')
        seg_mask = seg_mask.replace('.flo', '.exr')

        if int(img1.split('/')[-1][:-4]) % 10 == 9:
            continue

        if not (os.path.isfile(os.path.join(dir,img1)) and os.path.isfile(os.path.join(dir,img2))):
            continue

        images.append([[img1,img2],flow_map, seg_mask])

    return images


def make_dataset_new(dir, phase='test',flowmap_exist=False):
    '''Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm    [name]_flow.flo' '''
    images = []
    flow_map=None
    for img1 in sorted( glob.glob(os.path.join(dir, phase+'/*/composition/*.png')) ):
        #flow_map = os.path.relpath(flow_map, dir)
        if flowmap_exist:
            flow_map = img1.replace('/composition/', '/flow/')
            flow_map = flow_map.replace('.png', '.flo')

        img2 = img1[:-9] + str(int(img1.split('/')[-1][:-4])+1).zfill(5) + '.png'

        seg_mask = img1.replace('/composition/', '/segm_EXR/')
        seg_mask = seg_mask.replace('.png', '.exr')

        if int(img1.split('/')[-1][:-4]) % 10 == 9:
            continue

        if not (os.path.isfile(os.path.join(dir,img1)) and os.path.isfile(os.path.join(dir,img2))):
            continue

        images.append([[img1,img2],flow_map, seg_mask])

    return images


def make_real_dataset(dir):
    '''Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm    [name]_flow.flo' '''
    images = []
    for img1 in sorted( glob.glob(os.path.join(dir, '*/composition/*.png')) ):
        img2 = img1[:-9] + str(int(img1.split('/')[-1][:-4])+1).zfill(5) + '.png'

        if int(img1.split('/')[-1][:-4]) % 10 == 9:
            continue

        if not (os.path.isfile(os.path.join(dir,img1)) and os.path.isfile(os.path.join(dir,img2))):
            continue

        images.append([[img1,img2],None, None])

    return images

TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'.encode()

def flow_write(filename,uv,v=None):
    """ Write optical flow to file.
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        uv_ = np.array(uv)
        assert(uv_.ndim==3)
        if uv_.shape[0] == 2:
            u = uv_[0,:,:]
            v = uv_[1,:,:]
        elif uv_.shape[2] == 2:
            u = uv_[:,:,0]
            v = uv_[:,:,1]
        else:
            raise UVError('Wrong format for flow input')
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float16).tofile(f)
    f.close()

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

BODY_MAP = {'global': 1, 'head': 16, 'lIndex0': 23, 'lIndex1': 33, 'lIndex2': 43,
        'lMiddle0': 24, 'lMiddle1': 34, 'lMiddle2': 44, 'lPinky0': 25,
        'lPinky1': 35, 'lPinky2': 45, 'lRing0': 26, 'lRing1': 36, 'lRing2': 46,
        'lThumb0': 27, 'lThumb1': 37, 'lThumb2': 47, 'leftCalf': 5, 'leftFoot': 8,
        'leftForeArm': 19, 'leftHand': 21, 'leftShoulder': 14, 'leftThigh': 2, 'leftToes': 11,
        'leftUpperArm': 17, 'neck': 13, 'rIndex0': 28, 'rIndex1': 38, 'rIndex2': 48,
        'rMiddle0': 29, 'rMiddle1': 39, 'rMiddle2': 49, 'rPinky0': 30, 'rPinky1': 40,
        'rPinky2': 50, 'rRing0': 31, 'rRing1': 41, 'rRing2': 51, 'rThumb0': 32,
        'rThumb1': 42, 'rThumb2': 52, 'rightCalf': 6, 'rightFoot': 9, 'rightForeArm': 20,
        'rightHand': 22, 'rightShoulder': 15, 'rightThigh': 3, 'rightToes': 12, 'rightUpperArm': 18,
        'spine': 4, 'spine1': 7, 'spine2': 10}


if __name__ == '__main__':
    main()
