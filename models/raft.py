from typing import Sequence
import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary
from .utils import coords_grid, bilinear_sampler

autocast = torch.cuda.amp.autocast


def make_residual_block(in_channels, out_channels, stride):
    layer1 = ResidualBlock(in_channels, out_channels, stride=stride)
    layer2 = ResidualBlock(out_channels, out_channels, stride=1)

    return [layer1, layer2]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm_fn='bn'):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        if norm_fn == 'bn':
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
            self.norm3 = nn.BatchNorm2d(out_channels)
            
        elif norm_fn == 'in':
            self.norm1 = nn.InstanceNorm2d(out_channels)
            self.norm2 = nn.InstanceNorm2d(out_channels)
            self.norm3 = nn.InstanceNorm2d(out_channels)
        
        if stride == 1:
            self.down_sampling = None
        else:
            self.down_sampling = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                self.norm3
            )
    
    def forward(self, x):
        y = x.clone()
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        
        if self.down_sampling is not None:
            x = self.down_sampling(x)
        
        return self.relu(x + y)
        

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_fn='bn', stride=1):
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//4, 
                               kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=out_channels//4, out_channels=out_channels//4,
                               kernel_size=3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(in_channels=out_channels//4, out_channels=out_channels,
                               kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        
        if norm_fn == 'bn':
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
            self.norm3 = nn.BatchNorm2d(out_channels)
            self.norm4 = nn.BatchNorm2d(out_channels)
            
        elif norm_fn == 'in':
            self.norm1 = nn.InstanceNorm2d(out_channels)
            self.norm2 = nn.InstanceNorm2d(out_channels)
            self.norm3 = nn.InstanceNorm2d(out_channels)
            self.norm4 = nn.InstanceNorm2d(out_channels)
        
        if stride == 1:
            self.down_sampling = None
        else:
            self.down_sampling = nn.Sequential(
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=1, stride=stride),
                self.norm4
            )
        
        super().__init__()
    
    def forward(self, x):
        y = x.clone()
        
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))
        
        if self.down_sampling is not None:
            y = self.down_sampling(y)
            
        return self.relu(x + y)


class Encoder(nn.Module):
    def __init__(self, output_dim=256, norm_fn='bn', drop_out=0.0):
        super().__init__()
        
        if norm_fn == 'bn':
            norm_layer = nn.BatchNorm2d(64)
        elif norm_fn == 'in':
            norm_layer = nn.InstanceNorm2d(64)
        else:
            norm_layer = nn.Sequential()
        
        modules = [
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=7, stride=2, padding=3),
            norm_layer,
            nn.ReLU(inplace=True)
        ]
        
        modules.extend(make_residual_block(64, 64, stride=1))
        modules.extend(make_residual_block(64, 96, stride=2))
        modules.extend(make_residual_block(96, 128, stride=2))
        
        modules.append(
            nn.Conv2d(in_channels=128, out_channels=output_dim, kernel_size=1)
        )
        
        if drop_out > 0.0:
            modules.append(nn.Dropout2d(drop_out))
        
        self.encoder = nn.Sequential(*modules)
        self.init_params()
        
    def init_params(self):
        for block in self._modules:
            for m in block:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        is_list = isinstance(x, Sequence)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        
        x = self.encoder(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        
        return x



class ConvGRU(nn.Module):
    def __init__(self, h_dim=128, x_dim=128 + 192):
        super().__init__()

        self.conv_z = nn.Conv2d(in_channels=h_dim + x_dim,
                               out_channels=h_dim, kernel_size=3, padding=1)
        self.conv_r = nn.Conv2d(in_channels=h_dim + x_dim,
                               out_channels=h_dim, kernel_size=3, padding=1)
        self.conv_h = nn.Conv2d(in_channels=h_dim + x_dim,
                               out_channels=h_dim, kernel_size=3, padding=1)
        
    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        
        z = torch.sigmoid(self.conv_z(hx))
        r = torch.sigmoid(self.conv_r(hx))
        
        rh = torch.cat([r * h, x], dim=1)
        h_tilde = torch.tanh(self.conv_h(rh))
        
        return (1 - z) * h + z * h_tilde



class Motionencoder(nn.Module):
    def __init__(self, corr_levels=4, corr_radius=4):
        super().__init__()
        out_corr_dim = 192
        out_flow_dim = 64
        corr_channels = corr_levels * (2 * corr_radius + 1) ** 2
        self.corr_conv = nn.Sequential(
            nn.Conv2d(in_channels=corr_channels, out_channels=256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=out_corr_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.flow_conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=out_flow_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_corr_dim + out_flow_dim, out_channels=128-2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, flow, corr):
        out_flow = self.flow_conv(flow)
        out_corr = self.corr_conv(corr)
        out = torch.cat([out_corr, out_flow], dim=1)
        out = self.final_conv(out)
        
        return torch.cat([out, flow], dim=1)
        
        
class UpdateBlock(nn.Module):
    def __init__(self, corr_levels=4, corr_radius=4, h_dim=128, in_dim=128):
        super().__init__()
        self.motion_encoder = Motionencoder(corr_levels, corr_radius)
        self.conv_gru = ConvGRU(h_dim, in_dim + h_dim)
        
        final_in_channels = h_dim
        final_out_channels = 128
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=final_in_channels,
                      out_channels=final_out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=final_out_channels,
                      out_channels=2, kernel_size=3, padding=1),
        )
        
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        
    def forward(self, prev_state, context, corr, flow):
        motion_features = self.motion_encoder(flow, corr)
        inp = torch.cat([context, motion_features], dim=1)

        state = self.conv_gru(prev_state, inp)
        final_flow = self.final_conv(state)

        # scale mask to balence gradients
        mask = .25 * self.mask(state)
        return state, mask, final_flow


class CorrelationBlock(object):
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrelationBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for _ in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx),
                                axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class RAFTNet(nn.Module):
    def __init__(self, hdim=128, cdim=128, corr_levels=4, corr_radius=4, drop_out=0.0, mixed_precision=False):
        super().__init__()
        
        self.hdim = hdim
        self.cdim = cdim
        self.corr_radius = corr_radius
        self.mixed_precision = mixed_precision
        
        self.fnet = Encoder(output_dim=256, norm_fn='in', drop_out=drop_out)        
        self.cnet = Encoder(output_dim=hdim+cdim, norm_fn='bn', drop_out=drop_out)
        self.update_block = UpdateBlock(corr_levels=corr_levels, corr_radius=corr_radius, h_dim=hdim)
    
    
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)
        
    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hdim
        cdim = self.cdim

        # run the feature network
        with autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrelationBlock(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        with autocast(enabled=self.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions
