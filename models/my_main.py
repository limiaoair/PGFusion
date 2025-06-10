import torch
from torch import nn
import torch.nn.functional as F
from models.vgg import VGG_Backbone
from util import *
from models.ssm import *
from models.my_module import *
# from torch import nn, einsum
# from einops import rearrange
from models.sct import *


def weights_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class EnLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(EnLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x


class LatLayer(nn.Module):
    def __init__(self, in_channel):
        super(LatLayer, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x


class DSLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0))#, nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x


class half_DSLayer(nn.Module):
    def __init__(self, in_channel=512):
        super(half_DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, int(in_channel/4), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(int(in_channel/4), 1, kernel_size=1, stride=1, padding=0)) #, nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x


class AugAttentionModule(nn.Module):
    def __init__(self, input_channels=512):
        super(AugAttentionModule, self).__init__()
        self.query_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.key_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.value_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.scale = 1.0 / (input_channels ** 0.5)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        x = self.conv(x)
        x_query = self.query_transform(x).view(B, C, -1).permute(0, 2, 1)  # B,HW,C
        # x_key: C,BHW
        x_key = self.key_transform(x).view(B, C, -1)  # B, C,HW
        # x_value: BHW, C
        x_value = self.value_transform(x).view(B, C, -1).permute(0, 2, 1)  # B,HW,C
        attention_bmm = torch.bmm(x_query, x_key)*self.scale # B, HW, HW
        attention = F.softmax(attention_bmm, dim=-1)
        attention_sort = torch.sort(attention_bmm, dim=-1, descending=True)[1]
        attention_sort = torch.sort(attention_sort, dim=-1)[1]
        #####
        attention_positive_num = torch.ones_like(attention).cuda()
        attention_positive_num[attention_bmm < 0] = 0
        att_pos_mask = attention_positive_num.clone()
        attention_positive_num = torch.sum(attention_positive_num, dim=-1, keepdim=True).expand_as(attention_sort)
        attention_sort_pos = attention_sort.float().clone()
        apn = attention_positive_num-1
        attention_sort_pos[attention_sort > apn] = 0
        attention_mask = ((attention_sort_pos+1)**3)*att_pos_mask + (1-att_pos_mask)
        out = torch.bmm(attention*attention_mask, x_value)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)
        return out+x


class my_AugAttentionModule(nn.Module):
    def __init__(self, input_channels=512):
        super(my_AugAttentionModule, self).__init__()
        self.query_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.key_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.value_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.scale = 1.0 / (input_channels ** 0.5)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        # x.permute(0, 2, 3, 1)
        x = self.conv(x)
        x_query = self.query_transform(x).view(B, C, -1).permute(0, 2, 1)  # B,HW,C
        # x_key: C,BHW

        x_key = self.key_transform(x).view(B, C, -1)  # B, C,HW
        # x_value: BHW, C
        x_value = self.value_transform(x).view(B, C, -1).permute(0, 2, 1)  # B,HW,C

        attention_bmm = torch.bmm(x_query, x_key)*self.scale # B, HW, HW
        attention = F.softmax(attention_bmm, dim=-1)
        attention_sort = torch.sort(attention_bmm, dim=-1, descending=True)[1]
        attention_sort = torch.sort(attention_sort, dim=-1)[1]
        #####
        attention_positive_num = torch.ones_like(attention).cuda()
        attention_positive_num[attention_bmm < 0] = 0
        att_pos_mask = attention_positive_num.clone()
        attention_positive_num = torch.sum(attention_positive_num, dim=-1, keepdim=True).expand_as(attention_sort)
        attention_sort_pos = attention_sort.float().clone()
        apn = attention_positive_num-1
        attention_sort_pos[attention_sort > apn] = 0
        attention_mask = ((attention_sort_pos+1)**3)*att_pos_mask + (1-att_pos_mask)
        out = torch.bmm(attention*attention_mask, x_value)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)
        return out+x


class AttLayer(nn.Module):
    def __init__(self, input_channels=512):
        super(AttLayer, self).__init__()
        self.query_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.key_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.scale = 1.0 / (input_channels ** 0.5)
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)

    def correlation(self, x5, seeds):
        B, C, H5, W5 = x5.size()
        if self.training:
            correlation_maps = F.conv2d(x5, weight=seeds)  # B,B,H,W
        else:
            correlation_maps = torch.relu(F.conv2d(x5, weight=seeds))  # B,B,H,W
        correlation_maps = correlation_maps.mean(1).view(B, -1)
        min_value = torch.min(correlation_maps, dim=1, keepdim=True)[0]
        max_value = torch.max(correlation_maps, dim=1, keepdim=True)[0]
        correlation_maps = (correlation_maps - min_value) / (max_value - min_value + 1e-12)  # shape=[B, HW]
        correlation_maps = correlation_maps.view(B, 1, H5, W5)  # shape=[B, 1, H, W]
        return correlation_maps

    def forward(self, x5):
        # print(x5.shape) #torch.Size([16, 512, 24, 24])
        # x: B,C,H,W
        x5 = self.conv(x5)+x5
        B, C, H5, W5 = x5.size()
        x_query = self.query_transform(x5).view(B, C, -1)
        # print(x_query.shape) #torch.Size([16, 512, 576])
        # x_query: B,HW,C
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C)  # BHW, C
        # x_key: B,C,HW
        # print(x_query.shape)  #torch.Size([9216, 512])
        x_key = self.key_transform(x5).view(B, C, -1)
        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1)  # C, BHW
        # W = Q^T K: B,HW,HW
        # print(x_key.shape)  #torch.Size([512, 9216])
        x_w1 = torch.matmul(x_query, x_key) * self.scale # BHW, BHW
        x_w = x_w1.view(B * H5 * W5, B, H5 * W5)
        # print('x_w1:{}'.format(x_w.shape)) #[9216, 16, 576]
        x_w = torch.max(x_w, -1).values  # BHW, B
        # print('x_w2:{}'.format(x_w.shape)) # [9216, 16]
        x_w = x_w.mean(-1)
        # print('x_w3:{}'.format(x_w.shape)) #[9216]
        x_w = x_w.view(B, -1)   # B, HW
        # print('x_w4:{}'.format(x_w.shape)) #[16, 576]
        x_w = F.softmax(x_w, dim=-1)  # B, HW
        # print('x_w5:{}'.format(x_w.shape)) #[16, 576]
        #####  mine ######
        # x_w_max = torch.max(x_w, -1)
        # max_indices0 = x_w_max.indices.unsqueeze(-1).unsqueeze(-1)
        norm0 = F.normalize(x5, dim=1)
        # norm = norm0.view(B, C, -1)
        # max_indices = max_indices0.expand(B, C, -1)
        # seeds = torch.gather(norm, 2, max_indices).unsqueeze(-1)
        x_w = x_w.unsqueeze(1)
        # print('x_w6:{}'.format(x_w.shape)) # [16, 1, 576]
        x_w_max = torch.max(x_w, -1).values.unsqueeze(2).expand_as(x_w)
        # print('x_w7:{}'.format(x_w_max.shape)) #[16, 1, 576]
        mask = torch.zeros_like(x_w).cuda()
        # print('mask:{}'.format(mask.shape)) #[16, 1, 576]
        mask[x_w == x_w_max] = 1
        mask = mask.view(B, 1, H5, W5)
        # print('mask1:{}'.format(mask.shape)) #[16, 1, 24, 24]
        seeds = norm0 * mask
        seeds = seeds.sum(3).sum(2).unsqueeze(2).unsqueeze(3)
        cormap = self.correlation(norm0, seeds)
        x51 = x5 * cormap
        proto1 = torch.mean(x51, (0, 2, 3), True)
        # return x5, proto1, x5*proto1+x51#, mask
        return proto1, x5*proto1+x51#, mask


class my_AttLayer(nn.Module):
    def __init__(self, input_channels=512):
        super(my_AttLayer, self).__init__()
        # self.query_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        # self.key_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        # self.scale = 1.0 / (input_channels ** 0.5)
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.primary_capsules = my_CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=512, out_channels=32,
                                             kernel_size=9, stride=1)
        # self.digit_capsules = my_CapsuleLayer(num_capsules=8, num_route_nodes=32 * 14*14, in_channels=8,
                                        #    out_channels=10)
        # self.cap_decoder = nn.Sequential(
            # nn.Linear(8 * 10, 512),
            # nn.ReLU(inplace=True),
            # nn.Linear(512, 1024),
            # nn.ReLU(inplace=True),
            # nn.Linear(1024, 784),
            # nn.Sigmoid())
        self.convcap = nn.Sequential(nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(inplace=True)
        )
        
        
    def correlation(self, x5, seeds):
        B, C, H5, W5 = x5.size()
        if self.training:
            correlation_maps = F.conv2d(x5, weight=seeds)  # B,B,H,W
        else:
            correlation_maps = torch.relu(F.conv2d(x5, weight=seeds))  # B,B,H,W
        correlation_maps = correlation_maps.mean(1).view(B, -1)
        min_value = torch.min(correlation_maps, dim=1, keepdim=True)[0]
        max_value = torch.max(correlation_maps, dim=1, keepdim=True)[0]
        correlation_maps = (correlation_maps - min_value) / (max_value - min_value + 1e-12)  # shape=[B, HW]
        correlation_maps = correlation_maps.view(B, 1, H5, W5)  # shape=[B, 1, H, W]
        return correlation_maps

    def forward(self, x5, y=None):
        # print(x5.shape) #torch.Size([16, 512, 24, 24])
        # x: B,C,H,W
        x5 = self.conv(x5)+x5
        B, C, H5, W5 = x5.size()
        cap, cap1, cap2 = self.primary_capsules(x5)
        # print(cap[0].shape, cap1.shape, cap2.shape)  
        # torch.Size([16, 18432, 1]) torch.Size([16, 18432, 8]) torch.Size([16, 18432, 8])
        # print(len(cap), len(cap1), len(cap2)) 

        # di_cap = self.digit_capsules(cap2).squeeze().transpose(0, 1)
        # # print(di_cap.shape)  # torch.Size([16, 8, 10])
        # # 8是类别，16是batch， 10是输出的通道数
        # classes = (di_cap ** 2).sum(dim=-1) ** 0.5
        # classes = F.softmax(classes, dim=-1)

        cap3 = cap2.view(cap2.size(0), 32, -1, cap2.size(2))
        # # print(cap2.shape) # torch.Size([16, 32, 576, 8])
        cap4 = torch.max(cap3, 1).values
        # # print(cap3.shape)  # torch.Size([16, 576, 8])
        cap5 = F.softmax(cap4, dim=1)
        # print(cap5.shape) #torch.Size([16, 576, 8])
        cap_list = [cap5[:, :, i] for i in range(8)] 

        norm0 = F.normalize(x5, dim=1) # torch.Size([16, 512, 24, 24],

        for i in range(len(cap_list)):
            cap6 = cap_list[i].unsqueeze(1) 
            # print(cap6.shape) #torch.Size([16, 1, 576])
            cap6_max = torch.max(cap6, -1).values.unsqueeze(2).expand_as(cap6)
            # print('aaa',cap6.shape) # [16, 1, 576])
            # print(cap6_max.shape)
            mask = torch.zeros_like(cap6).cuda()
            mask[cap6 == cap6_max] = 1
            mask = mask.view(B, 1, H5, W5) # [16, 1, 24, 24]

            seeds = norm0 * mask
            # print(seeds.shape)  #torch.Size([16, 512, 24, 24])
            seeds = seeds.sum(3).sum(2).unsqueeze(2).unsqueeze(3)
            # print(seeds.shape)  # torch.Size([16, 512, 1, 1])
            cormap = self.correlation(norm0, seeds)
            # print(cormap.shape)  # torch.Size([16, 1, 24, 24])
            x51 = x5 * cormap
            # print(x51.shape)  #torch.Size([16, 512, 24, 24])
            proto1 = torch.mean(x51, (0, 2, 3), True)
            # print(proto1.shape)  # torch.Size([1, 512, 1, 1])
            out = x5*proto1+x51
            out = out.unsqueeze(-1)
            proto1 = proto1.unsqueeze(-1)
            if i == 0:
                out_map = out
                out_pro = proto1
            else:
                out_map = torch.cat([out_map,out],-1)
                out_pro = torch.cat([out_pro, proto1], -1)
        out_map = torch.mean(out_map,dim=-1)
        out_pro = torch.mean(out_pro,dim=-1)
        out_map = F.normalize(out_map)
        out_pro = F.normalize(out_pro)

        return out_pro, out_map#, reconstructions


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.toplayer = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.latlayer4 = LatLayer(in_channel=512)
        self.latlayer3 = LatLayer(in_channel=256)
        self.latlayer2 = LatLayer(in_channel=128)
        self.latlayer1 = LatLayer(in_channel=64)

        self.enlayer4 = EnLayer()
        self.enlayer3 = EnLayer()
        self.enlayer2 = EnLayer()
        self.enlayer1 = EnLayer()

        self.dslayer4 = DSLayer()
        self.dslayer3 = DSLayer()
        self.dslayer2 = DSLayer()
        self.dslayer1 = DSLayer()

    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x + y

    def forward(self, weighted_x5, x4, x3, x2, x1, H, W):
        preds = []
        p5 = self.toplayer(weighted_x5)
        p4 = self._upsample_add(p5, self.latlayer4(x4))
        p4 = self.enlayer4(p4)
        _pred = self.dslayer4(p4)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))

        p3 = self._upsample_add(p4, self.latlayer3(x3))
        p3 = self.enlayer3(p3)
        _pred = self.dslayer3(p3)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))

        p2 = self._upsample_add(p3, self.latlayer2(x2))
        p2 = self.enlayer2(p2)
        _pred = self.dslayer2(p2)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))

        p1 = self._upsample_add(p2, self.latlayer1(x1))
        p1 = self.enlayer1(p1)
        _pred = self.dslayer1(p1)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))
        return preds


class DCFMNet1(nn.Module):
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, mode='train'):
        super(DCFMNet1, self).__init__()
        self.gradients = None
        self.backbone = VGG_Backbone()
        self.mode = mode
        # self.aug = my_AugAttentionModule()
        self.middle_decoder = my_middle_decoder()
        # self.fusion = AttLayer(512)
        self.fusion = my_AttLayer(512)
        self.decoder = Decoder()

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x, gt):
        if self.mode == 'train':
            preds = self._forward(x, gt)
        else:
            with torch.no_grad():
                preds = self._forward(x, gt)

        return preds

    def featextract(self, x):
        x1 = self.backbone.conv1(x)
        x2 = self.backbone.conv2(x1)
        x3 = self.backbone.conv3(x2)
        x4 = self.backbone.conv4(x3)
        x5 = self.backbone.conv5(x4)
        return x5, x4, x3, x2, x1

    def _forward(self, x, gt):
        [B, _, H, W] = x.size()
        x5, x4, x3, x2, x1 = self.featextract(x) 
        # torch.Size([16, 512, 14, 14],[16, 512, 28, 28],[16, 256, 56, 56],[16, 128, 112, 112],[16, 64, 224, 224],[16, 3, 224, 224])
        # torch.Size([16, 512, 24, 24],[16, 512, 48, 48],[16, 256, 96, 96],[16, 128, 192, 192],[16, 64, 384, 384],[16, 3, 384, 384])
        # print(x5.shape, x4.shape, x3.shape, x2.shape, x1.shape, x.shape)
        # feat, proto, weighted_x5, cormap = self.fusion(x5)
        #  torch.Size([16, 512, 14, 14],[1, 512, 1, 1],[16, 512, 14, 14],[16, 1, 14, 14])
        # print(feat.shape, proto.shape, weighted_x5.shape, cormap.shape)
        # feataug = self.aug(weighted_x5)
        #  torch.Size([16, 512, 14, 14])
        # print(feataug.shape)

        # ***************************************************
        out1, out2 = self.fusion(x5)
        
        # feataug = self.aug(out2)
        feataug, x4, x3, x2, x1 = self.middle_decoder(out2, x4, x3, x2, x1)
        # print(feataug.shape, x4.shape, x3.shape, x2.shape, x1.shape)
        # *********************************************************

        preds = self.decoder(out2, x4, x3, x2, x1, H, W)

        if self.training:
            gt = gtg(x, gt)
            gt = F.interpolate(gt, size=x5.size()[2:], mode='bilinear', align_corners=False)
            # mask11 = torch.zeros_like(gt).cuda()
            # gt[gt == 0] = 0.01
            # gt[gt == 1] = 0.99
            # gt = torch.where(gt == 0, torch.tensor(0.01, dtype=torch.float32), torch.tensor(0.9, dtype=torch.float32))  
            proto_pos, weighted_x5_pos = self.fusion(x5 * gt)
            proto_neg, weighted_x5_neg = self.fusion(x5*(1-gt))
            return x1, preds, out1, proto_pos, proto_neg#, cls, cls_pos, cls_neg
        return preds

class DCFMNet(nn.Module):
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, mode='train'):
        super(DCFMNet, self).__init__()
        self.gradients = None
        self.backbone = VGG_Backbone()
        self.mode = mode
        self.aug = AugAttentionModule()
        self.fusion = AttLayer(512)
        self.decoder = Decoder()

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x, gt):
        if self.mode == 'train':
            preds = self._forward(x, gt)
        else:
            with torch.no_grad():
                preds = self._forward(x, gt)

        return preds

    def featextract(self, x):
        x1 = self.backbone.conv1(x)
        x2 = self.backbone.conv2(x1)
        x3 = self.backbone.conv3(x2)
        x4 = self.backbone.conv4(x3)
        x5 = self.backbone.conv5(x4)
        return x5, x4, x3, x2, x1

    def _forward(self, x, gt):
        [B, _, H, W] = x.size()
        x5, x4, x3, x2, x1 = self.featextract(x)
        feat, proto, weighted_x5, cormap = self.fusion(x5)
        feataug = self.aug(weighted_x5)
        preds = self.decoder(feataug, x4, x3, x2, x1, H, W)
        if self.training:
            gt = F.interpolate(gt, size=weighted_x5.size()[2:], mode='bilinear', align_corners=False)
            feat_pos, proto_pos, weighted_x5_pos, cormap_pos = self.fusion(x5 * gt)
            feat_neg, proto_neg, weighted_x5_neg, cormap_neg = self.fusion(x5*(1-gt))
            return preds, proto, proto_pos, proto_neg
        return preds


def gtg(x, gt):
    rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140]).to(torch.device("cuda")) .view(1, 3, 1, 1)
    grayscale = (x * rgb_weights).sum(dim=1, keepdim=True)
    # x_ = torch.matmul(x, torch.tensor([0.299,0.587,0.114]).view(1, 3, 1, 1))
    out = torch.where(gt==0,0.01,grayscale)
    # print(gt.shape)
    return out

class DCFM(nn.Module):
    def __init__(self, mode='train'):
        super(DCFM, self).__init__()
        set_seed(123)
        # self.dcfmnetmy = my_DCFMNet()
        # self.dcfmnet1 = DCFMNet()
        self.dcfmnet = DCFMNet()
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode
        self.dcfmnet.set_mode(self.mode)

    def forward(self, x, gt):
        ########## Co-SOD ############
        preds = self.dcfmnet(x, gt)
        return preds


class my_DCFMNet2(nn.Module):
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, mode='train'):
        super(my_DCFMNet2, self).__init__()
        self.gradients = None
        self.backbone = VGG_Backbone()
        self.mode = mode
        # self.aug = my_AugAttentionModule()
        self.middle_decoder = my_middle_decoder()
        self.fusion = AttLayer(512)
        # self.fusion = my_AttLayer(512)
        self.decoder = Decoder()
        self.conv = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Softmax()
        )

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x, gt):
        if self.mode == 'train':
            preds = self._forward(x, gt)
        else:
            with torch.no_grad():
                preds = self._forward(x, gt)

        return preds

    def featextract(self, x):
        x1 = self.backbone.conv1(x)
        x2 = self.backbone.conv2(x1)
        x3 = self.backbone.conv3(x2)
        x4 = self.backbone.conv4(x3)
        x5 = self.backbone.conv5(x4)
        return x5, x4, x3, x2, x1

    def _forward(self, x, gt):
        [B, _, H, W] = x.size()
        x5, x4, x3, x2, x1 = self.featextract(x) 
        # torch.Size([16, 512, 14, 14],[16, 512, 28, 28],[16, 256, 56, 56],[16, 128, 112, 112],[16, 64, 224, 224],[16, 3, 224, 224])
        # torch.Size([16, 512, 24, 24],[16, 512, 48, 48],[16, 256, 96, 96],[16, 128, 192, 192],[16, 64, 384, 384],[16, 3, 384, 384])
        # print(x5.shape, x4.shape, x3.shape, x2.shape, x1.shape, x.shape)
        # feat, proto, weighted_x5, cormap = self.fusion(x5)
        #  torch.Size([16, 512, 14, 14],[1, 512, 1, 1],[16, 512, 14, 14],[16, 1, 14, 14])
        # print(feat.shape, proto.shape, weighted_x5.shape, cormap.shape)
        # feataug = self.aug(weighted_x5)
        #  torch.Size([16, 512, 14, 14])
        # print(feataug.shape)

        # ***************************************************
        out1, out2 = self.fusion(x5)
        
        # feataug = self.aug(out2)
        feataug, x4, x3, x2, x1 = self.middle_decoder(out2, x4, x3, x2, x1)
        # print(feataug.shape, x4.shape, x3.shape, x2.shape, x1.shape)
        # *********************************************************

        preds = self.decoder(feataug, x4, x3, x2, x1, H, W)
        # print(len(preds), preds[0].shape)
        x1 = self.conv(x1)
        if self.training:
            gt = gtg(x, gt)
            gt = F.interpolate(gt, size=x5.size()[2:], mode='bilinear', align_corners=False)
            # mask11 = torch.zeros_like(gt).cuda()
            # gt[gt == 0] = 0.01
            # gt[gt == 1] = 0.99
            # gt = torch.where(gt == 0, torch.tensor(0.01, dtype=torch.float32), torch.tensor(0.9, dtype=torch.float32))  
            proto_pos, weighted_x5_pos = self.fusion(x5 * gt)
            proto_neg, weighted_x5_neg = self.fusion(x5*(1-gt))
            return x1, preds, out1, proto_pos, proto_neg#, cls, cls_pos, cls_neg
        return preds