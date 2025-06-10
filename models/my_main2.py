import torch
from torch import nn
import torch.nn.functional as F
from models.vgg import VGG_Backbone
from util import *
# from models.sct import *
from models.ssm import *
from models.my_module import *


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
        # x: B,C,H,W
        x5 = self.conv(x5)+x5
        B, C, H5, W5 = x5.size()
        x_query = self.query_transform(x5).view(B, C, -1)
        # x_query: B,HW,C
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C)  # BHW, C
        # x_key: B,C,HW
        x_key = self.key_transform(x5).view(B, C, -1)
        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1)  # C, BHW
        # W = Q^T K: B,HW,HW
        x_w1 = torch.matmul(x_query, x_key) * self.scale # BHW, BHW
        x_w = x_w1.view(B * H5 * W5, B, H5 * W5)
        x_w = torch.max(x_w, -1).values  # BHW, B
        x_w = x_w.mean(-1)
        x_w = x_w.view(B, -1)   # B, HW
        x_w = F.softmax(x_w, dim=-1)  # B, HW
        #####  mine ######
        # x_w_max = torch.max(x_w, -1)
        # max_indices0 = x_w_max.indices.unsqueeze(-1).unsqueeze(-1)
        norm0 = F.normalize(x5, dim=1)
        # norm = norm0.view(B, C, -1)
        # max_indices = max_indices0.expand(B, C, -1)
        # seeds = torch.gather(norm, 2, max_indices).unsqueeze(-1)
        x_w = x_w.unsqueeze(1)
        x_w_max = torch.max(x_w, -1).values.unsqueeze(2).expand_as(x_w)
        mask = torch.zeros_like(x_w).cuda()
        mask[x_w == x_w_max] = 1
        mask = mask.view(B, 1, H5, W5)
        seeds = norm0 * mask
        seeds = seeds.sum(3).sum(2).unsqueeze(2).unsqueeze(3)
        cormap = self.correlation(norm0, seeds)
        x51 = x5 * cormap
        proto1 = torch.mean(x51, (0, 2, 3), True)
        return x5, proto1, x5*proto1+x51, mask


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


class DCFMNet(nn.Module):
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, mode='train'):
        super(DCFMNet, self).__init__()
        self.gradients = None
        self.backbone = VGG_Backbone()
        self.mode = mode
        # self.aug = my_AugAttentionModule()
        self.my_aug = NonLocalBlock(512)
        self.fusion = AttLayer3(512)
        self.middle = my_middle_decoder2()
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
        # print(x5.shape, x4.shape, x3.shape, x2.shape, x1.shape, x.shape)
        # feat, proto, weighted_x5, cormap = self.fusion(x5)
        feat, proto, weighted_x5, cormap = self.fusion(x5)
        
        x44, x33, x22, x11 = self.middle(x4, x3, x2, x1) 

        feataug = self.my_aug(weighted_x5)
        preds = self.decoder(feataug, x44, x33, x22, x11, H, W)
        # print(len(preds), preds[0].shape, preds[1].shape, preds[2].shape, preds[3].shape)
        if self.training:
            gt = gtg(x, gt)
            gt = F.interpolate(gt, size=weighted_x5.size()[2:], mode='bilinear', align_corners=False)
            feat_pos, proto_pos, weighted_x5_pos, cormap_pos = self.fusion(x5 * gt)
            feat_neg, proto_neg, weighted_x5_neg, cormap_neg = self.fusion(x5*(1-gt))
            return preds, proto, proto_pos, proto_neg
        return preds


class DCFM(nn.Module):
    def __init__(self, mode='train'):
        super(DCFM, self).__init__()
        set_seed(123)
        self.dcfmnet = DCFMNet()
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode
        self.dcfmnet.set_mode(self.mode)

    def forward(self, x, gt):
        ########## Co-SOD ############
        preds = self.dcfmnet(x, gt)
        return preds


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
        self.ssm = VSSBlock(
                hidden_dim=512,
                norm_layer=nn.LayerNorm,
            )

    def forward(self, x):
        B, C, H, W = x.size()
        x = self.conv(x)

        x0 = x.permute(0,2,3,1)
        x1 = self.ssm(x0).permute(0, 3, 1, 2)
        x2 = self.ssm(x0).permute(0, 3, 1, 2)

        x_query = self.query_transform(x1).view(B, C, -1).permute(0, 2, 1)  # B,HW,C
        # x_key: C,BHW
        x_key = self.key_transform(x1).view(B, C, -1)  # B, C,HW
        # x_value: BHW, C
        x_value = self.value_transform(x2).view(B, C, -1).permute(0, 2, 1)  # B,HW,C
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

        # ***********************this is mine*****************************************
        out = out.view(B, H, W, C)
        # print(out.shape)  #  torch.Size([16, 512, 24, 24])
        out = self.ssm(out)
        # print(out.shape)
        out = out.permute(0, 3, 1, 2)
        # ****************************************************************************
        # out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # 这个是原版的
        # print(out.shape)

        return out+x


class my_middle_decoder2(nn.Module):
    def __init__(self, 
                 input_channels=512, 
                 dim=[512, 256, 128, 64],
                 heads = [1,2,4,8],
                 ffn_expansion_factor = 2.66,
                 bias = False,
                 LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
                 dual_pixel_task = False 
                 ):
        super(my_middle_decoder2, self).__init__()

        self.do1 = dsfa(64, 128, 128, 0.25)
        self.do2 = dsfa(128, 256, 256, 0.25)
        self.do3 = dsfa(256, 512, 512, 0.25)
        

        # self.nnb1 = NonLocalBlock(512)
        # self.nnb2 = NonLocalBlock(256)
        # self.nnb3 = NonLocalBlock(128)
        # self.nnb4 = NonLocalBlock(64)
        
        # self.refinement = my_TransformerBlock(dim=int(dim[2]), num_heads=heads[0])
        
    def forward(self, x44, x33, x22, x11):
        # B, C, H, W = x5.size()
        # print(x5.shape, x44.shape, x33.shape, x22.shape, x11.shape)
        #[16, 512, 24, 24]，[16, 512, 48, 48]，[16, 256, 96, 96]，[16, 128, 192, 192]，[16, 64, 384, 384])
        # x.permute(0, 2, 3, 1)
        # x11 = self.nnb4(x11)
        x22 = self.do1(x11, x22)
        # print(x22.shape)
        x33 = self.do2(x22, x33)
        # print(x33.shape)
        x44 = self.do3(x33, x44)
        # print(x44.shape)
        # print(x44.shape, x33.shape, x22.shape, x11.shape)
        return x44, x33, x22, x11

class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), 
                    nn.BatchNorm2d(features),
                    nn.ReLU()
                            ))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            # print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            # print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

class dsfa(nn.Module):
    def __init__(self, ch_in, ch_in1, ch_ou, rate):
        super(dsfa, self).__init__()

        self.big_conv           = nn.Sequential(nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ch_in), nn.PReLU())
        self.small_conv         = nn.Sequential(nn.Conv2d(ch_in1, ch_in, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ch_in), nn.PReLU()) 

        self.big_do             = nn.Sequential(nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=2, padding=1),
                                                nn.Conv2d(ch_in, int(rate*ch_in), kernel_size=3, stride=1, padding=1), 
                                                nn.BatchNorm2d(int(rate*ch_in)), nn.Sigmoid())
        self.small_up           = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                                nn.Conv2d(ch_in, int(rate*ch_in), kernel_size=3, stride=1, padding=1),
                                                nn.BatchNorm2d(int(rate*ch_in)), nn.Sigmoid())

        self.big_conv1          = nn.Sequential(nn.Conv2d(ch_in, int(rate*ch_in), kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(int(rate*ch_in)), nn.PReLU()) 
        self.small_conv1        = nn.Sequential(nn.Conv2d(ch_in, int(rate*ch_in), kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(int(rate*ch_in)), nn.PReLU()) 
        
        self.big_conv2          = nn.Sequential(nn.Conv2d(int(rate*ch_in), int(rate*ch_in), kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(int(rate*ch_in)), nn.PReLU()) 
        self.small_conv2        = nn.Sequential(nn.Conv2d(int(rate*ch_in), int(rate*ch_in), kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(int(rate*ch_in)), nn.PReLU()) 
        
        self.big_conv3          = nn.Sequential(nn.Conv2d(int(rate*ch_in), int(rate*ch_in), kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(int(rate*ch_in)), nn.PReLU()) 
        self.big_conv4          = nn.Sequential(nn.Conv2d(int(2*rate*ch_in), ch_ou, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(int(ch_ou)), nn.Sigmoid()) 


    def forward(self, f_big, f_small):
        f_big_conv            = self.big_conv(f_big)
        f_small_conv          = self.small_conv(f_small)

        f_small_up            = self.small_up(f_small_conv)
        f_big_down            = self.big_do(f_big_conv)
        # print(f_small_up.shape, f_big_down.shape)

        f_big_conv1           = self.big_conv1(f_big_conv)
        f_small_conv1         = self.small_conv1(f_small_conv)
        # print(f_small_conv1.shape, f_big_conv1.shape)

        f_cat_big             = f_small_up + f_big_conv1
        f_cat_small           = f_small_conv1 + f_big_down
        # print(f_cat_big.shape, f_cat_small.shape)

        out1                  = self.big_conv2(f_cat_big)
        out2                  = self.small_conv2(f_cat_small)

        out3                  = self.big_conv3(out1)
        out4                  = torch.cat([out3, out2], dim=1)
        # print(out4.shape)
        out                   = self.big_conv4(out4)
        # print(out.shape)
        return out

class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

class AttLayer2(nn.Module):
    def __init__(self, input_channels=512):
        super(AttLayer2, self).__init__()
        self.query_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.key_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.scale = 1.0 / (input_channels ** 0.5)
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        # self.primary_capsules = my_CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=512, out_channels=32,
        #                                      kernel_size=9, stride=1)
        # self.digit_capsules = my_CapsuleLayer(num_capsules=8, num_route_nodes=32 * 14*14, in_channels=8,
        #                                    out_channels=8)

        self.ss2d = my_SS2D(d_model=512, d_state=16)
        
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
        # B, C, H5, W5 = x5.size()
        # cap0, cap1, cap2 = self.primary_capsules(x5)
        # cap00, cap11, cap22 = self.primary_capsules(x5)
        # # print(cap2.shape)
        

        # cap3 = cap2.view(cap2.size(0), 32, -1, cap2.size(2))
        # cap33 = cap22.view(cap22.size(0), 32, -1, cap2.size(2))
        # cap4 = torch.max(cap3, 1).values
        # cap44 = torch.max(cap33, 1).values
        # # print(cap3.shape)  # torch.Size([16, 576, 8])

        # cap5 = F.softmax(cap4, dim=1)
        # cap55 = F.softmax(cap44, dim=1)
        # # print(cap5.shape)

        # cap66 =  torch.matmul(cap55, cap5.permute(0,2,1))
        # # print(cap66.shape)  #torch.Size([16, 576, 576])
        # cap7 = cap66.mean(-1)
        # x_w = F.softmax(cap7, dim=-1)
        # # print(cap7.shape)  #torch.Size([16, 576])

        # di_cap = self.digit_capsules(cap2).squeeze().transpose(0, 1)
        # print(di_cap.shape)  # torch.Size([16, 8, 10])
        # # # 8是类别，16是batch， 10是输出的通道数
        # # classes = (di_cap ** 2).sum(dim=-1) ** 0.5
        # # classes = F.softmax(classes, dim=-1)

        B, C, H5, W5 = x5.size()
        x5 = self.conv(x5)+x5

        x00 = x5.permute(0, 2, 3, 1)
        x00 = self.ss2d(x00)
        x00 = x00.permute(0, 3, 1, 2)

        # B, C, H5, W5 = x5.size()
        x_query = self.query_transform(x00).view(B, C, -1)
        # print(x_query.shape) #torch.Size([16, 512, 576])
        # x_query: B,HW,C
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C)  # BHW, C
        # x_key: B,C,HW
        # print(x_query.shape)  #torch.Size([9216, 512])
        x_key = self.key_transform(x00).view(B, C, -1)
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
        return x5, proto1, x5*proto1+x51, mask
        # return proto1, x5*proto1+x51#, mask


class AttLayer3(nn.Module):
    def __init__(self, input_channels=512):
        super(AttLayer3, self).__init__()
        self.query_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.key_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.scale = 1.0 / (input_channels ** 0.5)
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        # self.primary_capsules = my_CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=512, out_channels=32,
        #                                      kernel_size=9, stride=1)
        # self.digit_capsules = my_CapsuleLayer(num_capsules=8, num_route_nodes=32 * 14*14, in_channels=8,
        #                                    out_channels=8)

        self.ss2d = my_SS2D(d_model=512, d_state=16)
        self.ssm = VSSBlock(
                            hidden_dim=512,
                            norm_layer=nn.LayerNorm,
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

    def forward(self, x5):

        B, C, H5, W5 = x5.size()
        x5 = self.conv(x5)+x5

        x0 = x5.permute(0, 2, 3, 1).contiguous()
        x00 = self.ssm(x0)
        xquery = x00.permute(0, 3, 1, 2).contiguous().view(B, C, -1)
        x_query = torch.transpose(xquery, 1, 2).contiguous().view(-1, C)
        
        x11 = self.ssm(x0)
        xkey = x11.permute(0, 3, 1, 2).contiguous().view(B, C, -1)
        x_key = torch.transpose(xkey, 0, 1).contiguous().view(C, -1)

        # # B, C, H5, W5 = x5.size()
        # x_query = self.query_transform(x00).view(B, C, -1)
        # # print(x_query.shape) #torch.Size([16, 512, 576])
        # # x_query: B,HW,C
        # x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C)  # BHW, C
        # # x_key: B,C,HW
        # # print(x_query.shape)  #torch.Size([9216, 512])
        # x_key = self.key_transform(x00).view(B, C, -1)
        # x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1)  # C, BHW
        # # W = Q^T K: B,HW,HW
        # # print(x_key.shape)  #torch.Size([512, 9216])

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
        return x5, proto1, x5*proto1+x51, mask
        # return proto1, x5*proto1+x51#, mask

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
                                    #  nn.ReLU(inplace=True),
                                    nn.ReLU(),
                                     nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
                                    #  nn.ReLU(inplace=True)
                                    nn.ReLU()
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

def gtg(x, gt):
    rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140]).to(torch.device("cuda")) .view(1, 3, 1, 1)
    grayscale = (x * rgb_weights).sum(dim=1, keepdim=True)
    # x_ = torch.matmul(x, torch.tensor([0.299,0.587,0.114]).view(1, 3, 1, 1))
    aaa = torch.ones_like(grayscale)
    bbb = 0.01*aaa
    out = torch.where(gt==0,bbb,grayscale)
    # print(gt.shape)
    return out
