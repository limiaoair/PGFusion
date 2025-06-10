import torch
import torch.nn as nn
import torch.optim as optim
from util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse
from tqdm import tqdm
from dataset import get_loader
from loss import *
from config import Config
from evaluation.dataloader import EvalDataset
from evaluation.evaluator import Eval_thread

from models.my_main2 import *

import torch.nn.functional as F
import pytorch_toolbelt.losses as PTL

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Parameter from command line
parser = argparse.ArgumentParser(description='')

parser.add_argument('--loss',
                    default='Scale_IoU',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--bs', '--batch_size', default=1, type=int)
parser.add_argument('--lr',
                    '--learning_rate',
                    default=1e-4,
                    type=float,
                    help='Initial learning rate')
parser.add_argument('--resume',
                    # default='project/co-sod/DCFM-master/temp/0828/checkpoint.pth',
                    default=None,
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset',
                    default='CoCo',
                    type=str,
                    help="Options: 'CoCo'")
parser.add_argument('--testsets',
                    default='CoCA+CoSOD3k+CoSal2015',
                    type=str,
                    help="Options: 'CoCA','CoSal2015','CoSOD3k','iCoseg','MSRC'")
parser.add_argument('--size',
                    default=384,
                    type=int,
                    help='input size')
parser.add_argument('--tmp', default='./temp/0907', help='Temporary folder')
parser.add_argument('--save_root', default='./CoSODmaps/0907', type=str, help='Output folder')
parser.add_argument('--save_best_root', default='./results/0907/', type=str, help='Output best folder')
args = parser.parse_args()
config = Config()


if args.trainset == 'CoCo':
    # Prepare dataset 是coco-seg加上DUT-class的全部
    train_img_path = '/home/lm/data_for_lm/dataset/DUTS_class_seg/im/'
    train_gt_path = '/home/lm/data_for_lm/dataset/DUTS_class_seg/gt_png/'

    # ## Prepare dataset 是coco-seg加上DUT-class的全部
    # train_img_path = '/home/lm/data_for_lm/dataset/DUTS_class/im/'
    # train_gt_path = '/home/lm/data_for_lm/dataset/DUTS_class/gt/'

    # 原版的数据
    # train_img_path = './data/CoCo/img/'
    # train_gt_path = './data/CoCo/gt/'
    train_loader = get_loader(train_img_path,
                              train_gt_path,
                              args.size,
                              args.bs,
                              max_num=16, #20,
                              istrain=True,
                              shuffle=False,
                              num_workers=8, #4,
                              pin=True)

else:
    print('Unkonwn train dataset')
    print(args.dataset)

# for testset in ['CoCA']:
#     if testset == 'CoCA':
#         test_img_path = '/home/lm/data_for_lm/dataset/CoSalBenchmark/CoCA/image/'
#         test_gt_path = '/home/lm/data_for_lm/dataset/CoSalBenchmark/CoCA/binary/'
#         saved_root = os.path.join(args.save_root, 'CoCA')
#     elif testset == 'CoSOD3k':
#         test_img_path = '/home/lm/data_for_lm/dataset/CoSalBenchmark/CoSOD3K/Image/'
#         # test_gt_path = './data/gts/CoSOD3k/'
#         test_gt_path = '/home/lm/data_for_lm/dataset/CoSalBenchmark/CoSOD3K/GroundTruth/'
#         saved_root = os.path.join(args.save_root, 'CoSOD3k')
#     elif testset == 'CoSal2015':
#         test_img_path = '/home/lm/data_for_lm/dataset/CoSalBenchmark/CoSal2015/Image/'
#         test_gt_path = '/home/lm/data_for_lm/dataset/CoSalBenchmark/CoSal2015/GroundTruth/'
#         saved_root = os.path.join(args.save_root, 'CoSal2015')
#     # elif testset == 'CoCo':
#     #     test_img_path = './data/images/CoCo/'
#     #     test_gt_path = './data/gts/CoCo/'
#     #     saved_root = os.path.join(args.save_root, 'CoCo')
#     else:
#         print('Unkonwn test dataset')
#         print(args.dataset)
    
for testset in ['CoCA']:
    if testset == 'CoCA':
        test_img_path = './data/images/CoCA/'
        test_gt_path = './data/gts/CoCA/'
        saved_root = os.path.join(args.save_root, 'CoCA')
    elif testset == 'CoSOD3k':
        test_img_path = './data/images/CoSOD3k/'
        test_gt_path = './data/gts/CoSOD3k/'
        saved_root = os.path.join(args.save_root, 'CoSOD3k')
    elif testset == 'CoSal2015':
        test_img_path = './data/images/CoSal2015/'
        test_gt_path = './data/gts/CoSal2015/'
        saved_root = os.path.join(args.save_root, 'CoSal2015')
    # elif testset == 'CoCo':
    #     test_img_path = './data/images/CoCo/'
    #     test_gt_path = './data/gts/CoCo/'
    #     saved_root = os.path.join(args.save_root, 'CoCo')
    else:
        print('Unkonwn test dataset')
        print(args.dataset)

    # test_loader = get_loader(
    #     test_img_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True)

test_loaders = {}
for testset in args.testsets.split('+'):
    test_loader = get_loader(
        os.path.join('./data', 'images', testset), os.path.join('./data', 'gts', testset),
        args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True
    )
    test_loaders[testset] = test_loader


# make dir for tmp
os.makedirs(args.tmp, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.tmp, "log.txt"))
set_seed(123)

# Init model
device = torch.device("cuda")

model = DCFM()
model = model.to(device)
model.apply(weights_init)

model.dcfmnet.backbone._initialize_weights(torch.load('./models/vgg16-397923af.pth'))

backbone_params = list(map(id, model.dcfmnet.backbone.parameters()))
base_params = filter(lambda p: id(p) not in backbone_params,
                     model.dcfmnet.parameters())

all_params = [{'params': base_params}, {'params': model.dcfmnet.backbone.parameters(), 'lr': args.lr*0.1}]

# Setting optimizer
optimizer = optim.Adam(params=all_params,lr=args.lr, weight_decay=1e-4, betas=[0.9, 0.99])

for key, value in model.named_parameters():
    if 'dcfmnet.backbone' in key and 'dcfmnet.backbone.conv5.conv5_3' not in key:
        value.requires_grad = False

for key, value in model.named_parameters():
    print(key,  value.requires_grad)

# log model and optimizer pars
logger.info("Model details:")
logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)
logger.info("Scheduler details:")
# logger.info(scheduler)
logger.info("Other hyperparameters:")
logger.info(args)

# Setting Loss
exec('from loss import ' + args.loss)
IOUloss = eval(args.loss+'()')

# def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=60):
#     decay = decay_rate ** (epoch // decay_epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = decay*init_lr
#         lr=param_group['lr']
#     return lr

def main():
    val_measures = []
    # Optionally resume from a checkpoint
    if args.resume:  # 断点续训
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.dcfmnet.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    print('训练的总轮数：{}'.format(args.epochs))
    
    for epoch in range(args.start_epoch, args.epochs):
        # cur_lr = adjust_lr(optimizer, args.lr, epoch)
        train_loss = train(epoch)  # 得到损失值

        if config.validation:
            measures = validate(model, test_loaders, args.testsets) # 生成每一轮次的预测图，把s指标导出

            val_measures.append(measures)  # 追加到新创建的列表里

            print(
                'Validation: S_measure on CoCA for epoch-{} is {:.4f}. Best epoch is epoch-{} with S_measure {:.4f}'.format(
                    epoch, measures[0], np.argmax(np.array(val_measures)[:, 0].squeeze()),
                    np.max(np.array(val_measures)[:, 0]))
            )
            # Save checkpoint
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.dcfmnet.state_dict(),
                #'scheduler': scheduler.state_dict(),
            },
            path=args.tmp)
        if config.validation:
            if np.max(np.array(val_measures)[:, 0].squeeze()) == measures[0]:
                # best_weights_before = [os.path.join(args.tmp, weight_file) for weight_file in
                #                        os.listdir(args.tmp) if 'best_' in weight_file]
                # for best_weight_before in best_weights_before:
                #     os.remove(best_weight_before)
                torch.save(model.dcfmnet.state_dict(),
                           os.path.join(args.tmp, 'best_ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0])))
                num = test(model, test_loaders, args.testsets)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            torch.save(model.dcfmnet.state_dict(), args.tmp + '/model-' + str(epoch + 1) + '.pt')
       
        # if epoch > 188:
        #     torch.save(model.dcfmnet.state_dict(), args.tmp+'/model-' + str(epoch + 1) + '.pt')
    #dcfmnet_dict = model.dcfmnet.state_dict()
    #torch.save(dcfmnet_dict, os.path.join(args.tmp, 'final.pth'))

def sclloss(x, xt, xb):
    cosc = (1+compute_cos_dis(x, xt))*0.5
    cosb = (1+compute_cos_dis(x, xb))*0.5
    loss = -torch.log(cosc+1e-5)-torch.log(1-cosb+1e-5)
    return loss.sum()

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

class my_CapsuleLoss(nn.Module):
    def __init__(self):
        super(my_CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, classes, labels):
        loss = self.reconstruction_loss(classes,labels)
        return loss
        # assert torch.numel(images) == torch.numel(reconstructions)
        # images = images.view(reconstructions.size()[0], -1)
        # reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        # reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        # return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)

def clsloss(classes, labels):
    # Loss = nn.SmoothL1Loss(reduce=True, size_average=True)
    Loss = torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)
    loss = Loss(classes, labels)
    # loss = smooth_l1_loss(classes, labels)
    return loss
  
def smooth_l1_loss(input, target, beta=1. / 9, reduction = 'none'):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    ret = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

def train(epoch):
    # Switch to train mode
    model.train()
    model.set_mode('train')
    loss_sum = 0.0
    loss_sumkl = 0.0
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch[0].to(device).squeeze(0)
        # print(inputs.shape)
        gts = batch[1].to(device).squeeze(0)
        pred, proto, protogt, protobg = model(inputs, gts)

        loss_iou = IOUloss(pred, gts)
        # loss_x1 = structure_loss(x1, gts)
        loss_scl = sclloss(proto, protogt, protobg)
        # print(cls.shape, clsgt.shape, clsgt.shape)
        #  # torch.Size([8, 16, 1, 1],[8, 16, 1, 1],[8, 16, 1, 1])
        # loss_cls = clsloss(clss, clsgt)
        # print(loss_cls.shape)
        loss = loss_iou+0.1*loss_scl#+0.1*loss_x1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum = loss_sum + loss_iou.detach().item()

        if batch_idx % 100 == 0:
            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}]  '
                        'Train Loss: loss_iou: {4:.3f}, loss_scl: {5:.3f} '.format(
                            epoch,
                            args.epochs,
                            batch_idx,
                            len(train_loader),
                            loss_iou,
                            loss_scl#,
                            #loss_cls
                        ))
    loss_mean = loss_sum / len(train_loader)
    return loss_mean


def test(model, test_loaders, testsets):
    model.eval()
    testsets = testsets.split('+')
    for testset in testsets:
        print('prediction {} images...'.format(testset))
        test_loader = test_loaders[testset]
        saved_best_root = os.path.join(args.save_best_root, testset)
        for batch in test_loader:
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            with torch.no_grad():
                scaled_preds = model(inputs, gts)[-1].sigmoid()

            os.makedirs(os.path.join(saved_best_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)
            # print('模型输出的个数 {}...'.format(num))
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                    align_corners=True)
                save_tensor_img(res, os.path.join(saved_best_root, subpath)[:-4] + '.png')
    return num

def validate(model, test_loaders, testsets):
    model.eval()

    testsets = testsets.split('+')
    # print("测试的{}".format(testsets))
    measures = []
    for testset in testsets[:1]:
        print('Validating {}...'.format(testset))
        test_loader = test_loaders[testset]
        # print("测试的{}".format(testset))
        saved_root = os.path.join(args.save_root, testset)
        # print(len(test_loader))

        for batch in test_loader:
            # print(batch[2])
            inputs = batch[0].to(device).squeeze(0)
            # print(inputs.shape)
            gts = batch[1].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            with torch.no_grad():
                scaled_preds = model(inputs, gts)[-1].sigmoid()

            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)
            # print('模型输出的个数 {}...'.format(num))
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                    align_corners=True)
                # subpath = x[:-4]+'.png'
                save_tensor_img(res, os.path.join(saved_root, subpath)[:-4] + '.png')

        # eval_loader = EvalDataset(
        #     saved_root,  # preds
        #     os.path.join('./data/gts', testset)  # GT
        # )
        # print(saved_root)
        eval_loader = EvalDataset(
            saved_root,  
            # os.path.join(saved_root, testset),# preds
            os.path.join('./data/gts/', testset)  # GT
        )
        # print(len(eval_loader))
        evaler = Eval_thread(eval_loader, cuda=True)
        # Use S_measure for validation
        s_measure = evaler.Eval_Smeasure()
        if s_measure > config.val_measures['Smeasure']['CoCA'] and 0:
            # TODO: evluate others measures if s_measure is very high.
            e_max = evaler.Eval_Emeasure().max().item()
            f_max = evaler.Eval_fmeasure().max().item()
            print('Emax: {:4.f}, Fmax: {:4.f}'.format(e_max, f_max))
        measures.append(s_measure)

    model.train()
    return measures

if __name__ == '__main__':
    main()
