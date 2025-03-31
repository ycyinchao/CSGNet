import torch
import cv2
import torch.nn.functional as F
import numpy as np
import os, argparse

from lib.models.detectors.detector11_DRANet_onlyFgBranch import Network
from utils.sdy_data_val import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--pth_path', type=str, default='./checkpoints/CSGNet/Net_epoch_best.pth')
opt = parser.parse_args()

# for _data_name in ['CAMO', 'COD10K','NC4K']:
# for _data_name in ['CDS2K']:

save_path = './res/{}/'.format(opt.pth_path.split('/')[-2])
for _data_name in ['CHAMELEON','CAMO', 'COD10K','NC4K']:
    data_path = '../Dataset/TestDataset/{}/'.format(_data_name)
    save_path_final = save_path + '/final/{}/'.format(_data_name)
    if not os.path.exists(save_path_final):
        os.makedirs(save_path_final)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = Network()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    image_root = '{}/Image/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        fg_1, fg_2, fg_3,fg_4,fg_5 = model(image)
        #
        res = F.upsample(fg_4+fg_5, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # print('> {} - {}'.format(_data_name, name))
        res = 255 * res
        res = res.astype(np.uint8)
        cv2.imwrite(save_path_final + name, res)
        #
        # res = F.upsample(bg_3, size=gt.shape, mode='bilinear', align_corners=False)
        # res = res.sigmoid().data.cpu().numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # # print('> {} - {}'.format(_data_name, name))
        # res = 255 * res
        # res = res.astype(np.uint8)
        # cv2.imwrite(save_path_bg3 + name, res)
        #
        # res = F.upsample(fg_3 - bg_3, size=gt.shape, mode='bilinear', align_corners=False)
        # res = res.sigmoid().data.cpu().numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # # print('> {} - {}'.format(_data_name, name))
        # res = 255 * res
        # res = res.astype(np.uint8)
        # cv2.imwrite(save_path_final + name, res)
        print('{} - {}'.format(_data_name, name))

    print('{} Finish!'.format(_data_name))