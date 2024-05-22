import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from lib.Network_Res2Net_GRA_NCD import Network
from utils.data_val import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshot/MRI/Net_epoch_best.pth')
opt = parser.parse_args()

data_path = './dataset_test/'
save_path = './res/'
model = Network(imagenet_pretrained=False)
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()
os.makedirs(save_path, exist_ok=True)
image_root = '{}/images/'.format(data_path)
gt_root = '{}/masks/'.format(data_path)
test_loader = test_dataset(image_root, gt_root, opt.testsize)
for i in range(test_loader.size):
    image, gt, name, _ = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.cuda()
    res5, res4, res3, res2 = model(image)
    res = res2
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    print('> {} - {}'.format("", name))
    misc.imsave(save_path+name, res)
    # If `mics` not works in your environment, please comment it and then use CV2
    # cv2.imwrite(save_path+name,res*255)