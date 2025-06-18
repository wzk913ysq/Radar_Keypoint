import torch
import time
import transform as transforms
import importlib
from collections import OrderedDict, namedtuple
from torch.utils.data import DataLoader
import torchvision

from PIL import Image, ImageDraw
import PIL
from os.path import join
import ImageProcess as ip

# wh2021-09-18: user define...
from dataset_loader_wh01 import *
from iouEval_test import iouEval #, getColorEntry

def model_eval(model, args):
    #日志文件清空
    with open('d:\\Radar_Keypoint_wh01\\result_save\\bad_result\\log.txt', "w") as myfile:
        print('clear')

    print()
    print('starting eval'.center(100,'-'))
    print()

    #测试模型
    model.eval()
    
    #读取数据集
    get_dataset = RadarDataset(args)
    datasets = args.datasets  #获取数据集路径    
    dataset_eval = {dname: get_dataset(dname, 'val') for dname in datasets}  #数据集对象创建
    loader = {dname:DataLoader(dataset_eval[dname], batch_size=args.batch_size, shuffle=True) for dname in datasets}  #加载数据集
    for d in datasets:
        for itr, (images_l, targets_l) in enumerate(loader[d]):
            start_time = time.time()
            #模型测试，计算loss
            dec_outputs = model(images_l)
            print('type(dec_outputs) = ', type(dec_outputs))        # type(dec_outputs) =  <class 'tuple'>
            print('type(dec_outputs[0]) = ', type(dec_outputs[0]))  # type(dec_outputs[0]) =  <class 'dict'>
            print('type(dec_outputs[1]) = ', type(dec_outputs[1]))  # type(dec_outputs[1]) =  <class 'torch.Tensor'>          
            print('dec_outputs[1].shape = ', dec_outputs[1].shape)  # torch.Size([1, 100, 830, 417])
            print('len(dec_outputs[1]) = ', len(dec_outputs[1]))    # len(dec_outputs[1]) =  1
            
            B, C, H, W = images_l.shape 
            print('B, C, H, W = ', B, C, H, W)  # B, C, H, W =  1 3 830 417
            
            #保存图像
            for bat in range(B):
                colored_tensor = torch.zeros(2, 3, H, W*2)
                colored_tensor[0, :, :, 0:W] = images_l[bat, :, :, :]
                colored_tensor[1, :, :, 0:W] = images_l[bat, :, :, :]  

                # #画点
                # for i in range(3):
                # 	colored_tensor[1, :, targets_l.int()[bat, i, 1]-4:targets_l.int()[bat, i, 1]+5, targets_l.int()[bat, i, 0]-4:targets_l.int()[bat, i, 0]+5] = 0
                # 	colored_tensor[1, :, dec_outputs.int()[bat, i, 1]-4:dec_outputs.int()[bat, i, 1]+5, dec_outputs.int()[bat, i, 0]-4:dec_outputs.int()[bat, i, 0]+5] = 0
                # 	colored_tensor[1, 0, targets_l.int()[bat, i, 1]-4:targets_l.int()[bat, i, 1]+5, targets_l.int()[bat, i, 0]-4:targets_l.int()[bat, i, 0]+5] = 1
                # 	colored_tensor[1, 2, targets_l.int()[bat, i, 1]-4:targets_l.int()[bat, i, 1]+5, targets_l.int()[bat, i, 0]-4:targets_l.int()[bat, i, 0]+5] = 1
                # 	colored_tensor[1, :2 , dec_outputs.int()[bat, i, 1]-4:dec_outputs.int()[bat, i, 1]+5, dec_outputs.int()[bat, i, 0]-1:dec_outputs.int()[bat, i, 0]+2] = 1
                # 	colored_tensor[1, :2 , dec_outputs.int()[bat, i, 1]-1:dec_outputs.int()[bat, i, 1]+2, dec_outputs.int()[bat, i, 0]-4:dec_outputs.int()[bat, i, 0]+5] = 1
                # colored_tensor[1, :, targets_l.int()[bat, 3, 1]-4:targets_l.int()[bat, 3, 1]+5, 413:417] = 0
                # colored_tensor[1, :, dec_outputs.int()[bat, 3, 1]-4:dec_outputs.int()[bat, 3, 1]+5, 413:417] = 0
                # colored_tensor[1, 0, targets_l.int()[bat, 3, 1]-4:targets_l.int()[bat, 3, 1]+5, 413:417] = 1
                # colored_tensor[1, 2, targets_l.int()[bat, 3, 1]-4:targets_l.int()[bat, 3, 1]+5, 413:417] = 1
                # colored_tensor[1, :2 , dec_outputs.int()[bat, 3, 1]-4:dec_outputs.int()[bat, 3, 1]+5, 415:417] = 1
                # pos = int(dec_outputs[bat, 3, 1])
                # colored_tensor[1, :2 , (pos-1):(pos+2), 413:417] = 1

                #画折线	
                img = torchvision.transforms.ToPILImage()(colored_tensor[1, :, :, :W])
                img_draw = ImageDraw.Draw(img)
                # index = dec_outputs[bat].view(8).int().tolist()
                # img_draw.line(index, fill = (255, 0, 0), width = 3)
                colored_tensor[1, :, :, :W] = torchvision.transforms.ToTensor()(img)
                colored_tensor[:, :, :, W:W*2] = torch.flip(colored_tensor[:, :, :,:W], [3])

                #保存
                torchvision.utils.save_image(colored_tensor, os.path.join('d:\\Radar_Keypoint_wh01\\result_save\\val_result', f'val_{d}_{itr}.jpg'))

    print()
    print('finished'.center(100,'-'))
    print()

class config:
    def __init__(self):
        self.datasets = ['ELY', 'MID', 'LTE']
        self.basedir = 'd:\\Radar_Keypoint_wh01\\'
        self.model = 'drnet'
        self.trained_model = 'model_best.pth' # 'model_64.pth'
        self.batch_size = 1
        
if __name__ == '__main__':
    print()
    print('main()'.center(100,'-'))
    print()
    
    #参数加载
    args = config()
    
    #模型加载
    assert os.path.exists(args.model + ".py"), f"Error: model definition for {args.model} not found"
    sfile = 'd:\\Radar_Keypoint_wh01\\model_save\\'+ args.trained_model	
    model2 = torch.load(sfile, map_location='cpu').module
    print('load over')
    
    model_eval(model2, args)