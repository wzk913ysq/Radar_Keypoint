import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset
import glob

from collections import OrderedDict, namedtuple
from torch.utils.data import DataLoader
from PIL import Image
import ImageProcess as ip
import matplotlib.pyplot as plt
import transform as transforms

class Relabel:
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor

class SegmentationDataset(Dataset):
    def __init__(self, root, subset,
                img_path, label_path, pattern, img_suffix, label_suffix,  file_path=False, transform=None, num_images=None):
        # print(img_path)
        self.images_root = f'{root}/{img_path}/{subset}'
        self.labels_root = f'{root}/{label_path}/{subset}'
        self.image_paths = glob.glob(f'{self.images_root}/{pattern}')
        self.label_paths = [ img.replace(self.images_root, self.labels_root).replace(img_suffix, label_suffix) for img in self.image_paths  ]
        if "idd" in root:
            self.image_paths = self.image_paths[:4000]
            self.label_paths = self.label_paths[:4000]
        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]
            self.label_paths = self.label_paths[:num_images]

        self.file_path = file_path
        self.transform = transform
        self.unlabeled_path = glob.glob(f'{self.images_root}/unlabeled/*/*.png')
        self.relabel = Relabel(255, self.num_classes) if transform != None else None

    def __getitem__(self, index):
        if self.mode == 'labeled':
            filename = self.image_paths[index]
            filenameGt = self.label_paths[index]

            with Image.open(filename) as f:
                image = f.convert('RGB')
                if self.beforedeal:
                    image = ip.fixednoise_filter(image)

            with Image.open(filenameGt) as f:
                label = f.convert('P')
        else:
            filename = self.unlabeled_path[index]
            with Image.open(filename) as f:
                image = f.convert('RGB')

        if self.transform !=None:
            image, label = self.transform(image, label)
            C, H, W = image.size()
            image = image[:, :, 0 : int(W / 2)]
            label = label[:, :, 0 : int(W / 2)]

        if self.relabel != None and self.mode == 'labeled':
            label = self.relabel(label)

        if self.mode == 'unlabeled':
            return image
        else:
            return image, label

    def __len__(self):
        return len(self.image_paths)

class FeedLine_EL(SegmentationDataset):
    num_classes = 1
    label_names = ['feedline']

    color_map   = np.array([
        [128, 0, 0], #feedline
    ], dtype=np.uint8)

    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None, mode='labeled', beforedeal=False):
        self.d_idx = 'feedline_E'
        self.mode = mode
        self.beforedeal = beforedeal
        super().__init__(root, subset,  
                img_path = 'ELY', label_path='ELY', pattern='*/*/img.png',
                img_suffix = 'img.png' , label_suffix='label.png', transform=transform, file_path=file_path, num_images=num_images)

class FeedLine_MD(SegmentationDataset):
    num_classes = 1
    label_names = ['feedline']

    color_map   = np.array([
        [128, 0, 0], #feedline
    ], dtype=np.uint8)

    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None, mode='labeled', beforedeal=False):
        self.d_idx = 'feedline_M'
        self.mode = mode
        self.beforedeal = beforedeal
        super().__init__(root, subset,  
                img_path = 'MID', label_path='MID', pattern='*/*/img.png',
                img_suffix = 'img.png' , label_suffix='label.png', transform=transform, file_path=file_path, num_images=num_images)

class FeedLine_LT(SegmentationDataset):
    num_classes = 1
    label_names = ['feedline']

    color_map   = np.array([
        [128, 0, 0], #feedline
    ], dtype=np.uint8)

    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None, mode='labeled', beforedeal=False):
        self.d_idx = 'feedline_L'
        self.mode = mode
        self.beforedeal = beforedeal
        super().__init__(root, subset,  
                img_path = 'LTE', label_path='LTE', pattern='*/*/img.png',
                img_suffix = 'img.png' , label_suffix='label.png', transform=transform, file_path=file_path, num_images=num_images)

class RadarDataset():
    def __init__(self, args):
        #初始化数据集信息
        dinf = namedtuple('dinf', ['name', 'n_labels', 'func', 'path', 'size'])
        self.metadata = [dinf('ELY', 2, FeedLine_EL, 'datasets', (834, 830)),
                         dinf('MID', 2, FeedLine_MD, 'datasets', (834, 830)),
                         dinf('LTE', 2, FeedLine_LT, 'datasets', (834, 830)),
                         ]

        self.num_labels = {entry.name: entry.n_labels for entry in self.metadata if entry.name in args.datasets}
        self.d_func = {entry.name: entry.func for entry in self.metadata}
        basedir = args.basedir
        self.d_path = {entry.name: basedir + entry.path for entry in self.metadata}
        self.d_size = {entry.name: entry.size for entry in self.metadata}

    def __call__(self, name, split='train', num_images=None, mode='labeled', file_path=False, beforedeal=False):
        transform = self.Img_transform(name, self.d_size[name])
        return self.d_func[name](self.d_path[name], split, transform, file_path, num_images, beforedeal=beforedeal)

    def Img_transform(self, name, size):
        assert (isinstance(size, tuple) and len(size) == 2)
        t = [transforms.Resize(size),
             transforms.ToTensor()]
        return transforms.Compose(t)
    
def colorize(img, color, fallback_color=[0,0,0]): 
    img = np.array(img)
    W,H = img.shape
    view = np.tile(np.array(fallback_color, dtype = np.uint8), (W,H,1))
    for i, c in enumerate(color):
        indices = (img == i)
        view[indices] = c
    return view

class config:
    def __init__(self):
        self.datasets = ['ELY', 'MID', 'LTE']
        self.basedir = 'd:\\Radar_Keypoint_wh01\\'
        self.batch_size = 1
        
if __name__ == "__main__":
    print()
    print('main()'.center(100,'-'))
    print()
    
    #参数加载
    args = config()
    
    #读取数据集
    get_dataset = RadarDataset(args)
    DataNames = args.datasets  #获取数据集路径    
    data_eval = {dname: get_dataset(dname, 'val') for dname in DataNames}  #数据集对象创建
    loader = {dname: DataLoader(data_eval[dname], batch_size=args.batch_size, shuffle=True) for dname in DataNames}  #加载数据集
    
    for d in DataNames:
        print('d = ', d)
        print('len(loader[d]) = ', len(loader[d]))
        
        # loop for input images
        for itr, (imgs, gts) in enumerate(loader[d]):
            # B, C, H, W = imgs.shape 
            # print('B, C, H, W = ', B, C, H, W)  # B, C, H, W =  1 3 830 417
            B, C, H, W = gts.shape 
            print('B, C, H, W = ', B, C, H, W)  # B, C, H, W =  1 3 830 417
            
            # loop for a batch of images
            for bat in range(B):
                img = imgs[bat,:,:,:]
                gt = gts[bat,0,:,:]
                
                plt.figure(figsize=(16,9)) #设置窗口大小
                plt.suptitle('Radar Data') # 图片名称
                #----------------------
                plt.subplot(1,2,1), plt.title('img')
                img = np.transpose(img, axes=[1, 2, 0])
                plt.imshow(img)
                #----------------------
                plt.subplot(1,2,2), plt.title('gt')
                color_gt = colorize(gt, data_eval[d].color_map)
                plt.imshow(gt)
                plt.show()
    

