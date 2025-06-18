import numpy as np
import os

from PIL import Image
import torch

from torch.utils.data import Dataset
import glob
import ImageProcess as ip

class Relabel:
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor

class SegmentationDataset(Dataset):
    def __init__(self, root, subset, img_path, label_path, pattern, img_suffix, label_suffix,  file_path=False, transform=None, num_images=None):
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
        
        coord = torch.nonzero(label.squeeze())
        min, _ = torch.min(coord,dim = 0)
        max, _ = torch.max(coord,dim = 0)
        label_coord = (min[0] + max[0])/2
        
        if self.mode == 'unlabeled':
            return image
        else:
            return image, label_coord, filename 
            
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
