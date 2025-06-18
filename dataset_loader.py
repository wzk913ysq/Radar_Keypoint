import numpy as np
import os
import random

from PIL import Image
import torch

from torch.utils.data import Dataset
import glob
import ImageProcess as ip


class Relabel:
    
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel
        '''定义一个 Relabel 类，用于替换标签值
        :param olabel: 旧标签值
        :param nlabel: 新标签值'''
    def __call__(self, tensor):
        """
        对 tensor 进行处理，将 tensor 中的旧标签值替换成新标签值
        :param tensor: 输入的 tensor,需要为 torch.LongTensor 或 torch.ByteTensor 类型
        :return: 处理后的 tensor
        """
        assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor

class SegmentationDataset(Dataset):
    def __init__(self, root, subset,
                img_path, label_path, pattern, img_suffix, label_suffix,  file_path=False, transform=None, num_images=None):
                 
        '''定义一个 SegmentationDataset 类，用于处理语义分割数据集的数据和标签，并对其进行变换
                 :param root: 数据集的根目录
                 :param subset: 子目录，如 train 或 val
                 :param img_path: 图像文件夹名称
                 :param label_path: 标签文件夹名称
                 :param pattern: 图像文件名的模式
                 :param img_suffix: 图像文件的后缀
                 :param label_suffix: 标签文件的后缀
                 :param file_path: 是否返回文件路径
                 :param transform: 数据变换函数
                 :param num_images: 处理的图像数量'''
                 
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
        """
        获取数据集中的一组数据
        :param index: 数据的索引值
        :return: 处理后的图像和标签坐标或者图像
        """
        if self.mode == 'labeled':
             # 获取图像和标签路径
            filename = self.image_paths[index]
            filenameGt = self.label_paths[index]

            # 打开图像和标签文件
            with Image.open(filename) as f:
                image = f.convert('RGB')
                if self.beforedeal:
                    image = ip.fixednoise_filter(image)

            with Image.open(filenameGt) as f:
                label = f.convert('P')
        else:
            # 获取无标签图像路径并打开文件
            filename = self.unlabeled_path[index]
            with Image.open(filename) as f:
                image = f.convert('RGB')

#        if self.mode == 'labeled':
#            with Image.open(filenameGt) as f:
#                label = f.convert('P')
#        else:
#            label = image
        # print(image.size, label.size)
        if self.transform !=None:
            # 如果数据集中定义了transform，则对图像和标签进行预处理
            image, label = self.transform(image, label)
            C, H, W = image.size()# 获取图像的通道数、高度和宽度
            image = image[:, :, 0 : int(W / 2)]# 将图像宽度减半
            label = label[:, :, 0 : int(W / 2)]# 将标签宽度减半

        if self.relabel != None and self.mode == 'labeled':
            # 如果数据集中定义了relabel，并且当前数据集的模式是'labeled'，则进行标签重标记
            label = self.relabel(label)

        label_coord = torch.zeros((8,2))
        label_coord[:,0] = torch.tensor([165,205,235,265,295,325,355,405])# 设置每个标签的水平位置
        for i in range(label_coord.shape[0]):
            coord = torch.nonzero(label[...,int(label_coord[i,0])].squeeze())# 获取当前标签的非零坐标
          
        
            avg_coord = (coord[0]+coord[-1])/2 # 求出非零坐标的平均值
            label_coord[i,1] = torch.tensor(round(float(avg_coord)), dtype=torch.int)# 将平均值作为当前标签的垂直位置

        if self.mode == 'unlabeled':
            # 如果当前数据集的模式是'unlabeled'，则只返回图像
            return image
        else:
            # 如果当前数据集的模式是'labeled'，则返回图像和标签坐标
            return image, label_coord

    def __len__(self):
        return len(self.image_paths)

class CityscapesDataset(SegmentationDataset):
    num_classes = 19 # 数据集中的类别数量
    label_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    color_map = np.array([ # 类别颜色映射
        [128, 64,128],
        [244, 35,232],
        [ 70, 70, 70],
        [102,102,156],
        [190,153,153],
        [153,153,153],
        [250,170, 30],
        [220,220,  0],
        [107,142, 35],
        [152,251,152],
        [ 70,130,180],
        [220, 20, 60],
        [255,  0,  0],
        [  0,  0,142],
        [  0,  0, 70],
        [  0, 60,100],
        [  0, 80,100],
        [  0,  0,230],
        [119, 11, 32]
    ], dtype=np.uint8)

    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None , mode='labeled'):
        self.d_idx = 'CS'# 数据集索引
        self.mode = mode# 数据集模式（labeled或unlabeled）
        super(CityscapesDataset, self).__init__(root, subset,  
                img_path = 'leftImg8bit', label_path='gtFine', pattern='*/*',
                img_suffix = '_leftImg8bit.png' , label_suffix='_gtFine_labelTrainIds.png', transform=transform, file_path=file_path, num_images=num_images)

class ANL4Transform(object):
    def __call__(self, image, label):
        indices = label >= 30
        label[indices] = 255
        return image, label

class ANUEDatasetL4(SegmentationDataset):
    num_classes = 30
    label_names = ['road', 'parking', 'drivable fallback', 'sidewalk',  'non-drivable fallback', 'person', 'animal', 'rider', 'motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus', 'caravan',  'vehicle fallback', 'curb', 'wall', 'fence', 'guard rail', 'billboard', 'traffic sign', 'traffic light', 'pole', 'obs-str-bar-fallback', 'building', 'bridge', 'vegetation', 'sky', 'fallback background']

    color_map = np.array([[128, 64, 128], [250, 170, 160], [81, 0, 81], [244, 35, 232], [152, 251, 152], [220, 20, 60], [246, 198, 145], [255, 0, 0], [0, 0, 230], [119, 11, 32], [255, 204, 54], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 0, 90], [136, 143, 153], [220, 190, 40], [102, 102, 156], [190, 153, 153], [180, 165, 180], [174, 64, 67], [220, 220, 0], [250, 170, 30], [153, 153, 153], [0, 0, 0], [70, 70, 70], [150, 100, 100], [107, 142, 35], [70, 130, 180], [169, 187, 214]], dtype=np.uint8)

    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None):
        self.d_idx = 'ANUE'
        super(ANUEDatasetL4, self).__init__(root, subset,  
                img_path = 'leftImg8bit', label_path='gtFine', pattern='*/*',
                img_suffix = '_leftImg8bit.png' , label_suffix='_gtFine_labellevel4Ids.png', transform=transform, file_path=file_path, num_images=num_images)

class IDD_Dataset(SegmentationDataset):
    num_classes = 26
    label_names = ['road', 'drivable fallback', 'sidewalk', 'non-drivable fallback', 'animal', 'rider', 'motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus', 'vehicle fallback', 'curb', 'wall', 'fence', 'guard rail', 'billboard', 'traffic sign', 'traffic light', 'pole', 'obs-str-bar-fallback', 'building', 'bridge', 'vegetation', 'sky']
    color_map   = np.array([
        [128, 64, 128], #road
        [ 81,  0, 81], #drivable fallback
        [244, 35, 232], #sidewalk
        [152, 251, 152], #nondrivable fallback
        [220, 20, 60], #pedestrian
        [255, 0, 0],  #rider
        [0, 0, 230], #motorcycle
        [119, 11, 32], #bicycle
        [255, 204, 54], #autorickshaw
        [0, 0, 142], #car
        [0, 0, 70], #truck
        [0, 60, 100], #bus
        [136, 143, 153], #vehicle fallback
        [220, 190, 40], #curb
        [102, 102, 156], #wall
        [190, 153, 153], #fence
        [180, 165, 180], #guard rail
        [174, 64, 67], #billboard
        [220, 220, 0], #traffic sign
        [250, 170, 30], #traffic light
        [153, 153, 153], #pole
        [169, 187, 214], #obs-str-bar-fallback
        [70, 70, 70], #building
        [150, 120, 90], #bridge
        [107, 142, 35], #vegetation
        [70, 130, 180] #sky
    ], dtype=np.uint8)

    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None, mode='labeled'):
        self.d_idx = 'IDD'
        self.mode = mode
        super().__init__(root, subset,  
                img_path = 'leftImg8bit', label_path='gtFine', pattern='*/*',
                img_suffix = '_leftImg8bit.png' , label_suffix='_gtFine_labellevel3Ids.png', transform=transform, file_path=file_path, num_images=num_images)

class CamVid(SegmentationDataset):
    num_classes = 11
    # label_names = ["Animal", "Archway", "Bicyclist", "Bridge", "Building", "Car", "CartLuggagePram", "Child", "Column_Pole", "Fence", "LaneMkgsDriv", "LaneMkgsNonDriv", "Misc_Text", "MotorcycleScooter", "OtherMoving", "ParkingBlock", "Pedestrian", "Road", "RoadShoulder", "Sidewalk", "SignSymbol", "Sky", "SUVPickupTruck", "TrafficCone", "TrafficLight", "Train", "Tree", "Truck_Bus", "Tunnel", "VegetationMisc", "Void", "Wall"]
    # color_map = np.array([64,128,64], [192,0,128], [0,128,192], [0,128,64], [128,0,0], [64,0,128], [64,0,192], [192,128,64], [192,192,128], [64,64,128], [128,0,192], [192,0,64], [128,128,64], [192,0,192], [128,64,64], [64,192,128], [64,64,0], [128,64,128], [128,128,192], [0,0,192], [192,128,128], [128,128,128], [64,128,192], [0,0,64], [0,64,64], [192,64,128], [128,128,0], [192,128,192], [64,0,64], [192,192,0], [0,0,0], [64,192,0])
    
    def __init__(self, root, subset='train', transform=None,  file_path=False, num_images=None, mode="labeled"):
        self.d_idx = 'CVD'
        self.mode = mode

        self.images_root = f"{root}/{subset}/"
        self.labels_root = f"{root}/{subset}annot/"
        
        self.image_paths = glob.glob(f'{self.images_root}/*.png')
        self.label_paths = glob.glob(f'{self.labels_root}/*.png')

        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]
            self.label_paths = self.label_paths[:num_images]
            
        self.file_path = file_path
        self.transform = transform
        self.relabel = Relabel(255, self.num_classes) if transform != None else None

class SunRGB(SegmentationDataset):
    num_classes = 37
    # label_names = ["Animal", "Archway", "Bicyclist", "Bridge", "Building", "Car", "CartLuggagePram", "Child", "Column_Pole", "Fence", "LaneMkgsDriv", "LaneMkgsNonDriv", "Misc_Text", "MotorcycleScooter", "OtherMoving", "ParkingBlock", "Pedestrian", "Road", "RoadShoulder", "Sidewalk", "SignSymbol", "Sky", "SUVPickupTruck", "TrafficCone", "TrafficLight", "Train", "Tree", "Truck_Bus", "Tunnel", "VegetationMisc", "Void", "Wall"]
    # color_map = np.array([64,128,64], [192,0,128], [0,128,192], [0,128,64], [128,0,0], [64,0,128], [64,0,192], [192,128,64], [192,192,128], [64,64,128], [128,0,192], [192,0,64], [128,128,64], [192,0,192], [128,64,64], [64,192,128], [64,64,0], [128,64,128], [128,128,192], [0,0,192], [192,128,128], [128,128,128], [64,128,192], [0,0,64], [0,64,64], [192,64,128], [128,128,0], [192,128,192], [64,0,64], [192,192,0], [0,0,0], [64,192,0])
    
    def __init__(self, root, subset='train', transform=None,  file_path=False, num_images=None, mode="labeled"):
        self.d_idx = 'SUN'
        self.mode = mode

        listname = f"{root}/{subset}37.txt"

        with open(listname , 'r') as fh:
            self.image_paths = [os.path.join(root , l.split()[0]) for l in fh]

        with open(listname , 'r') as fh:
            self.label_paths = [os.path.join(root , l.split()[-1]) for l in fh]

        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]
            self.label_paths = self.label_paths[:num_images]
            
        self.file_path = file_path
        self.transform = transform
        self.relabel = Relabel(255, self.num_classes) if transform != None else None

class NYUv2_seg(SegmentationDataset):
    num_classes = 13
    # label_names = ["Animal", "Archway", "Bicyclist", "Bridge", "Building", "Car", "CartLuggagePram", "Child", "Column_Pole", "Fence", "LaneMkgsDriv", "LaneMkgsNonDriv", "Misc_Text", "MotorcycleScooter", "OtherMoving", "ParkingBlock", "Pedestrian", "Road", "RoadShoulder", "Sidewalk", "SignSymbol", "Sky", "SUVPickupTruck", "TrafficCone", "TrafficLight", "Train", "Tree", "Truck_Bus", "Tunnel", "VegetationMisc", "Void", "Wall"]
    # color_map = np.array([64,128,64], [192,0,128], [0,128,192], [0,128,64], [128,0,0], [64,0,128], [64,0,192], [192,128,64], [192,192,128], [64,64,128], [128,0,192], [192,0,64], [128,128,64], [192,0,192], [128,64,64], [64,192,128], [64,64,0], [128,64,128], [128,128,192], [0,0,192], [192,128,128], [128,128,128], [64,128,192], [0,0,64], [0,64,64], [192,64,128], [128,128,0], [192,128,192], [64,0,64], [192,192,0], [0,0,0], [64,192,0])
    def __init__(self, root, subset='train', transform=None,  file_path=False, num_images=None, mode="labeled"):
        self.d_idx = 'NYU_s'
        self.mode = mode

        # listname = f"{root}/{subset}13.txt"

        images = os.listdir(os.path.join(root , subset , 'images'))
        labels = os.listdir(os.path.join(root , subset , 'labels'))

        self.image_paths = [f"{root}/{subset}/images/"+im_id for im_id in images]
        self.label_paths = [f"{root}/{subset}/labels/"+lb_id for lb_id in labels]

        # with open(listname , 'r') as fh:
        #     self.image_paths = [os.path.join(root , l.split()[0]) for l in fh]

        # with open(listname , 'r') as fh:
        #     self.label_paths = [os.path.join(root , l.split()[-1]) for l in fh]

        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]
            self.label_paths = self.label_paths[:num_images]
            
        self.file_path = file_path
        self.transform = transform

        self.relabel = Relabel(255, self.num_classes) if transform != None else None

class FeedLine_EL(SegmentationDataset):
    num_classes = 1
    label_names = ['feedline'] # 定义标签名称

    color_map   = np.array([
        [128, 0, 0], #feedline
    ], dtype=np.uint8) # 定义颜色映射

    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None, mode='labeled', beforedeal=False):
        self.d_idx = 'feedline_E' # 定义数据集编号
        self.mode = mode# 定义数据集模式
        self.beforedeal = beforedeal# 定义是否在处理之前处理数据
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

'''
        self.images_root = f'{root}/{img_path}/{subset}'
        self.labels_root = f'{root}/{label_path}/{subset}'
        self.image_paths = glob.glob(f'{self.images_root}/{pattern}')
        self.label_paths = [ img.replace(self.images_root, self.labels_root).replace(img_suffix, label_suffix) for img in self.image_paths  ]
'''
def colorize(img, color, fallback_color=[0,0,0]): 
    img = np.array(img)# 将输入的img转换为numpy数组
    W,H = img.shape# 获取img的宽和高
    view = np.tile(np.array(fallback_color, dtype = np.uint8), (W,H, 1) )# 构造一个与img同样大小的三维numpy数组，并填充为默认颜色
    for i, c in enumerate(color):
        indices = (img == i)# 将img中等于i的像素位置赋值为True，其余位置赋值为False，并生成一个与img大小相同的bool类型numpy数组
        view[indices] = c# 将view中与img中True位置对应的像素值修改为颜色映射中i对应的颜色值
    return view# 返回修改后的三维numpy数组

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    def show_data(ds):
        print(len(ds))# 打印数据集的长度
        i = random.randrange(len(ds))# 从数据集中随机选择一张图像和标签
        img, gt = ds[i]# 获取该图像和标签
        color_gt = colorize(gt, ds.color_map)# 将标签进行颜色映射
        print(img.size,color_gt.shape)
        plt.imshow(img)# 显示图像
        plt.imshow(color_gt, alpha=0.25)# 将标签叠加在图像上显示
        plt.show() # 显示图像和标签

    # cs = CityscapesDataset('/ssd_scratch/cvit/girish.varma/dataset/cityscapes')
    # show_data(cs)

    # an = ANUEDataset('/ssd_scratch/cvit/girish.varma/dataset/anue')
    # show_data(an)

    # bd = BDDataset('/ssd_scratch/cvit/girish.varma/dataset/bdd100k')
    # show_data(bd)

    # mv = MVDataset('/ssd_scratch/cvit/girish.varma/dataset/mvd')
    # show_data(mv)