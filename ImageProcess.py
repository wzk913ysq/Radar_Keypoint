import cv2
import numpy as np
from PIL import Image,ImageOps
import math
import torchvision
import torch

basic = (123, 0)
angle_min = -8.8 * math.pi / 180.0
angle_max = 53.5 * math.pi / 180.0
maxrad = 330
# target_col = (62, 38, 168)

class ImageProcess():
    def __init__(self):
        self.dilate = [5, 5]
        self.erode = [5, 5]
        self.mean_filter = [5, 5]
        self.median_filter = [5, 5]
        self.dict = {0: [5, 5], 1: [5, 5], 2: [5, 5], 3: [5, 5], }
        
#閼垫劘娈�
def erode(img_pil,kernel_size = (5, 5)):
    img = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    kernel = np.ones(kernel_size, np.uint8)
    erosion = cv2.erode(img, kernel)
    # cv2.imshow('img', img)
    # cv2.imshow('eorsion', erosion)
    # cv2.waitKey(0)
    pil = Image.fromarray(cv2.cvtColor(erosion, cv2.COLOR_RGB2BGR))
    return pil

#閼躲劏鍎�
def dilate(img_pil, kernel_size = (5, 5)):
    img = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    kernel = np.ones(kernel_size, np.uint8)
    erosion = cv2.dilate(img, kernel)
    pil = Image.fromarray(cv2.cvtColor(erosion, cv2.COLOR_RGB2BGR))
    return pil

#閸у洤鈧吋鎶ら敓锟�?
def mean_filter(img_pil, kernel=(5, 5)):
    img = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    mean = cv2.blur(img, kernel)
    pil = Image.fromarray(cv2.cvtColor(mean, cv2.COLOR_RGB2BGR))
    return pil

#娑擃厼鈧吋鎶ら敓锟�?
def median_filter(img_pil, kernel=(5,5)):
    img = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    median = cv2.medianBlur(img, kernel[0])
    pil = Image.fromarray(cv2.cvtColor(median, cv2.COLOR_RGB2BGR))
    return pil

def fixednoise_filter(img_pil):
    gray = img_pil.convert('L')
    last_cor = (0, 0)
    avelist = []
    for r in range(1,maxrad):
        pix = 0
        cnt = 0
        for angle in np.arange(angle_min, angle_max, 0.001):
            corX = int(basic[0] + r * math.sin(angle))
            corY = int(basic[1] + r * math.cos(angle))
            if last_cor != (corX, corY):
                last_cor = (corX, corY)
                pix += gray.getpixel((corX, corY))
                cnt += 1
                # pixadd(img_pil, last_cor, col=(int(pix/cnt), 0, 0))
            else:
                pass
        avelist.append(int(pix/cnt))
    mean = np.mean(avelist)
    noise = [index for index, a in zip(range(0, len(avelist)), avelist) if a >mean]
    # for r in range(1,maxrad):
    #     for angle in np.arange(angle_min, angle_max, 0.001):
    #         corX = int(basic[0] + r * math.sin(angle))
    #         corY = int(basic[1] + r * math.cos(angle))
    #         if last_cor != (corX, corY):
    #             last_cor = (corX, corY)
    #             if gray.getpixel((corX, corY)) >= mean:
    #                 pixadd(img_pil, last_cor, col=(255, 255, 255))
    #             else:
    #                 pixadd(img_pil, last_cor, col=(0, 0, 0))
    # print(mean)
    # target_col = img_pil.getpixel((10, 10))
    for index in noise:
        ratio = float(mean) / avelist[index]
        # ratio = 0.1
        for angle in np.arange(angle_min, angle_max, 0.001):
            corX = int(basic[0] + index * math.sin(angle))
            corY = int(basic[1] + index * math.cos(angle))
            if last_cor != (corX, corY):
                last_cor = (corX, corY)
                col = img_pil.getpixel((5, 5))
                # col = [int(col[k] * ratio + target_col[k] * (1 - ratio)) for k in range(3)]
                col = img_pil.getpixel((10, 10))
                pixadd(img_pil, last_cor, col=tuple(col))
    return img_pil
    # img_pil.show()

def pixadd(img_pil, pos=(122, 0), col=(255, 0, 0)):
    try:
        # img_pil.putpixel((pos[0]-1, pos[1]), col)
        img_pil.putpixel((pos[0], pos[1]), col)
        # img_pil.putpixel((pos[0]+1, pos[1]), col)
    except:
        img_pil.putpixel((pos[0], pos[1]), 128)
    return

def background_wipe(img_pil):
    # Three points
    im = img_pil.convert('RGB')
    # basic
    # middle = (416, int((416. - basic[0])/math.tan(angle_max)))
    # left = (0, int(basic[0] / math.tan(angle_min)))
    f = lambda a, b: float(a[0]*b[0] + a[1]*b[1]) / (math.sqrt(a[0]**2 + a[1]**2) * math.sqrt(b[0]**2 + b[1]**2))
    if img_pil.size == (417, 830):
        vect0 = (-basic[0], -basic[1])
        for corX in range(417):
            for corY in range(830):
                if (corX,corY)==basic:
                    continue
                vect = (corX - basic[0], corY - basic[1])
                a = f(vect, vect0)
                angle = math.acos(a)
                if angle < 0 or angle > math.pi:
                    continue
                if angle < (angle_min + math.pi/2):
                    pixadd(im, (corX,corY), col=(0,0,0))
                elif angle > (angle_max + math.pi/2):
                    pixadd(im, (corX,corY), col=(0,0,0))

    elif img_pil.size == (834, 830):
        vect0 = (-basic[0], -basic[1])
        for corX in range(417):
            for corY in range(830):
                if (corX,corY)==basic:
                    continue
                vect = (corX - basic[0], corY - basic[1])
                angle = math.acos(f(vect,vect0))
                if angle < 0 or angle > math.pi:
                    continue
                if angle < (angle_min + math.pi/2):
                    pixadd(im, (corX,corY), col=(0,0,0))
                elif angle > (angle_max + math.pi/2):
                    pixadd(im, (corX,corY), col=(0,0,0))
        basic_r = (833 - basic[0], 0)
        vect0 = (833 - basic_r[0], -basic[1])
        for corX in range(417, 834):
            for corY in range(830):
                if (corX,corY)==basic_r:
                    continue
                vect = (corX - basic_r[0], corY - basic_r[1])
                angle = math.acos(f(vect,vect0))
                if angle < 0 or angle > math.pi:
                    continue
                if angle < (angle_min + math.pi/2):
                    pixadd(im, (corX,corY), col=(0,0,0))
                elif angle > (angle_max + math.pi/2):
                    pixadd(im, (corX,corY), col=(0,0,0))
    else:
        return img_pil

    return im

def tensor_to_pil(tensor):
    image = tensor
    grid = torchvision.utils.make_grid(image, nrow=8, padding=2, pad_value=0,normalize=False, range=None, scale_each=False)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    image = Image.fromarray(ndarr)
    return ImageOps.invert(image)
