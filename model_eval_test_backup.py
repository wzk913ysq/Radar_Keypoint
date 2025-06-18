import torch
import time
from dataset_loader import *
import transform as transforms
import importlib
from collections import OrderedDict , namedtuple
from torch.utils.data import DataLoader
import torchvision
from iouEval_test import iouEval#, getColorEntry
from PIL import Image, ImageDraw
import PIL
from os.path import join
import ImageProcess as ip

class load_data():

	def __init__(self, args):

		#初始化数据集信息
		dinf = namedtuple('dinf', ['name', 'n_labels', 'func', 'path', 'size'])
		self.metadata = [dinf('ELY', 2, FeedLine_EL, 'datasets_copy', (834, 830)),
						 dinf('MID', 2, FeedLine_MD, 'datasets', (834, 830)),
						 dinf('LTE', 2, FeedLine_LT, 'datasets', (834, 830)),
						 ]

		self.num_labels = {entry.name: entry.n_labels for entry in self.metadata if entry.name in args.datasets}

		self.d_func = {entry.name: entry.func for entry in self.metadata}
		basedir = args.basedir
		self.d_path = {entry.name: basedir + entry.path for entry in self.metadata}
		self.d_size = {entry.name: entry.size for entry in self.metadata}

	def __call__(self, name, split='train', num_images=None, mode='labeled', file_path=False,beforedeal=False):

		transform = self.Img_transform(name, self.d_size[name])
		return self.d_func[name](self.d_path[name], split, transform, file_path, num_images, beforedeal=beforedeal)

	def Img_transform(self, name, size):

		assert (isinstance(size, tuple) and len(size) == 2)
		t = [transforms.Resize(size),
			 transforms.ToTensor()]

		return transforms.Compose(t)


def model_eval(model, args):

	#获取数据集路径
	datasets = args.datasets

	#数据集对象创建
	dataset_eval = {dname: get_dataset(dname, 'val') for dname in datasets}
	dataset_train = {dname: get_dataset(dname, 'train') for dname in datasets}

	#加载数据集
	loader_train = {dname:DataLoader(dataset_train[dname],  batch_size=args.batch_size, 
							shuffle=True) for dname in datasets}
	loader = {dname:DataLoader(dataset_eval[dname], batch_size=args.batch_size,
							shuffle=True) for dname in datasets}

	#日志文件清空
	with open('/data1/liwenbo/projects/wang/result_save/bad_result/log.txt', "w") as myfile:
		print('clear')

	print()
	print('starting eval'.center(100,'-'))
	print()

	#测试模型
	model.eval()

	for d in datasets:

		for itr, (images_l, targets_l, path) in enumerate(loader[d]):

			start_time = time.time()
			#模型测试，计算loss
			dec_outputs = model(images_l)
			print(time.time() - start_time)
			loss_s = torch.nn.MSELoss().cuda()(dec_outputs, targets_l)

			#若loss过大，保存图像，并保存路径信息
			if loss_s > 40:

				B, C, H, W = images_l.shape

				#保存图像
				for bat in range(B):
					colored_tensor = torch.zeros(2, 3, H, W*2)
					colored_tensor[0, :, :, 0:W] = images_l[bat, :, :, :]
					colored_tensor[1, :, :, 0:W] = images_l[bat, :, :, :]  

					#画点
					for i in range(3):
						colored_tensor[1, :, targets_l.int()[bat, i, 1]-4:targets_l.int()[bat, i, 1]+5, targets_l.int()[bat, i, 0]-4:targets_l.int()[bat, i, 0]+5] = 0
						colored_tensor[1, :, dec_outputs.int()[bat, i, 1]-4:dec_outputs.int()[bat, i, 1]+5, dec_outputs.int()[bat, i, 0]-4:dec_outputs.int()[bat, i, 0]+5] = 0
						colored_tensor[1, 0, targets_l.int()[bat, i, 1]-4:targets_l.int()[bat, i, 1]+5, targets_l.int()[bat, i, 0]-4:targets_l.int()[bat, i, 0]+5] = 1
						colored_tensor[1, 2, targets_l.int()[bat, i, 1]-4:targets_l.int()[bat, i, 1]+5, targets_l.int()[bat, i, 0]-4:targets_l.int()[bat, i, 0]+5] = 1
						colored_tensor[1, :2 , dec_outputs.int()[bat, i, 1]-4:dec_outputs.int()[bat, i, 1]+5, dec_outputs.int()[bat, i, 0]-1:dec_outputs.int()[bat, i, 0]+2] = 1
						colored_tensor[1, :2 , dec_outputs.int()[bat, i, 1]-1:dec_outputs.int()[bat, i, 1]+2, dec_outputs.int()[bat, i, 0]-4:dec_outputs.int()[bat, i, 0]+5] = 1
					colored_tensor[1, :, targets_l.int()[bat, 3, 1]-4:targets_l.int()[bat, 3, 1]+5, 413:417] = 0
					colored_tensor[1, :, dec_outputs.int()[bat, 3, 1]-4:dec_outputs.int()[bat, 3, 1]+5, dec_outputs.int()[bat, 3, 0]-4:dec_outputs.int()[bat, 3, 0]+1] = 0
					colored_tensor[1, 0, targets_l.int()[bat, 3, 1]-4:targets_l.int()[bat, 3, 1]+5, 413:417] = 1
					colored_tensor[1, 2, targets_l.int()[bat, 3, 1]-4:targets_l.int()[bat, 3, 1]+5, 413:417] = 1
					colored_tensor[1, :2 , dec_outputs.int()[bat, 3, 1]-4:dec_outputs.int()[bat, 3, 1]+5, dec_outputs.int()[bat, 3, 0]-1:dec_outputs.int()[bat, 3, 0]+1] = 1
					colored_tensor[1, :2 , dec_outputs.int()[bat, 3, 1]-1:dec_outputs.int()[bat, 3, 1]+2, dec_outputs.int()[bat, 3, 0]-4:dec_outputs.int()[bat, 3, 0]+1] = 1

					#画折线	
					img = torchvision.transforms.ToPILImage()(colored_tensor[1, :, :, :W])
					img_draw = ImageDraw.Draw(img)
					index = dec_outputs[bat].view(8).int().tolist()
					img_draw.line(index,fill = (255, 0, 0),width = 3)
					colored_tensor[1, :, :, :W] = torchvision.transforms.ToTensor()(img)
					colored_tensor[:, :, :, W:W*2] = torch.flip(colored_tensor[:, :, :,:W], [3])

					#保存
					torchvision.utils.save_image(colored_tensor, os.path.join('/data1/liwenbo/projects/wang/result_save/bad_result', f'val_{d}_{itr}.jpg'))

					#保存路径信息
					with open('/data1/liwenbo/projects/wang/result_save/bad_result/log.txt', "a") as myfile:
						myfile.write("\nval_{}_{}:{}".format(d, itr, path))

		'''for itr, (images_l, targets_l, path) in enumerate(loader_train[d]):

			dec_outputs = model(images_l)
			loss_s = torch.nn.MSELoss().cuda()(dec_outputs, targets_l)
			
			if loss_s > 40:

				B, C, H, W = images_l.shape

				for bat in range(B):
					colored_tensor = torch.zeros(2, 3, H, W*2)
					colored_tensor[0, :, :, 0:W] = images_l[bat, :, :, :]
					colored_tensor[1, :, :, 0:W] = images_l[bat, :, :, :]  
					for i in range(3):
						colored_tensor[1, :, targets_l.int()[bat, i, 1]-4:targets_l.int()[bat, i, 1]+5, targets_l.int()[bat, i, 0]-4:targets_l.int()[bat, i, 0]+5] = 0
						colored_tensor[1, :, dec_outputs.int()[bat, i, 1]-4:dec_outputs.int()[bat, i, 1]+5, dec_outputs.int()[bat, i, 0]-4:dec_outputs.int()[bat, i, 0]+5] = 0
						colored_tensor[1, 0, targets_l.int()[bat, i, 1]-4:targets_l.int()[bat, i, 1]+5, targets_l.int()[bat, i, 0]-4:targets_l.int()[bat, i, 0]+5] = 1
						colored_tensor[1, 2, targets_l.int()[bat, i, 1]-4:targets_l.int()[bat, i, 1]+5, targets_l.int()[bat, i, 0]-4:targets_l.int()[bat, i, 0]+5] = 1
						colored_tensor[1, :2 , dec_outputs.int()[bat, i, 1]-4:dec_outputs.int()[bat, i, 1]+5, dec_outputs.int()[bat, i, 0]-1:dec_outputs.int()[bat, i, 0]+2] = 1
						colored_tensor[1, :2 , dec_outputs.int()[bat, i, 1]-1:dec_outputs.int()[bat, i, 1]+2, dec_outputs.int()[bat, i, 0]-4:dec_outputs.int()[bat, i, 0]+5] = 1
					colored_tensor[1, :, targets_l.int()[bat, 3, 1]-4:targets_l.int()[bat, 3, 1]+5, 413:417] = 0
					colored_tensor[1, :, dec_outputs.int()[bat, 3, 1]-4:dec_outputs.int()[bat, 3, 1]+5, dec_outputs.int()[bat, 3, 0]-4:dec_outputs.int()[bat, 3, 0]+1] = 0
					colored_tensor[1, 0, targets_l.int()[bat, 3, 1]-4:targets_l.int()[bat, 3, 1]+5, 413:417] = 1
					colored_tensor[1, 2, targets_l.int()[bat, 3, 1]-4:targets_l.int()[bat, 3, 1]+5, 413:417] = 1
					colored_tensor[1, :2 , dec_outputs.int()[bat, 3, 1]-4:dec_outputs.int()[bat, 3, 1]+5, dec_outputs.int()[bat, 3, 0]-1:dec_outputs.int()[bat, 3, 0]+1] = 1
					colored_tensor[1, :2 , dec_outputs.int()[bat, 3, 1]-1:dec_outputs.int()[bat, 3, 1]+2, dec_outputs.int()[bat, 3, 0]-4:dec_outputs.int()[bat, 3, 0]+1] = 1	
					img = torchvision.transforms.ToPILImage()(colored_tensor[1, :, :, :W])
					img_draw = ImageDraw.Draw(img)
					index = dec_outputs[bat].view(8).int().tolist()
					img_draw.line(index,fill = (255, 0, 0),width = 3)
					colored_tensor[1, :, :, :W] = torchvision.transforms.ToTensor()(img)
					colored_tensor[:, :, :, W:W*2] = torch.flip(colored_tensor[:, :, :,:W], [3])
					torchvision.utils.save_image(colored_tensor, os.path.join('/data1/liwenbo/projects/wang/result_save/bad_result', f'train_{d}_{itr}.jpg'))

					with open('/data1/liwenbo/projects/wang/result_save/bad_result/log.txt', "a") as myfile:
						myfile.write("\ntrain_{}_{}:{}".format(d, itr, path))'''

	print()
	print('finished'.center(100,'-'))
	print()

def main(args,get_dataset):

	#模型加载
	assert os.path.exists(args.model + ".py"), f"Error: model definition for {args.model} not found"
	model2 = torch.load('/data1/liwenbo/projects/wang/model_save/'+ args.trained_model, map_location='cpu').module
	print('load over')
	model_eval(model2,args)



class config:
	def __init__(self):
		self.datasets = ['ELY', 'MID', 'LTE']
		self.basedir = '/data1/liwenbo/projects/wang/'
		self.model = 'drnet'
		self.trained_model = 'model_64.pth'
		self.batch_size = 1


if __name__ == '__main__':

	#参数加载
	args = config()
	get_dataset = load_data(args)

	main(args, get_dataset)