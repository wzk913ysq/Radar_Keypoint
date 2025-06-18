import torch

from dataset_loader import *
import transform as transforms
import importlib
from collections import OrderedDict , namedtuple
from torch.utils.data import DataLoader
import torchvision
from iouEval import iouEval#, getColorEntry
from PIL import Image
import PIL
from os.path import join
import ImageProcess as ip

class load_data():

	def __init__(self, args):

		## First, a bit of setup
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

	def __call__(self, name, split='train', num_images=None, mode='labeled', file_path=False,beforedeal=False):

		transform = self.Img_transform(name, self.d_size[name])
		return self.d_func[name](self.d_path[name], split, transform, file_path, num_images, beforedeal=beforedeal)

	def Img_transform(self, name, size):

		assert (isinstance(size, tuple) and len(size) == 2)
		t = [transforms.Resize(size),
			 transforms.ToTensor()]

		return transforms.Compose(t)


def model_eval(model, args,savedir):
	datasets = args.datasets

	dataset_eval = {dname: get_dataset(dname, 'val', args.num_samples, beforedeal=args.beforedeal) for dname in datasets}
	loader = {dname:DataLoader(dataset_eval[dname], batch_size=args.batch_size,
							shuffle=True) for dname in datasets}
	n_iters = min([len(dataset_eval[d]) for d in datasets])
	loss_criterion = {key: torch.nn.CrossEntropyLoss(ignore_index=2 - 1).cuda() for key in datasets}
	loss_criterion2 = {key: torch.nn.CrossEntropyLoss().cuda() for key in datasets}

	if (args.eval_iou):
		iou_data = {key: iouEval(2) for key in datasets}

	print()
	print('starting eval'.center(100,'-'))
	print()

	imcount = 0

	#model.eval()
	pic_cnt = 0
	deal_loader = transforms.Compose([transforms.ToTensor()])
	deal_unloader = torchvision.transforms.ToPILImage()
	imagedealed = torch.zeros(args.batch_size, 2, 830, 417)

	for d in datasets:

		print_cnt = 0
		for itr, (images, targets) in enumerate(loader[d]):
			with torch.set_grad_enabled(False):

				# torch.onnx._export(model, images, f"{args.trained_model}.onnx", export_params=True)
				seg_output = model(images, enc=False)
				ori_output = {key: seg_output[key].cpu().clone() for key in datasets}
				# loss = loss_criterion[d](seg_output[d], targets.squeeze(1))
				if args.afterdeal:
					# image = seg_output[d]
					for bat in range(args.batch_size):
						try:
							for c in range(2):
								# Tensor to PIL
								image = ip.tensor_to_pil(seg_output[d][bat][c]).convert('RGB')
								# grid = torchvision.utils.make_grid(image, nrow=8, padding=2, pad_value=0,
								#  normalize=False, range=None, scale_each=False)
								# ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
								# image = Image.fromarray(ndarr)
								# image = PIL.ImageOps.invert(image).convert('RGB')
								# dealing
								image = ip.background_wipe(image)
								#image = ip.erode(image, kernel_size=(5, 5)).convert('L')
								# PIL to Tensor
								# image, _ = deal_loader(image, image)
								seg_output[d][bat][c] = deal_loader(image, image)[0][0]
								seg_output[d][bat][c][500:] = 0
						except IndexError:
							pass

				if (args.eval_iou):
					# pred = seg_output[d].argmax(1,True).data
					pred = seg_output[d].data
					iou_target = targets
					iou_data[d].addBatch( pred , iou_target.data)
					print_cnt += 1
					iou_data[d].imSave(d, print_cnt, savedir)
					# seg_output[d] = pred

				B, C, H, W = seg_output[d].shape
				# print(f'loss:{loss.item()},loss_shape:{loss.size()}'
				# 	  f'\nB:{B},C:{C},H:{H},W:{W},iter:{itr}')
				print(f'B:{B},C:{C},H:{H},W:{W},iter:{itr}')
# 				print(type(loss))
# 				return None
				colored_tensor = torch.zeros(4, 3, H, W)
				for bat in range(B):
					for chan in range(3):
						colored_tensor[0, chan, :, :] = images[bat, chan, :, :]
					colored_tensor[1, :, :, :] = targets[bat, 0, : :]
					for j in range(len(seg_output[d][0])):
						try:
							colored_tensor[j + 2, :, :, :] = seg_output[d][bat, j, :, :]
						except IndexError:
							print(f'ERROR  bat:{B},j:{j}')
					colored_tensor[3, :, :, :] = ori_output[d][bat, 0, :, :]
					torchvision.utils.save_image(colored_tensor, os.path.join(savedir + '/test_result', f'{d}_{imcount}.jpg'))
					imcount += 1
					print(f'cnt:{imcount}')
	if (args.eval_iou):
		iou = {d: iou_data[d].getIoU() for d in datasets}

		iouStr_label = {d: '{:0.2f}'.format(iou[d][0] * 100) for d in datasets}
		for d in datasets:
			print("EPOCH IoU on {} dataset: {} %".format(d, iouStr_label[d]))


def main(args,get_dataset):
	savedir = f'./save_drnet'

	if not os.path.exists(savedir + '/test_result'):
		os.makedirs(savedir + '/test_result')
	#Load Model
	assert os.path.exists(args.model + ".py"), f"Error: model definition for {args.model} not found"

	# model_file = importlib.import_module(f'tests.{args.model}')
#	if args.bnsync:
#		model_file.BatchNorm = batchnormsync.BatchNormSync
#	else:
	#model_file.BatchNorm = torch.nn.BatchNorm2d


	NUM_LABELS = get_dataset.num_labels

	# model1 = model_file.Net(NUM_LABELS , args.em_dim , args.resnet)
	# print(model1.seg)
	# print(model1.up)
	model2 = torch.load('model_save/' + args.trained_model, map_location='cpu').module
	#model2 = None
	#print(model2)
	#print(model2.up)
	#print(model)
	print('load over')

	model_eval(model2,args,savedir)



class config:
	def __init__(self):
		self.datasets = ['ELY', 'MID', 'LTE']
		#self.datasets = ['MID']
		self.basedir = '/data1/liwenbo/projects/wang/'
		#self.basedir = 'D:/Net/'
		self.lr = 0.001
		self.num_epochs = 1
		#self.savedir = '1080_full18_test'
		self.num_samples = 100
		self.resnet = 'resnet_18'
		self.model = 'drnet'
		self.alpha = 0
		self.beta = 0
		self.bnsync = None
		self.trained_model = 'model0128.pth'
		self.num_workers = 1
		self.batch_size = 3
		self.em_dim = 100
		self.eval_iou = True
		self.afterdeal = True
		self.beforedeal = True


if __name__ == '__main__':
	args = config()
	get_dataset = load_data(args)
	main(args, get_dataset)
