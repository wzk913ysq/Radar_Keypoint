import os
import time
import numpy as np
import torch
import sys

from PIL import Image, ImageOps, ImageDraw
from argparse import ArgumentParser
from EntropyLoss import EmbeddingLoss
from iouEval import iouEval#, getColorEntry

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torch.nn.functional as F
import setproctitle as spt

from dataset_loader import *
import transform as transforms

import importlib
from collections import OrderedDict , namedtuple

from shutil import copyfile

import ImageProcess as ip

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class load_data():

	def __init__(self, args):
# 初始化
        # 元祖包括数据集名称、标签数、处理函数、路径和尺寸
		## First, a bit of setup
		dinf = namedtuple('dinf', ['name', 'n_labels', 'func', 'path', 'size'])
		self.metadata = [dinf('ELY', 2, FeedLine_EL, 'datasets', (834, 830)),
						 dinf('MID', 2, FeedLine_MD, 'datasets', (834, 830)),
						 dinf('LTE', 2, FeedLine_LT, 'datasets', (834, 830)),
						 ]
# 存储每个数据集的标签数 # 使用列表推导式获取数据集的类别数，将结果存储到字典 self.num_labels 中
		self.num_labels = {entry.name: entry.n_labels for entry in self.metadata if entry.name in args.datasets}
# 获取数据集的路径、函数和尺寸
		self.d_func = {entry.name: entry.func for entry in self.metadata}
		basedir = args.basedir
		self.d_path = {entry.name: basedir + entry.path for entry in self.metadata}
		self.d_size = {entry.name: entry.size for entry in self.metadata}

	def __call__(self, name, split='train', num_images=None, mode='labeled', file_path=False):
# 获取数据集  # 构造变换 transform
		transform = self.Img_transform(name, self.d_size[name])
		# 调用数据集的处理函数，并返回结果
		return self.d_func[name](self.d_path[name], split, transform, file_path, num_images, mode)

	def Img_transform(self, name, size):
# 判断 size 是否为二元组
		assert (isinstance(size, tuple) and len(size) == 2)
		t = [transforms.Resize(size),# 缩放图像
			 transforms.ToTensor()]  # 将 PIL.Image 或 numpy.ndarray 转化为 tensor
        # 返回 Compose 函数，将 t 中的变换串联起来

		return transforms.Compose(t)


def train(args, get_dataset, model, enc=False):
	best_acc = 2000
# 设置训练轮数
	num_epochs = 10 if args.debug else args.num_epochs
	
# 获取当前可用的GPU数量
	n_gpus = torch.cuda.device_count()
	print("\nWorking with {} GPUs".format(n_gpus))

	datasets = args.datasets

	NUM_LABELS = get_dataset.num_labels
# 获取训练集和验证集
	dataset_train = {dname: get_dataset(dname, 'train', args.num_samples) for dname in datasets}
	dataset_val = {dname: get_dataset(dname, 'val',args.num_samples) for dname in datasets}
	# dataset_unlabeled = {dname: get_dataset(dname, co_transform, 'train_extra' , mode='unlabeled') for dname in datasets}
	dataset_unlabeled = {dname: get_dataset(dname, 'train'  , mode='unlabeled') for dname in datasets}

	print("Working with {} Dataset(s):".format(len(datasets)))
	for key in datasets:
		print("{}: Unlabeled images {}, Training on {} images, Validation on {} images".format(key , len(dataset_unlabeled[key]), len(dataset_train[key]) , len(dataset_val[key])))
# 遍历数据集
	for d in datasets:
		 # 如果训练集中的值不相等
		if len(set(dataset_train.values())) != 1:
			# 获取最大训练集大小
			max_train_size = np.max([ len(dataset_train[dname]) for dname in datasets]) 
			# 将该数据集的图像路径和标签路径重复若干次，使其大小与最大训练集大小一致
			dataset_train[d].image_paths = dataset_train[d].image_paths*int(np.ceil(float(max_train_size)/len(dataset_train[d].image_paths)))
			dataset_train[d].label_paths = dataset_train[d].label_paths*int(np.ceil(float(max_train_size)/len(dataset_train[d].label_paths)))

		# 创建训练集和验证集的数据加载器
	loader_train = {dname:DataLoader(dataset_train[dname], num_workers=args.num_workers, batch_size=args.batch_size, 
							shuffle=True) for dname in datasets}
	loader_val = {dname:DataLoader(dataset_val[dname], num_workers=args.num_workers, batch_size=1, 
							shuffle=True, drop_last=True) for dname in datasets}
# 设置保存路径
	savedir = f'./save_drnet'

	if (enc):
		automated_log_path = savedir + "/automated_log_encoder.txt"
		modeltxtpath = savedir + "/model_encoder.txt"
	else:
		automated_log_path = savedir + "/automated_log.txt"
		modeltxtpath = savedir + "/model.txt"  

	loss_logpath = savedir + "/loss_log.txt"  
# 如果文件不存在，则写入文件
	if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
		with open(automated_log_path, "a") as myfile:
			if len(datasets) > 1:
				myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tlearningRate")
			else:
				myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tlearningRate")
# 写入模型参数到文件
	with open(modeltxtpath, "w") as myfile:
		myfile.write(str(model))
# 如果文件不存在，则写入文件
	if (not os.path.exists(loss_logpath)):
		with open(loss_logpath , "w") as myfile:
			if len(datasets) > 1:
				myfile.write("Epoch\t\tS1\t\tS2\t\tUS1\t\tUS2\t\tTotal\n")
			else:
				myfile.write("Epoch\t\tS1\t\tS2\t\tTotal\n")

	
# 根据模型类型设置优化器
	if args.model == 'drnet':
		#optimizer = SGD(model.optim_parameters(), args.lr, 0.9,  weight_decay=1e-4)
		optimizer = Adam(model.optim_parameters(), lr=args.lr, weight_decay=1e-4)
	""" if args.cuda:
		model = torch.nn.DataParallel(model).cuda() """
	model = model.cuda()# 将模型转移到 GPU 上
# 设置保存标签嵌入文件的路径，并初始化平均 epoch 损失
	le_file = savedir + '/label_embedding.pt'
	average_epoch_loss = {'train':np.inf , 'val':np.inf}
# 为每个数据集创建一个名为 'label_embedding' 的字典
# 键是数据集名称，值是大小为 (NUM_LABELS[key], args.em_dim) 的张量，其中 NUM_LABELS[key] 是该数据集中的标签数量，args.em_dim 是嵌入维度
# 初始值为随机张量
	label_embedding = {key:torch.randn(NUM_LABELS[key] , args.em_dim).cuda() for key in datasets} ## Random Initialization
## 如果提供了标签嵌入文件，则使用它进行初始化
	## If provided, use label embedddings
	if args.pt_em:
		fn = torch.load(args.pt_em)
		label_embedding = {key : torch.tensor(fn[key] , dtype=torch.float).cuda() for key in datasets}
# 初始化 start_epoch 为 1
	start_epoch = 1
	if args.resume:
		#Must load weights, optimizer, epoch and best value. 
		# 如果要从已保存的模型中继续训练，需要加载模型参数、优化器状态、当前 epoch 和最佳准确率
		if enc:
			filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
		else:
			filenameCheckpoint = savedir + '/checkpoint.pth.tar'

		assert os.path.exists(filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
		checkpoint = torch.load(filenameCheckpoint)
		start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		best_acc = checkpoint['best_acc']
		# 如果数据集数量大于 1，则从保存的标签嵌入文件中加载数据
    # 否则为 None
		label_embedding = torch.load(le_file) if len(datasets) >1 else None
		print("=> Loaded checkpoint at epoch {}".format(checkpoint['epoch']))

	# 使用LambdaLR创建一个学习率调度器，它可以在训练期间修改优化器的学习率
# 学习率会在每个epoch时更新，更新规则由一个lambda函数确定，该函数接收当前epoch号并返回新的学习率
	scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9))  ## scheduler 2
	#loss_criterion = {key:torch.nn.CrossEntropyLoss().cuda() for key in datasets}
	# 创建一个字典，为每个数据集设置使用Mean Squared Error损失函数的损失标准
# 如果有多个数据集，创建一个EmbeddingLoss模块并使用DataParallel在多个GPU上并行化
	loss_criterion = {key:torch.nn.MSELoss().cuda() for key in datasets}
 
	if len(datasets)>1:
		similarity_module = EmbeddingLoss(NUM_LABELS, args.em_dim, label_embedding, loss_criterion)
		similarity_module = torch.nn.DataParallel(similarity_module).cuda()
		# 将标签嵌入保存到文件中
		torch.save(label_embedding , le_file)

	print()
	print("========== STARTING TRAINING ===========")
	print()
	
	for epoch in range(start_epoch, num_epochs+1):
		spt.setproctitle('wh {}/{}'.format(epoch, num_epochs))
		epoch_start_time = time.time()
		usedLr = 0
	
		###### TRAIN begins  #################
		for phase in ['train']:

			print("-----", phase ,"- EPOCH", epoch, "-----")

			scheduler.step(epoch)    
			model.train()
# 打印当前的学习率，并将其记录在 usedLr 中
			for param_group in optimizer.param_groups:
				print("LEARNING RATE: " , param_group['lr'])
				usedLr = float(param_group['lr'])

			epoch_loss = {d:[] for d in datasets}
			time_taken = []    

			for d in datasets:
		
				loss_sup = {d:0 for d in datasets}
				
# 从 loader_train[d] 中加载数据并进行训练
				for itr, (images_l , targets_l) in enumerate(loader_train[d]):
# 将数据和标签转移到 GPU 上
					optimizer.zero_grad()
                              
					images_l = images_l.cuda()
					targets_l = targets_l.cuda()

					start_time = time.time()
 # 使用模型生成预测结果，并计算损失
					dec_outputs = model(images_l)  

					loss_s = loss_criterion[d](dec_outputs, targets_l)
     
					loss_s.backward()	

					optimizer.step()
# 记录每个 batch 的训练时间
					time_taken.append(time.time() - start_time)
                            
        			#visualization   # 每 args.steps_loss 个 batch 打印一次可视化结果
					if args.steps_loss > 0 and itr % args.steps_loss == 0 :                                
						B, C, H, W = images_l.shape
						imcount = 0
						
						for bat in range(B):
							# 将生成的可视化结果保存到磁盘
                        # 首先创建一个彩色张量，将真实关键点和模型预测的关键点绘制在张量上
                        # 然后将张量转换为 PIL 图像，使用 ImageDraw 库在图像上画出模型预测的关键点
                        # 最后将 PIL 图像转换为 Tensor，并保存到磁盘
							colored_tensor = torch.zeros(3, H, W)
							colored_tensor = images_l[bat, :, :, :] 
							for i in range(4):
								if dec_outputs.int()[bat, i, 1] < 825 and dec_outputs.int()[bat, i, 0] < 412 :
									colored_tensor[:, targets_l.int()[bat, i, 1]-4:targets_l.int()[bat, i, 1]+4, targets_l.int()[bat, i, 0]-4:targets_l.int()[bat, i, 0]+4] = 0
									colored_tensor[:, dec_outputs.int()[bat, i, 1]-4:dec_outputs.int()[bat, i, 1]+4, dec_outputs.int()[bat, i, 0]-4:dec_outputs.int()[bat, i, 0]+4] = 0
									colored_tensor[1, targets_l.int()[bat, i, 1]-4:targets_l.int()[bat, i, 1]+4, targets_l.int()[bat, i, 0]-4:targets_l.int()[bat, i, 0]+4] = 1
									colored_tensor[:2 , dec_outputs.int()[bat, i, 1]-4:dec_outputs.int()[bat, i, 1]+4, dec_outputs.int()[bat, i, 0]-1:dec_outputs.int()[bat, i, 0]+1] = 1
									colored_tensor[:2 , dec_outputs.int()[bat, i, 1]-1:dec_outputs.int()[bat, i, 1]+1, dec_outputs.int()[bat, i, 0]-4:dec_outputs.int()[bat, i, 0]+4] = 1
							img = torchvision.transforms.ToPILImage()(torch.Tensor.cpu(colored_tensor))
							img_draw = ImageDraw.Draw(img)
							index = dec_outputs[bat].view(8).int().tolist()
							img_draw.line(index,fill = (255, 0, 0),width = 3)
							colored_tensor = torchvision.transforms.ToTensor()(img)
							torchvision.utils.save_image(colored_tensor, os.path.join('/data/wanghao/Radar_Keypoint_wh01/result_save/test_result', f'train_{d}_{imcount}.jpg'))
							imcount += 1

					loss_sup[d] = loss_s.item()
					epoch_loss[d].append(loss_sup[d])			
				
			average = {d:np.around(sum(epoch_loss[d]) / len(epoch_loss[d]) , 3) for d in datasets}
			print(f'{phase} loss: {average} (epoch: {epoch}, step: {itr})', 
						"// Avg time: %.4f s" % (sum(time_taken) / len(time_taken) ))   

			average_epoch_loss[phase] = np.sum([np.mean(epoch_loss[d]) for d in datasets])			
		
		########## Train ends ###############################
#定义模型验证函数，每5轮epoch后进行一次验证
		##### Validation ###############
		if (epoch == 1) or (epoch%5==0): ## validation after every 5 epoch
			for phase in ['val']:

				print("-----", phase ,"- EPOCH", epoch, "-----")
# 开启模型的验证模式
				model.eval()
				# 创建一个字典，记录每个数据集的epoch验证损失
				epoch_val_loss = {d:[] for d in datasets}
        # 遍历每个数据集
				for d in datasets:
					# 初始化时间
					time_taken = []    
# 遍历数据集中的图像和目标
					for itr, (images, targets) in enumerate(loader_val[d]):
 # 开始计时
						start_time = time.time()
 # 将图像和目标移动到GPU
						images = images.cuda()
						targets = targets.cuda()
# 设置梯度计算为不可用
						with torch.set_grad_enabled(False):
 # 获取模型的输出和损失
							seg_output = model(images)
							loss = loss_criterion[d](seg_output, targets)
# 将epoch验证损失添加到列表中
							
							epoch_val_loss[d].append(loss.item())
						
 # 计算运行时间			
						time_taken.append(time.time() - start_time)
# 如果设置了输出损失步数，就在每个步骤输出平均损失
						if args.steps_loss > 0 and (itr % args.steps_loss == 0 or itr == len(loader_val[d])-1):	
							average = np.around(np.mean(epoch_val_loss[d]) , 3)
							print(f'{d}: {phase} loss: {average} (epoch: {epoch}, step: {itr})', 
									"// Avg time/img: %.4f s" % (sum(time_taken) / len(time_taken) / args.batch_size)) 
							
				 # 计算每个数据集的平均epoch验证损失

				average_epoch_loss[phase] = np.sum([np.mean(epoch_val_loss[d]) for d in datasets])

		############# VALIDATION ends #######################
#保存模型
#将模型保存到指定路径
		torch.save(model, f'{savedir}/model_save.pth')
#输出每轮epoch的训练时间
		print("Epoch time {} s".format(time.time() - epoch_start_time))
#记录当前的验证精度，并根据是否为最佳精度来保存checkpoint和best model
		# remember best valIoU and save checkpoint
		current_acc = average_epoch_loss['val']

		is_best = current_acc < best_acc
		best_acc = min(current_acc, best_acc)
		#定义checkpoint和best model的文件名
		filenameCheckpoint = savedir + '/checkpoint.pth.tar'
		filenameBest = savedir + '/model_best.pth.tar'
#调用保存checkpoint和best model的函数，并传入相应参数
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': str(model),
			'state_dict': model.state_dict(),
			'best_acc': best_acc,
			'optimizer' : optimizer.state_dict(),
		}, is_best, filenameCheckpoint, filenameBest)
        
		#SAVE MODEL AFTER EPOCH
		# 在每个 epoch 结束后保存模型
		# 根据 epoch 数量生成模型文件名，例如 model-001.pth
		filename = f'{savedir}/model-{epoch:03}.pth'
	# 定义模型的最佳保存路径
		filenamebest = f'{savedir}/model_best.pth'


# 当 epochs_save 大于 0 且 epoch 大于 0，且 epoch 数量是 epochs_save 的倍数时，保存模型
		if args.epochs_save > 0 and epoch > 0 and epoch % args.epochs_save == 0:
			#torch.save(model.state_dict(), filename)# 保存模型到 filename
			torch.save(model, filename)
			print(f'save: {filename} (epoch: {epoch})')

		if (is_best):
			# 当模型效果更好时，保存模型
			#torch.save(model.state_dict(), filenamebest)
			# 保存模型到 filenamebest
			torch.save(model, filenamebest)
			print(f'save: {filenamebest} (epoch: {epoch})')       

		with open(automated_log_path, "a") as myfile:
			# 打开日志文件并写入以下数据
			iouTrain = 0
			if len(datasets) > 1:
				myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss['train'], average_epoch_loss['val'], usedLr ))
				# 写入 epoch 数量、平均损失、训练集 IoU 和学习率
			else:
				myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss['train'], average_epoch_loss['val'], usedLr ))
				# 写入 epoch 数量、平均损失和学习率
	filename = f'{savedir}/model-final.pth'
	# 定义最终模型保存路径
	torch.save(model, filename)
	return(model)   

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
	# 定义保存检查点的函数，保存模型状态和最佳模型状态
	torch.save(state, filenameCheckpoint)
	if is_best:
		print ("Saving model as best")
		torch.save(state, filenameBest)

def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there# 定义自定义函数，用于在字典中不包含所有键时加载模型
	print(state_dict)
	own_state = model.state_dict()
	state_dict = {k.partition('module.')[2]: v for k,v in state_dict.items()}
	for name, param in state_dict.items():
		
		if name.startswith(('net')):
			continue
		elif name not in own_state:
			print("Not loading {}".format(name))
			continue
		own_state[name].copy_(param)

	print("Loaded pretrained model ... ")
	return model


def main(args, get_dataset):
	# savedir = f'../save_{args.model}/{args.savedir}'
	savedir = f'./save_drnet'

	if os.path.exists(savedir + '/model_best.pth') and not args.resume and not args.finetune:
		print("Save directory already exists ... ")

	if not os.path.exists(savedir):
		os.makedirs(savedir)

	if not args.resume:
		with open(savedir + '/opts.txt', "w") as myfile:
			myfile.write(str(args))

	#Load Model
	assert os.path.exists(args.model + ".py"), f"Error: model definition for {args.model} not found"

  #load pretrained model
	if args.loadmodel:
		model = torch.load('/data/wanghao/Radar_Keypoint_wh01/save_drnet/model_save.pth')
	else:
		model_file = importlib.import_module(args.model)
		model_file.BatchNorm = torch.nn.BatchNorm2d
		NUM_LABELS = get_dataset.num_labels
		model = model_file.Net(NUM_LABELS , args.em_dim , args.resnet)
		copyfile(args.model + ".py", savedir + '/' + args.model + ".py")

	'''if not args.loadmodel:
		state_dict = torch.load('/data1/liwenbo/projects/wang/save_drnet/model_save.pth')
		#model = load_my_state_dict(model, state_dict)
		model.load_state_dict(state_dict.module.state_dict(), strict=False)'''

	
	train_start = time.time()

	model = train(args, get_dataset, model, False)   #Train
	print("========== TRAINING FINISHED ===========")
	print(f"Took {(time.time()-train_start)/60} minutes")


def parse_args():

	parser = ArgumentParser()
	parser.add_argument('--model', type=str, default='drnet')
	parser.add_argument('--debug' , action='store_true')
	parser.add_argument('--basedir', type=str, default='/data/wanghao/Radar_Keypoint_wh01/')
	parser.add_argument('--bnsync' , action='store_true')
	parser.add_argument('--lr' , type=float, default= 1e-3)
	parser.add_argument('--random-rotate' , type=int, default=0)
	parser.add_argument('--random-scale' , type=int, default=0)
	parser.add_argument('--num-epochs', type=int, default=300)
	parser.add_argument('--batch-size', type=int, default=40)   
	parser.add_argument('--savedir', default='')
	parser.add_argument('--datasets' , nargs='+', default=['ELY','MID','LTE'])
	parser.add_argument('--em-dim', type=int, default=100)
	parser.add_argument('--K' , type=float , default=1e-4)
	parser.add_argument('--theta' , type=float , default=0)
	parser.add_argument('--num-samples' , type=int) ## Number of samples from each dataset. If empty, consider full dataset.
	parser.add_argument('--update-embeddings' , type=int , default=0)
	parser.add_argument('--pt-em')
	parser.add_argument('--alpha' , type=int, default=0) ## Cross dataset loss term coeff.
	parser.add_argument('--beta' , type=int , default=0) ## Within dataset loss term coeff. 
	parser.add_argument('--resnet',type=str, default='resnet_18')
	parser.add_argument('--pAcc' , action='store_true')

	### Optional ######
	parser.add_argument('--loadmodel' , action='store_true')#,default=True
	parser.add_argument('--finetune' , action='store_true')
	parser.add_argument('--cuda', action='store_true', default=True)  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
	parser.add_argument('--port', type=int, default=8097)
	parser.add_argument('--height', type=int, default=512)
	parser.add_argument('--num-workers', type=int, default=1)
	parser.add_argument('--steps-loss', type=int, default=5)
	parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
	parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes more time to train otherwise)
	parser.add_argument('--iouVal', action='store_true', default=True)  
	parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  

	args = parser.parse_args()

	return args


if __name__ == '__main__':

	try:
		args = parse_args()
		get_dataset = load_data(args)
		main(args, get_dataset)
	except KeyboardInterrupt:
		sys.exit(0)
