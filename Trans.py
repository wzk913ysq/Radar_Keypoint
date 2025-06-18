import os
import time
import numpy as np
import torch
import sys

from argparse import ArgumentParser

from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import setproctitle as spt

from dataset_loader import *
import transform as transforms

import importlib
from collections import namedtuple

from models.transformer import build_transformer
from models.position_encoding import build_position_encoding

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

class load_data():
	def __init__(self, args):
		## First, a bit of setup
		dinf = namedtuple('dinf', ['name', 'n_labels', 'func', 'path', 'size'])
		self.metadata = [dinf('ELY', 2, FeedLine_EL, 'datasets_copy1', (834, 830)),
						 dinf('MID', 2, FeedLine_MD, 'datasets_copy1', (834, 830)),
						 dinf('LTE', 2, FeedLine_LT, 'datasets_copy1', (834, 830)),
						 ]

		self.num_labels = {entry.name: entry.n_labels for entry in self.metadata if entry.name in args.datasets}

		self.d_func = {entry.name: entry.func for entry in self.metadata}
		basedir = args.basedir
		self.d_path = {entry.name: basedir + entry.path for entry in self.metadata}
		self.d_size = {entry.name: entry.size for entry in self.metadata}

	def __call__(self, name, split='train', num_images=None, mode='labeled', file_path=False):
		transform = self.Img_transform(name, self.d_size[name])
		return self.d_func[name](self.d_path[name], split, transform, file_path, num_images, mode)

	def Img_transform(self, name, size):
		assert (isinstance(size, tuple) and len(size) == 2)
		t = [transforms.Resize(size),
			 transforms.ToTensor()]

		return transforms.Compose(t)


def train(args, get_dataset, model, model1, enc=False):
	model1.eval()
	best_acc = 10000

	num_epochs = 10 if args.debug else args.num_epochs

	n_gpus = torch.cuda.device_count()
	print("\nWorking with {} GPUs".format(n_gpus))

	datasets = args.datasets

	dataset_train = {dname: get_dataset(dname, 'train', args.num_samples) for dname in datasets}
	dataset_val = {dname: get_dataset(dname, 'val',args.num_samples) for dname in datasets}
	# dataset_unlabeled = {dname: get_dataset(dname, co_transform, 'train_extra' , mode='unlabeled') for dname in datasets}
	dataset_unlabeled = {dname: get_dataset(dname, 'train'  , mode='unlabeled') for dname in datasets}

	print("Working with {} Dataset(s):".format(len(datasets)))
	for key in datasets:
		print("{}: Unlabeled images {}, Training on {} images, Validation on {} images".format(key , len(dataset_unlabeled[key]), len(dataset_train[key]) , len(dataset_val[key])))

	for d in datasets:
		if len(set(dataset_train.values())) != 1:
			max_train_size = np.max([ len(dataset_train[dname]) for dname in datasets ]) 
			dataset_train[d].image_paths = dataset_train[d].image_paths*int(np.ceil(float(max_train_size)/len(dataset_train[d].image_paths)))
			dataset_train[d].label_paths = dataset_train[d].label_paths*int(np.ceil(float(max_train_size)/len(dataset_train[d].label_paths)))

	loader_train = {dname:DataLoader(dataset_train[dname], num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True) for dname in datasets}
	loader_val = {dname:DataLoader(dataset_val[dname], num_workers=args.num_workers, batch_size=2, shuffle=True, drop_last=True) for dname in datasets}

	savedir = f'./save_drnet'
	if (enc):
		automated_log_path = savedir + "/automated_log_encoder.txt"
		modeltxtpath = savedir + "/model_encoder.txt"
	else:
		automated_log_path = savedir + "/automated_log.txt"
		modeltxtpath = savedir + "/model.txt"  

	loss_logpath = savedir + "/loss_log.txt"  

	if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
		with open(automated_log_path, "a") as myfile:
			if len(datasets) > 1:
				myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tlearningRate")
			else:
				myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tlearningRate")

	with open(modeltxtpath, "w") as myfile:
		myfile.write(str(model))

	if (not os.path.exists(loss_logpath)):
		with open(loss_logpath , "w") as myfile:
			if len(datasets) > 1:
				myfile.write("Epoch\t\tS1\t\tS2\t\tUS1\t\tUS2\t\tTotal\n")
			else:
				myfile.write("Epoch\t\tS1\t\tS2\t\tTotal\n")

	if args.model == 'drnet':
		optimizer = Adam(model.optim_parameters(), lr=args.lr, weight_decay=1e-4)

	if args.cuda:
		model = torch.nn.DataParallel(model).cuda()

	average_epoch_loss = {'train':np.inf , 'val':np.inf}

	start_epoch = 1
	if args.resume:
		#Must load weights, optimizer, epoch and best value. 
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
		print("=> Loaded checkpoint at epoch {}".format(checkpoint['epoch']))

	scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9))  ## scheduler 2
	#loss_criterion = {key:torch.nn.CrossEntropyLoss().cuda() for key in datasets}
	loss_criterion = {key:torch.nn.MSELoss().cuda() for key in datasets}

	print()
	print("========== STARTING TRAINING ===========")
	print()

	n_iters = min([len(loader_train[d]) for d in datasets])
	
	for epoch in range(start_epoch, num_epochs+1):
		spt.setproctitle('lwb {}/{}'.format(epoch, num_epochs))
		epoch_start_time = time.time()
		usedLr = 0
	
		###### TRAIN begins  #################
		for phase in ['train']:
			print("-----", phase ,"- EPOCH", epoch, "-----")

			scheduler.step(epoch) 
			model.train()   

			for param_group in optimizer.param_groups:
				print("LEARNING RATE: " , param_group['lr'])
				usedLr = float(param_group['lr'])

			## Initialize the iterables
			labeled_iterator = {dname:iter(loader_train[dname]) for dname in datasets}
			epoch_loss = {d:[] for d in datasets}
			time_taken = []    
			loss_sup = {d:0 for d in datasets}
			for itr in range(n_iters):		
				for d in datasets:
					optimizer.zero_grad()

					images_l , targets_l , targets_i= next(labeled_iterator[d])
                              
					images_l = images_l.cuda()
					targets_l = targets_l.cuda()

					start_time = time.time()

					coord_pre = model1(images_l)

					B, C, H, W = images_l.shape
					images_crop = torch.zeros(B, C, 200, W)
					for i in range(B):
						targets_l[i, :, 1] = targets_l[i, :, 1] - coord_pre[i].int() + 100
						images_crop[i, :, :, :] = images_l[i, :, coord_pre[i].int()-100:coord_pre[i].int()+100, :]

					dec_outputs = model(images_crop)
          
					loss_s = loss_criterion[d](dec_outputs[:, :3], targets_l[:, :3])/1000

					loss_s.backward()	

					optimizer.step()

					loss_sup[d] = loss_s.item()
					epoch_loss[d].append(loss_sup[d])

				time_taken.append(time.time() - start_time)		

			average = {d:np.around(sum(epoch_loss[d]) / len(epoch_loss[d]) , 3) for d in datasets}
			print(f'{phase} loss: {average} (epoch: {epoch}, step: {itr})', "// Avg time/img: %.4f s" % (sum(time_taken) / len(time_taken) / args.batch_size))   		

			average = {d:np.mean(epoch_loss[d]) for d in datasets}	
			average_epoch_loss[phase] = sum(average.values())		
		
		########## Train ends ###############################

		##### Validation ###############
		if (epoch == 1) or (epoch%5==0): ## validation after every 5 epoch
			for phase in ['val']:
				print("-----", phase ,"- EPOCH", epoch, "-----")

				model.eval()
				epoch_val_loss = {d:[] for d in datasets}
        
				for d in datasets:
					time_taken = []    

					for itr, (images, targets, targets_i) in enumerate(loader_val[d]):
						start_time = time.time()

						images = images.cuda()
						targets = targets.cuda()

						coord_pre = model1(images)
						B, C, H, W = images.shape
						images_crop = torch.zeros(B, C, 200, W)
						for i in range(B):
							targets[i, :, 1] = targets[i, :, 1] - coord_pre[i].int() + 100
							images_crop[i, :, :, :] = images[i, :, coord_pre[i].int()-100:coord_pre[i].int()+100, :]

						with torch.set_grad_enabled(False):
							seg_output = model(images_crop)
							loss = loss_criterion[d](seg_output[:, :3], targets[:, :3])

							epoch_val_loss[d].append(loss.item())

						time_taken.append(time.time() - start_time)

					average = np.around(np.mean(epoch_val_loss[d]) , 3)
					print(f'{d}: {phase} loss: {average} (epoch: {epoch}, step: {itr})', 
									"// Avg time/img: %.4f s" % (sum(time_taken) / len(time_taken) / args.batch_size)) 		
				
				average_epoch_loss[phase] = np.sum([np.mean(epoch_val_loss[d]) for d in datasets])

		############# VALIDATION ends #######################

		print("Epoch time {} s".format(time.time() - epoch_start_time))

		# remember best valIoU and save checkpoint
		current_acc = average_epoch_loss['val']

		is_best = current_acc < best_acc
		best_acc = min(current_acc, best_acc)
		
		filenameCheckpoint = savedir + '/checkpoint.pth.tar'
		filenameBest = savedir + '/model_best1.pth.tar'

		save_checkpoint({
			'epoch': epoch + 1,
			'arch': str(model),
			'state_dict': model.state_dict(),
			'best_acc': best_acc,
			'optimizer' : optimizer.state_dict(),
		}, is_best, filenameCheckpoint, filenameBest)

		#SAVE MODEL AFTER EPOCH
		
		filename = f'{savedir}/model-{epoch:03}.pth'
		filenamebest = f'{savedir}/model_best1.pth'

		if args.epochs_save > 0 and epoch > 0 and epoch % args.epochs_save == 0:
			#torch.save(model.state_dict(), filename)
			torch.save(model, filename)
			print(f'save: {filename} (epoch: {epoch})')

		if (is_best):
			#torch.save(model.state_dict(), filenamebest)
			torch.save(model, filenamebest)
			print(f'save: {filenamebest} (epoch: {epoch})')       

		with open(automated_log_path, "a") as myfile:
			if len(datasets) > 1:
				myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss['train'], average_epoch_loss['val'], usedLr ))
			else:
				myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss['train'], average_epoch_loss['val'], usedLr ))

	filename = f'{savedir}/model-final1.pth'
	torch.save(model, filename)

	return(model)   

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
	torch.save(state, filenameCheckpoint)
	if is_best:
		print ("Saving model as best")
		torch.save(state, filenameBest)


def main(args, get_dataset):
	# savedir = f'../save_{args.model}/{args.savedir}'
	savedir = f'./save_drnet'

	if os.path.exists(savedir + '/model_best.pth') and not args.resume and not args.finetune:
		print("Save directory already exists ... ")
		sys.exit(0)

	if not os.path.exists(savedir):
		os.makedirs(savedir)

	if not args.resume:
		with open(savedir + '/opts.txt', "w") as myfile:
			myfile.write(str(args))

	#Load Model

    #load pretrained model
	if args.loadmodel:
		model = torch.load('/data1/liwenbo/projects/wang/model_save/model_without_process_256_new.pth').module
	else:
		trans = build_transformer(args)
		pos = build_position_encoding(args)
		model_file = importlib.import_module('.'+args.model, 'models')
		NUM_LABELS = get_dataset.num_labels
		model = model_file.Net(NUM_LABELS, args.em_dim, args.resnet, trans, pos)		
	
	# 预测料面区域的模型？在哪里定义的？-----box_pre_net.py   class Net1(nn.Module)  ？？？
	model1 = torch.load('/data1/liwenbo/projects/wang/model_save/area_pre.pth').module

	train_start = time.time()

	model = train(args, get_dataset, model, model1, False)   #Train
	print("========== TRAINING FINISHED ===========")
	print(f"Took {(time.time()-train_start)/60} minutes")


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--model', type=str, default='drnet')
	parser.add_argument('--debug' , action='store_true')
	parser.add_argument('--basedir', type=str, default='/data1/liwenbo/projects/wang/')
	parser.add_argument('--bnsync' , action='store_true')
	parser.add_argument('--lr' , type=float, default= 1e-4)
	parser.add_argument('--random-rotate' , type=int, default=0)
	parser.add_argument('--random-scale' , type=int, default=0)
	parser.add_argument('--num-epochs', type=int, default=1000)
	parser.add_argument('--batch-size', type=int, default=8)
	parser.add_argument('--savedir', default='')
	parser.add_argument('--datasets' , nargs='+', default=['ELY','MID','LTE'])
	parser.add_argument('--em-dim', type=int, default=100)
	parser.add_argument('--K' , type=float , default=1e4)
	parser.add_argument('--theta' , type=float , default=0)
	parser.add_argument('--num-samples' , type=int) ## Number of samples from each dataset. If empty, consider full dataset.
	parser.add_argument('--update-embeddings' , type=int , default=0)
	parser.add_argument('--pt-em')
	parser.add_argument('--alpha' , type=int, default=0) ## Cross dataset loss term coeff.
	parser.add_argument('--beta' , type=int , default=0) ## Within dataset loss term coeff. 
	parser.add_argument('--resnet',type=str, default='resnet_18')
	parser.add_argument('--pAcc' , action='store_true')

	### Optional ######
	parser.add_argument('--loadmodel' , action='store_true' , default=False)
	parser.add_argument('--finetune' , action='store_true')
	parser.add_argument('--cuda', action='store_true', default=True)  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
	parser.add_argument('--state')
	parser.add_argument('--port', type=int, default=8097)
	parser.add_argument('--height', type=int, default=512)
	parser.add_argument('--num-workers', type=int, default=1)
	parser.add_argument('--steps-loss', type=int, default=5)
	parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
	parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes more time to train otherwise)
	parser.add_argument('--iouVal', action='store_true', default=True)  
	parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  
	parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
	parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
	parser.add_argument('--dim_feedforward', default=128, type=int,
						help="Intermediate size of the feedforward layers in the transformer blocks")
	parser.add_argument('--hidden_dim', default=64, type=int,
						help="Size of the embeddings (dimension of the transformer)")
	parser.add_argument('--dropout', default=0.1, type=float,
						help="Dropout applied in the transformer")
	parser.add_argument('--nheads', default=8, type=int,
						help="Number of attention heads inside the transformer's attentions")
	parser.add_argument('--num_queries', default=8, type=int,
						help="Number of query slots")
	parser.add_argument('--pre_norm', action='store_true')
	parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

	args = parser.parse_args()

	return args


if __name__ == '__main__':
	try:
		args = parse_args()
		get_dataset = load_data(args)
		main(args, get_dataset)
	except KeyboardInterrupt:
		sys.exit(0)
