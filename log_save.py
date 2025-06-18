import pickle
import torch
import glob
from argparse import ArgumentParser
import os
import shutil


import matplotlib.pyplot as plt
import numpy as np

def parse_args():

	parser = ArgumentParser()
	parser.add_argument('--model', type=str, default='drnet')
	parser.add_argument('--debug' , action='store_true')
	parser.add_argument('--basedir', type=str, default='/data/wenzhaokun/Radar_Keypoint_wh01/Radar_Keypoint_wh01/')
	parser.add_argument('--bnsync' , action='store_true')
	parser.add_argument('--lr' , type=float, default= 0.001)
	parser.add_argument('--random-rotate' , type=int, default=0)
	parser.add_argument('--random-scale' , type=int, default=0)
	parser.add_argument('--num-epochs', type=int, default=300)
	parser.add_argument('--batch-size', type=int, default=40)   
	parser.add_argument('--savedir', default='')
	#parser.add_argument('--datasets' , nargs='+', default=['ELY','MID','LTE'])
	parser.add_argument('--datasets' , nargs='+', default=['ELY'])

	parser.add_argument('--em-dim', type=int, default=100)
	parser.add_argument('--K' , type=float , default=1e-4)
	parser.add_argument('--theta' , type=float , default=0)
	parser.add_argument('--num-samples' , type=int, default=None) ## Number of samples from each dataset for train. If empty, consider full dataset.the number of train:val=7:3
	parser.add_argument('--update-embeddings' , type=int , default=0)
	parser.add_argument('--pt-em')  
	parser.add_argument('--alpha' , type=int, default=0) ## Cross dataset loss term coeff.
	parser.add_argument('--beta' , type=int , default=0) ## Within dataset loss term coeff. 
	parser.add_argument('--resnet',type=str, default='resnet_18')
	parser.add_argument('--pAcc' , action='store_true')
	

	### Optional ######
	parser.add_argument('--loadmodel' , type=bool,default=True)
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
	parser.add_argument('--pretrainweight', type=str, default='olddatasets_state_dict_save.pth')
	parser.add_argument('--information', type=str, default='test1')  
  
	parser.add_argument('--useautoencoder', type=bool, default=False) 
	parser.add_argument('--usetransfer' , action='store_true',default=True) 

	  


	args = parser.parse_args()

	return args

def save_myfiles(args):
	print(f"自动保存{args.information}的训练文件")
	pathroot = "/data/wenzhaokun/Radar_Keypoint_wh01/Radar_Keypoint_wh01"
	newfilepath = os.path.join(pathroot,"train_log",args.information)
	if not os.path.exists(newfilepath):
		os.makedirs(newfilepath)
		with open (os.path.join(newfilepath,"train_loss"),"a") as file :
				file.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tlearningRate\t\tinformation\n")#写入loss表头
		#os.makedirs(os.path.join(newfilepath,"args.txt"))#创建保存参数的文件
		#os.makedirs(os.path.join(newfilepath,"result_figures"))#创建保存结果图的文件夹
		open(os.path.join(newfilepath,"args.txt"), "w").close#创建保存参数的文件

	with open('save_drnet/automated_log.txt', 'r') as file_a:
		lines = file_a.readlines()
	last_lines = lines[-(args.num_epochs):]
	with open(os.path.join(newfilepath,"train_loss"),"a+") as myloss_log:
		myloss_log.writelines(last_lines)#写入loss_log



	with open(os.path.join(newfilepath,"args.txt"),"w") as argsfile :
		for k,v in sorted(vars(args).items()):
				#print(k,'=',v)
				argsfile.write(f"{k}={v}\n")#写入参数文件
		with open(os.path.join(pathroot,"save_drnet","model.txt"),"r") as f :
				model_structure = f.read()
		argsfile.write(model_structure)
		
	shutil.copy(os.path.join(pathroot,"Feedline_Keypoint.py"), newfilepath)
	shutil.copy(os.path.join(pathroot,"drnet.py"), newfilepath)
	if os.path.exists (os.path.join(newfilepath,"result_figures")) :
		shutil.rmtree(os.path.join(newfilepath,"result_figures"))
	shutil.copytree(os.path.join(pathroot, "result_save", f"{args.information}_test_result"),\
				os.path.join(newfilepath,"result_figures"))
      
def draw_loss(information):  
	losspath = os.path.join('train_log',str(information),'train_loss')
	index_list = []
	train_list = []
	test_list = []
	with open (losspath, "+r") as f :
		lossline = f.readlines()
		for loss in lossline[1:] :
			index = int(loss.split("\t\t")[0])
			index_list.append(index)
			
			train_loss = float(loss.split("\t\t")[1])
			train_list.append(train_loss)
			test_loss = float(loss.split("\t\t")[2])
			test_list.append(test_loss)

	plt.plot(index_list,train_list,color = "red",linewidth=1.0,label = "train_loss")
	plt.plot(index_list,test_list,color = "blue",linewidth=1.0, label = "val_loss")

	
	# #设置坐标轴名称
	plt.xlabel('epochs')
	plt.ylabel('loss')


	plt.legend()
	plt.savefig(os.path.join('train_log',str(information),f'{information}_loss_figure.png'))
	plt.show()
	
    
		
      
      
            

if __name__ == '__main__':
    # args = parse_args()
    # save_myfiles(args)
    draw_loss("test1")