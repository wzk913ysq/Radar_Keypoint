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

        ## First, a bit of setup
        dinf = namedtuple('dinf', ['name', 'n_labels', 'func', 'path', 'size'])
        self.metadata = [dinf('ELY', 8, FeedLine_EL, 'datasets', (834, 830)),
                         dinf('MID', 8, FeedLine_MD, 'datasets', (834, 830)),
                         dinf('LTE', 8, FeedLine_LT, 'datasets', (834, 830)),
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
    best_acc = 2000

    num_epochs = 10 if args.debug else args.num_epochs
    

    n_gpus = torch.cuda.device_count()
    print("\nWorking with {} GPUs".format(n_gpus))

    datasets = args.datasets

    NUM_LABELS = get_dataset.num_labels

    dataset_train = {dname: get_dataset(dname, 'train', args.num_samples) for dname in datasets}
    dataset_val = {dname: get_dataset(dname, 'val',args.num_samples) for dname in datasets}
    # dataset_unlabeled = {dname: get_dataset(dname, co_transform, 'train_extra' , mode='unlabeled') for dname in datasets}
    dataset_unlabeled = {dname: get_dataset(dname, 'train'  , mode='unlabeled') for dname in datasets}

    print("Working with {} Dataset(s):".format(len(datasets)))
    for key in datasets:
        print("{}: Unlabeled images {}, Training on {} images, Validation on {} images".format(key , len(dataset_unlabeled[key]), len(dataset_train[key]) , len(dataset_val[key])))

    for d in datasets:
        if len(set(dataset_train.values())) != 1:
            max_train_size = np.max([ len(dataset_train[dname]) for dname in datasets]) 
            dataset_train[d].image_paths = dataset_train[d].image_paths*int(np.ceil(float(max_train_size)/len(dataset_train[d].image_paths)))
            dataset_train[d].label_paths = dataset_train[d].label_paths*int(np.ceil(float(max_train_size)/len(dataset_train[d].label_paths)))

        
    loader_train = {dname:DataLoader(dataset_train[dname], num_workers=args.num_workers, batch_size=args.batch_size, 
                            shuffle=True) for dname in datasets}
    loader_val = {dname:DataLoader(dataset_val[dname], num_workers=args.num_workers, batch_size=1, 
                            shuffle=True, drop_last=True) for dname in datasets}

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
        #optimizer = SGD(model.optim_parameters(), args.lr, 0.9,  weight_decay=1e-4)
        optimizer = Adam(model.optim_parameters(), lr=args.lr, weight_decay=1e-4)
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    le_file = savedir + '/label_embedding.pt'
    average_epoch_loss = {'train':np.inf , 'val':np.inf}

    label_embedding = {key:torch.randn(NUM_LABELS[key] , args.em_dim).cuda() for key in datasets} ## Random Initialization

    ## If provided, use label embedddings
    if args.pt_em:
        fn = torch.load(args.pt_em)
        label_embedding = {key : torch.tensor(fn[key] , dtype=torch.float).cuda() for key in datasets}

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


        model.load_state_dict(checkpoint['state_dict'],False)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        label_embedding = torch.load(le_file) if len(datasets) >1 else None
        print("=> Loaded checkpoint at epoch {}".format(checkpoint['epoch']))

    
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9))  ## scheduler 2
    #loss_criterion = {key:torch.nn.CrossEntropyLoss().cuda() for key in datasets}
    loss_criterion = {key:torch.nn.MSELoss().cuda() for key in datasets}
 
    if len(datasets)>1:
        similarity_module = EmbeddingLoss(NUM_LABELS, args.em_dim, label_embedding, loss_criterion)
        similarity_module = torch.nn.DataParallel(similarity_module).cuda()
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

            for param_group in optimizer.param_groups:
                print("LEARNING RATE: " , param_group['lr'])
                usedLr = float(param_group['lr'])

            epoch_loss = {d:[] for d in datasets}
            time_taken = []    

            for d in datasets:
                    #查找问题样本
                '''for d in ['ELY']:
                    dataset = get_dataset(d)
                    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
                    for idx, (images_l, targets_l) in enumerate(dataloader):
                        try:
                            print(idx)
                            pass
                        except Exception as e:
                            print(f"Error occurred in dataset {d}, batch {idx}: {str(e)}")
                            continue'''
                
                loss_sup = {d:0 for d in datasets}
                for itr, (images_l , targets_l) in enumerate(loader_train[d]):

                    optimizer.zero_grad()
                              
                    images_l = images_l.cuda()
                    targets_l = targets_l.cuda()

                    start_time = time.time()

                    coord_pre = model1(images_l)    # 预测一个料面区域

                    B, C, H, W = images_l.shape
                    images_crop = torch.zeros(B, C, 200, W).cuda()
                    for i in range(B):
                        targets_l[i, :, 1] = targets_l[i, :, 1] - coord_pre[i].int() + 100
                        images_crop[i, :, :, :] = images_l[i, :, coord_pre[i].int()-100:coord_pre[i].int()+100, :]

                    dec_outputs = model(images_crop)  # 预测关键点坐标
                    
                    torchvision.utils.save_image(images_crop, os.path.join('/data/wanghao/Radar_Keypoint_wh01/result_save/test_result', f'show.jpg'))
                    

                    loss_s = loss_criterion[d](dec_outputs, targets_l)
     
                    loss_s.backward()	

                    optimizer.step()

                    time_taken.append(time.time() - start_time)

                    
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
                            for i in range(8):
                                if dec_outputs.int()[bat, i, 1] < 825 and dec_outputs.int()[bat, i, 0] < 412 :
                                    colored_tensor[:, targets_l.int()[bat, i, 1]-8:targets_l.int()[bat, i, 1]+8, targets_l.int()[bat, i, 0]-8:targets_l.int()[bat, i, 0]+8] = 0
                                    colored_tensor[:, dec_outputs.int()[bat, i, 1]-8:dec_outputs.int()[bat, i, 1]+8, dec_outputs.int()[bat, i, 0]-8:dec_outputs.int()[bat, i, 0]+8] = 0
                                    colored_tensor[1, targets_l.int()[bat, i, 1]-8:targets_l.int()[bat, i, 1]+8, targets_l.int()[bat, i, 0]-8:targets_l.int()[bat, i, 0]+8] = 1
                                    colored_tensor[:2 , dec_outputs.int()[bat, i, 1]-8:dec_outputs.int()[bat, i, 1]+8, dec_outputs.int()[bat, i, 0]-1:dec_outputs.int()[bat, i, 0]+1] = 1
                                    colored_tensor[:2 , dec_outputs.int()[bat, i, 1]-1:dec_outputs.int()[bat, i, 1]+1, dec_outputs.int()[bat, i, 0]-8:dec_outputs.int()[bat, i, 0]+8] = 1
                            img = torchvision.transforms.ToPILImage()(torch.Tensor.cpu(colored_tensor))
                            img_draw = ImageDraw.Draw(img)
                            index = dec_outputs[bat].view(16).int().tolist()
                            img_draw.line(index,fill = (255, 0, 0),width = 3)
                            colored_tensor = torchvision.transforms.ToTensor()(img)
                            torchvision.utils.save_image(colored_tensor, os.path.join('/data/wanghao/Radar_Keypoint_wh01/result_save/test_result', f'train_{d}_{imcount}.jpg'))
                            imcount += 1        
                    #visualization   
                    '''if args.steps_loss > 0 and itr % args.steps_loss == 0 :                                
                
                        imcount = 0
                        
                        for bat in range(B):
                            colored_tensor = torch.zeros(3, H, W)
                            colored_tensor = images_l[bat, :, :, :] 
                            for i in range(8):
                                if dec_outputs.int()[bat, i, 1] < 825 and dec_outputs.int()[bat, i, 0] < 412 :
                                    colored_tensor[:, targets_l.int()[bat, i, 1]-8:targets_l.int()[bat, i, 1]+8, targets_l.int()[bat, i, 0]-8:targets_l.int()[bat, i, 0]+8] = 0
                                    colored_tensor[:, dec_outputs.int()[bat, i, 1]-8:dec_outputs.int()[bat, i, 1]+8, dec_outputs.int()[bat, i, 0]-8:dec_outputs.int()[bat, i, 0]+4] = 0
                                    colored_tensor[1, targets_l.int()[bat, i, 1]-8:targets_l.int()[bat, i, 1]+8, targets_l.int()[bat, i, 0]-8:targets_l.int()[bat, i, 0]+8] = 1
                                    colored_tensor[:2 , dec_outputs.int()[bat, i, 1]-8:dec_outputs.int()[bat, i, 1]+8, dec_outputs.int()[bat, i, 0]-1:dec_outputs.int()[bat, i, 0]+1] = 1
                                    colored_tensor[:2 , dec_outputs.int()[bat, i, 1]-1:dec_outputs.int()[bat, i, 1]+1, dec_outputs.int()[bat, i, 0]-4:dec_outputs.int()[bat, i, 0]+4] = 1
                            img = torchvision.transforms.ToPILImage()(colored_tensor)
                            img_draw = ImageDraw.Draw(img)
                            index = dec_outputs[bat].view(16).int().tolist()
                            img_draw.line(index,fill = (255, 0, 0),width = 3)
                            colored_tensor = torchvision.transforms.ToTensor()(img)
                            torchvision.utils.save_image(colored_tensor, os.path.join('/data/wanghao/Radar_Keypoint_wh01/result_save/test_result/', f'train_{d}_{imcount}.jpg'))
                            imcount += 1'''

                    loss_sup[d] = loss_s.item()
                    epoch_loss[d].append(loss_sup[d])			
                
            average = {d:np.around(sum(epoch_loss[d]) / len(epoch_loss[d]) , 3) for d in datasets}
            print(f'{phase} loss: {average} (epoch: {epoch}, step: {itr})', 
                        "// Avg time: %.4f s" % (sum(time_taken) / len(time_taken) ))   

            average_epoch_loss[phase] = np.sum([np.mean(epoch_loss[d]) for d in datasets])			
        
        ########## Train ends ###############################

        ##### Validation ###############
        if (epoch == 1) or (epoch%5==0): ## validation after every 5 epoch
            for phase in ['val']:

                print("-----", phase ,"- EPOCH", epoch, "-----")

                model.eval()
                epoch_val_loss = {d:[] for d in datasets}
        
                for d in datasets:
                    time_taken = []    

                    for itr, (images, targets) in enumerate(loader_val[d]):

                        start_time = time.time()

                        images = images.cuda()
                        targets = targets.cuda()
                        coord_pre = model1(images_l)    # 预测一个料面区域

                        B, C, H, W = images_l.shape
                        images_crop = torch.zeros(B, C, 200, W).cuda()
                        for i in range(B):
                            targets_l[i, :, 1] = targets_l[i, :, 1] - coord_pre[i].int() + 100
                            images_crop[i, :, :, :] = images_l[i, :, coord_pre[i].int()-100:coord_pre[i].int()+100, :]

                        with torch.set_grad_enabled(False):

                            seg_output = model(images_crop)
                            loss = loss_criterion[d](seg_output, targets)

                            epoch_val_loss[d].append(loss.item())

                        time_taken.append(time.time() - start_time)

                        if args.steps_loss > 0 and (itr % args.steps_loss == 0 or itr == len(loader_val[d])-1):
                            average = np.around(np.mean(epoch_val_loss[d]) , 3)
                            print(f'{d}: {phase} loss: {average} (epoch: {epoch}, step: {itr})', 
                                    "// Avg time/img: %.4f s" % (sum(time_taken) / len(time_taken) / args.batch_size)) 
                            
                
                average_epoch_loss[phase] = np.sum([np.mean(epoch_val_loss[d]) for d in datasets])

        ############# VALIDATION ends #######################

        torch.save(model, f'{savedir}/model_save.pth')

        print("Epoch time {} s".format(time.time() - epoch_start_time))

        # remember best valIoU and save checkpoint
        current_acc = average_epoch_loss['val']

        is_best = current_acc < best_acc
        best_acc = min(current_acc, best_acc)
        
        filenameCheckpoint = savedir + '/checkpoint.pth.tar'
        filenameBest = savedir + '/model_best.pth.tar'

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)


        #SAVE MODEL AFTER EPOCH
        
        filename = f'{savedir}/model-{epoch:03}.pth'
        filenamebest = f'{savedir}/model_best.pth'

        if args.epochs_save > 0 and epoch > 0 and epoch % args.epochs_save == 0:
            #torch.save(model.state_dict(), filename)
            torch.save(model, filename)
            print(f'save: {filename} (epoch: {epoch})')

        if (is_best):
            #torch.save(model.state_dict(), filenamebest)
            torch.save(model, filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')       

        with open(automated_log_path, "a") as myfile:
            iouTrain = 0
            if len(datasets) > 1:
                myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss['train'], average_epoch_loss['val'], usedLr ))
            else:
                myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss['train'], average_epoch_loss['val'], usedLr ))
    filename = f'{savedir}/model-final.pth'
    torch.save(model, filename)
    return(model)   

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)

def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
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

class config:
    def __init__(self):
        self.datasets = ['ELY', 'MID', 'LTE']
        self.basedir = '/data/wanghao/Radar_Keypoint_wh01/'
        self.batch_size = 8
        
        self.model = 'drnet'
        self.debug = False
        self.bnsync = 'store_true'
        self.lr = 5e-3
        self.random_rotate = 0
        self.random_scale = 0
        self.num_epochs = 300
        # parser.add_argument('--batch-size', type=int, default=20)
        self.savedir  = ''
        # parser.add_argument('--datasets' , nargs='+', default=['ELY','MID','LTE'])
        
        self.em_dim = 100
        self.K = 1e-4
        self.theta = 0
        self.num_samples = 374
        self.update_embeddings = 0
        self.pt_em = 0
        self.alpha =0 ## Cross dataset loss term coeff.
        self.beta = 0 ## Within dataset loss term coeff. 
        self.resnet = 'resnet_18'
        self.pAcc = 'store_true'

        ### Optional ######
        self.loadmodel = False
        self.finetune =  'store_true'
        self.cuda = True ## action='store_true', default=True)  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
        self.port = 8097
        self.height = 512
        self.num_workers = 1
        self.steps_loss =5
        self.epochs_save =0   #You can use this value to save model every X epochs
        self.iouTrain = False # action='store_true', default=False) #recommended: False (takes more time to train otherwise)
        self.iouVal = True # action='store_true', default=True)  
        self.resume = False # action='store_true')    #Use this flag to load last checkpoint for training  

if __name__ == '__main__':
    # try:	
    # 	main(args, get_dataset)
    # except KeyboardInterrupt:
    # 	sys.exit(0)
  
    args = config()
    get_dataset = load_data(args)
    # savedir = f'../save_{args.model}/{args.savedir}'
    savedir = f'./save_drnet/{args.savedir}'

    if os.path.exists(savedir + '/model_best.pth') and not args.resume and not args.finetune:
        print("Save directory already exists ... ")
        sys.exit(0)

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if not args.resume:
        with open(savedir + '/opts.txt', "w") as myfile:
            myfile.write(str(args))

    #Load Model
    assert os.path.exists(args.model + ".py"), f"Error: model definition for {args.model} not found"

    #load pretrained model
    if args.loadmodel:
        model = torch.load('/data/wanghao/Radar_Keypoint_wh01/save_drnet//model_save.pth').module
    else:
        model_file = importlib.import_module(args.model)
        # if args.bnsync:
        #     model_file.BatchNorm = batchnormsync.BatchNormSync
        # else:
        #     model_file.BatchNorm = torch.nn.BatchNorm2d
        model_file.BatchNorm = torch.nn.BatchNorm2d
        NUM_LABELS = get_dataset.num_labels
        
        # 采用Transformer结构的模型
        model = model_file.Net(NUM_LABELS , args.em_dim , args.resnet)		
        copyfile(args.model + ".py", savedir + '/' + args.model + ".py")

    # 先检测出一个包含料面的小区域，然后再检测料面的关键点
    model1 = torch.load('/data/wanghao/Radar_Keypoint_wh01/model_save/area_pre.pth').module
    
    print("========== TRAINING STARTS ===========")
    train_start = time.time()
    model = train(args, get_dataset, model, model1, False)   #Train
    print("========== TRAINING FINISHED ===========")
    print(f"Took {(time.time()-train_start)/60} minutes")    
