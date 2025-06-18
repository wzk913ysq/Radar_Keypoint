import os
import time
import numpy as np
import torch
import sys
import importlib

import torchvision
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset

import ImageProcess as ip
from PIL import Image, ImageOps, ImageDraw
from collections import OrderedDict , namedtuple
from shutil import copyfile

#------------user defined...
import transform as transforms
from iouEval import iouEval#, getColorEntry
from dataset_loader_wh01 import *
from EntropyLoss import EmbeddingLoss

# import setproctitle as spt
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

##################################################### train
def train(args, get_dataset, model, enc=False):
    savedir = f'./save_drnet'
    best_acc = 2000
    num_epochs = 10 if args.debug else args.num_epochs
    
    #--------------------------------------------------------------------
    datasets = args.datasets
    NUM_LABELS = get_dataset.num_labels

    dataset_train = {dname: get_dataset(dname, 'train') for dname in datasets}
    dataset_val = {dname: get_dataset(dname, 'val') for dname in datasets}
    dataset_unlabeled = {dname: get_dataset(dname, 'train', mode='unlabeled') for dname in datasets}

    print("Working with {} Dataset(s):".format(len(datasets)))
    for key in datasets:
        print("{}: Unlabeled images {}, Training on {} images, Validation on {} images".format(key, len(dataset_unlabeled[key]), len(dataset_train[key]) , len(dataset_val[key])))

    for d in datasets:
        if len(set(dataset_train.values())) != 1:
            max_train_size = np.max([len(dataset_train[dname]) for dname in datasets]) 
            dataset_train[d].image_paths = dataset_train[d].image_paths*int(np.ceil(float(max_train_size)/len(dataset_train[d].image_paths)))
            dataset_train[d].label_paths = dataset_train[d].label_paths*int(np.ceil(float(max_train_size)/len(dataset_train[d].label_paths)))
        
    loader_train = {dname:DataLoader(dataset_train[dname], num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True) for dname in datasets}
    loader_val = {dname:DataLoader(dataset_val[dname], num_workers=args.num_workers, batch_size=1, shuffle=True, drop_last=True) for dname in datasets}

    #--------------------------------------------------------------------
    if enc:
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"  

    if not os.path.exists(automated_log_path):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            if len(datasets) > 1:
                myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tlearningRate")
            else:
                myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    #--------------------------------------------------------------------
    loss_logpath = savedir + "/loss_log.txt"  
    if not os.path.exists(loss_logpath):
        with open(loss_logpath , "w") as myfile:
            if len(datasets) > 1:
                myfile.write("Epoch\t\tS1\t\tS2\t\tUS1\t\tUS2\t\tTotal\n")
            else:
                myfile.write("Epoch\t\tS1\t\tS2\t\tTotal\n")

    #--------------------------------------------------------------------
    if args.model == 'drnet':
        #optimizer = SGD(model.optim_parameters(), args.lr, 0.9,  weight_decay=1e-4)
        optimizer = Adam(model.optim_parameters(), lr=args.lr, weight_decay=1e-4)
    
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
        label_embedding = torch.load(le_file) if len(datasets) >1 else None
        print("=> Loaded checkpoint at epoch {}".format(checkpoint['epoch']))

    #--------------------------------------------------------------------
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9))  ## scheduler 2
    
    le_file = savedir + '/label_embedding.pt'
    if args.cuda:
        label_embedding = {key: torch.randn(NUM_LABELS[key], args.em_dim).cuda() for key in datasets} ## Random Initialization
    else:
        label_embedding = {key: torch.randn(NUM_LABELS[key], args.em_dim) for key in datasets} ## Random Initialization

    ## If provided, use label embedddings
    if args.pt_em:
        fn = torch.load(args.pt_em)
        if args.cuda:
            label_embedding = {key: torch.tensor(fn[key], dtype=torch.float).cuda() for key in datasets}
        else:
            label_embedding = {key: torch.tensor(fn[key], dtype=torch.float) for key in datasets}
    
    #--------------------------------------------------------------------
    if args.cuda:
        n_gpus = torch.cuda.device_count()
        print("\nWorking with {} GPUs".format(n_gpus))
        model = torch.nn.DataParallel(model).cuda()
        
        #loss_criterion = {key: torch.nn.CrossEntropyLoss().cuda() for key in datasets}
        loss_criterion = {key: torch.nn.MSELoss().cuda() for key in datasets}
        
        if len(datasets)>1:
            similarity_module = EmbeddingLoss(NUM_LABELS, args.em_dim, label_embedding, loss_criterion)
            similarity_module = torch.nn.DataParallel(similarity_module).cuda()
    else:
        #loss_criterion = {key: torch.nn.CrossEntropyLoss() for key in datasets}
        loss_criterion = {key: torch.nn.MSELoss() for key in datasets}
        
        if len(datasets)>1:
            similarity_module = EmbeddingLoss(NUM_LABELS, args.em_dim, label_embedding, loss_criterion)
    
    torch.save(label_embedding , le_file)

    print()
    print("========== STARTING TRAINING ===========")
    print()

    average_epoch_loss = {'train':np.inf , 'val':np.inf}
    start_epoch = 1
    for epoch in range(start_epoch, num_epochs+1):
        if args.cuda:
            # spt.setproctitle('wh {}/{}'.format(epoch, num_epochs))
            pass
        
        epoch_start_time = time.time()
        usedLr = 0
    
        ###### TRAIN begins  #################
        for phase in ['train']:
            print("-----", phase, "- EPOCH", epoch, "-----")
            optimizer.step()
            scheduler.step()    
            #-------------------------------
            model.train()
            
            #-------------------------------
            for param_group in optimizer.param_groups:
                print("LEARNING RATE: " , param_group['lr'])
                usedLr = float(param_group['lr'])

            epoch_loss = {d:[] for d in datasets}
            time_taken = []    
            for d in datasets:
                loss_sup = {d:0 for d in datasets}
                for itr, (images_l, targets_l) in enumerate(loader_train[d]):
                    optimizer.zero_grad()
                    if args.cuda:
                        images_l = images_l.cuda()
                        targets_l = targets_l.cuda()

                    #-----------------------------------------
                    start_time = time.time()
                    dec_outputs = model(images_l)
                    #-----------------------------------------
                    loss_s = loss_criterion[d](dec_outputs[d], targets_l/20)
                    loss_sup[d] = loss_s.item()
                    epoch_loss[d].append(loss_sup[d])		
                    loss_s.backward()	
                    
                    time_taken.append(time.time() - start_time)
                            
                    # #visualization   
                    # if args.steps_loss > 0 and itr % args.steps_loss == 0 :                                
                    #     B, C, H, W = images_l.shape
                    #     imcount = 0
                        
                    #     for bat in range(B):
                    #         colored_tensor = torch.zeros(3, H, W)
                    #         colored_tensor = images_l[bat, :, :, :] 
                    #         for i in range(4):
                    #             if dec_outputs[d].int()[bat, i, 1] < 825 and dec_outputs[d].int()[bat, i, 0] < 412 :
                    #                 colored_tensor[:, targets_l.int()[bat, i, 1]-4:targets_l.int()[bat, i, 1]+4, targets_l.int()[bat, i, 0]-4:targets_l.int()[bat, i, 0]+4] = 0
                    #                 colored_tensor[:, dec_outputs[d].int()[bat, i, 1]-4:dec_outputs[d].int()[bat, i, 1]+4, dec_outputs[d].int()[bat, i, 0]-4:dec_outputs[d].int()[bat, i, 0]+4] = 0
                    #                 colored_tensor[1, targets_l.int()[bat, i, 1]-4:targets_l.int()[bat, i, 1]+4, targets_l.int()[bat, i, 0]-4:targets_l.int()[bat, i, 0]+4] = 1
                    #                 colored_tensor[:2 , dec_outputs[d].int()[bat, i, 1]-4:dec_outputs[d].int()[bat, i, 1]+4, dec_outputs[d].int()[bat, i, 0]-1:dec_outputs[d].int()[bat, i, 0]+1] = 1
                    #                 colored_tensor[:2 , dec_outputs[d].int()[bat, i, 1]-1:dec_outputs[d].int()[bat, i, 1]+1, dec_outputs[d].int()[bat, i, 0]-4:dec_outputs[d].int()[bat, i, 0]+4] = 1
                    #         img = torchvision.transforms.ToPILImage()(colored_tensor)
                    #         img_draw = ImageDraw.Draw(img)
                    #         index = dec_outputs[d][bat].view(8).int().tolist()
                    #         img_draw.line(index,fill = (255, 0, 0),width = 3)
                    #         colored_tensor = torchvision.transforms.ToTensor()(img)
                    #         torchvision.utils.save_image(colored_tensor, os.path.join('./result_save/test_result', f'train_{d}_{imcount}.jpg'))
                    #         imcount += 1
                    
            average = {d: np.around(sum(epoch_loss[d]) / len(epoch_loss[d]), 3) for d in datasets}
            print(f'{phase} loss: {average} (epoch: {epoch}, step: {itr})', 
                        "// Avg time: %.4f s" % (sum(time_taken) / len(time_taken) ))   

            average_epoch_loss[phase] = np.sum([np.mean(epoch_loss[d]) for d in datasets])			
        ########## Train ends ###############################

        ##### Validation ###############
        if (epoch > 1) or (epoch % 5 == 0): ## validation after every 5 epoch
            for phase in ['val']:
                print("-----", phase ,"- EPOCH", epoch, "-----")

                model.eval()
                
                epoch_val_loss = {d:[] for d in datasets}
                for d in datasets:
                    time_taken = []    
                    for itr, (images, targets, targets_i) in enumerate(loader_val[d]):
                        start_time = time.time()
                        if args.cuda:
                            images = images.cuda()
                            targets = targets.cuda()

                        with torch.set_grad_enabled(False):
                            #--------------------------------
                            seg_output = model(images)
                            #--------------------------------
                            loss = loss_criterion[d](seg_output[d], targets)
                            epoch_val_loss[d].append(loss.item())

                        time_taken.append(time.time() - start_time)

                        if args.steps_loss > 0 and (itr % args.steps_loss == 0 or itr == len(loader_val[d])-1):
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
        
        #SAVE MODEL AFTER EPOCH
        filename = f'{savedir}/KeyPoints_{epoch:03}.pth'
        filenamebest = f'{savedir}/KeyPoints_best_{epoch:03}.pth'

        if args.epochs_save > 0 and epoch > 0 and epoch % args.epochs_save == 0:
            torch.save(model, filename)
            print(f'save: {filename} (epoch: {epoch})')

        if is_best:
            torch.save(model, filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')       

        if epoch > 0 and epoch % 10 == 0:
            filenameCheckpoint = savedir + '/KeyPoints_normal.pth.tar'
            filenameBest = savedir + '/KeyPoints_best.pth.tar'

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': str(model),
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filenameCheckpoint, filenameBest)
            
        # with open(automated_log_path, "a") as myfile:
        #     iouTrain = 0
        #     if len(datasets) > 1:
        #         myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss['train'], average_epoch_loss['val'], usedLr))
        #     else:
        #         myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss['train'], average_epoch_loss['val'], usedLr))
    
    filename = f'{savedir}/KeyPoints_final.pth'
    torch.save(model, filename)
    return(model)   

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)
    else:
        torch.save(state, filenameCheckpoint)

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
        self.batch_size = 1
        self.lr = 1e-2
        
        self.model = 'drnet'
        self.debug = True
        self.bnsync = False
        self.random_rotate = 0
        self.random_scale = 0
        self.num_epochs = 300
        self.savedir  = ''
        
        self.em_dim = 100
        self.K = 1e-4
        self.theta = 0
        self.num_samples = 0
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
    get_dataset = RadarDataset(args)
    # # savedir = f'../save_{args.model}/{args.savedir}'
    # savedir = f'../save_drnet/{args.savedir}'

    # if os.path.exists(savedir + '/model_best.pth') and not args.resume and not args.finetune:
    #     print("Save directory already exists ... ")
    #     sys.exit(0)

    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)

    # if not args.resume:
    #     with open(savedir + '/opts.txt', "w") as myfile:
    #         myfile.write(str(args))

    #Load Model
    assert os.path.exists(args.model + ".py"), f"Error: model definition for {args.model} not found"

    #load pretrained model
    if args.loadmodel:
        model = torch.load('..\\save_drnet\\KeyPoints.pth').module
    else:
        model_file = importlib.import_module(args.model)
        # if args.bnsync:
        #     model_file.BatchNorm = batchnormsync.BatchNormSync
        # else:
        #     model_file.BatchNorm = torch.nn.BatchNorm2d
        model_file.BatchNorm = torch.nn.BatchNorm2d
        NUM_LABELS = get_dataset.num_labels
        model = model_file.Net(NUM_LABELS , args.em_dim , args.resnet)		
        # copyfile(args.model + ".py", savedir + '/' + args.model + ".py")

    print("========== TRAINING STARTS =============")
    train_start = time.time()
    model = train(args, get_dataset, model, False)
    print("========== TRAINING FINISHED ===========")
    print(f"Took {(time.time()-train_start)/60} minutes")