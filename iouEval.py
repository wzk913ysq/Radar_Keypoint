import torch
import torchvision
import os
deal_unloader = torchvision.transforms.ToPILImage()

class iouEval:

    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.ignoreIndex = self.nClasses-1 ## Ignore the last labeled class, which by default *should* be background. 
        self.reset()

    def reset (self):
        classes = self.nClasses if self.ignoreIndex==-1 else self.nClasses-1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()        

    def addBatch(self, x, y):   
        
        if (x.is_cuda or y.is_cuda):
            x = x.cuda()
            y = y.cuda()

        #if size is "batch_size x 1 x H x W" scatter to onehot
        if (x.size(1) == 1):
            x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3))
            if x.is_cuda:
                x_onehot = x_onehot.cuda()
            x_onehot.scatter_(1, x, 1).float()
        else:
            x_onehot = x.float()
            

        if (y.size(1) == 1):
            y_onehot = torch.zeros(y.size(0), self.nClasses, y.size(2), y.size(3))
            if y.is_cuda:
                y_onehot = y_onehot.cuda()
            y_onehot.scatter_(1, y, 1).float()
        else:
            y_onehot = y.float()

        if (self.ignoreIndex != -1): 
            # ignores = y_onehot[:,self.ignoreIndex].unsqueeze(1)
            ignores = y_onehot[:,1].unsqueeze(1)
            # x_onehot = x_onehot[:, :self.ignoreIndex]
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
            y_onehot = 1 - y_onehot
        else:
            ignores=0

        tpmult = x_onehot * y_onehot   
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fpmult = x_onehot * (1-y_onehot-ignores)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fnmult = (1-x_onehot) * (y_onehot) 
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()

        self.tpmult = tpmult
        self.fpmult = fpmult
        self.fnmult = fnmult

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()


    def imSave(self, dataset, index, savedir):
        self.tpmult = self.tpmult.repeat(1, 3, 1, 1)
        self.fpmult = self.fpmult.repeat(1, 3, 1, 1)
        self.fnmult = self.fnmult.repeat(1, 3, 1, 1)

        for bat in range(self.tpmult.shape[0]):
            for channel in range(3):
                # self.tpmult[bat][channel] = self.tpmult[bat][0]               # white
                if channel != 0:
                    self.fpmult[bat][channel] = 0                               # red
                if channel != 1:
                    self.tpmult[bat][channel] = 0                               # green
                if channel != 2:
                    self.fnmult[bat][channel] = 0                               # green
        print = self.tpmult + self.fpmult + self.fnmult
        torchvision.utils.save_image(print , os.path.join(savedir + '/test_result', f'IoU_{dataset}_{index}.jpg'))

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return torch.mean(iou), iou     #returns "iou mean", "iou per class"
