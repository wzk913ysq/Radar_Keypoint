import torch
import sys
from models.box_pre_net import *

a = Net1(BasicBlock, [2, 2, 2, 2])	
a = torch.nn.DataParallel(a).cuda()
b = torch.load('state.pth')
a.load_state_dict(b)
torch.save(a, 'area_pre3.pth')