import torch
import torch.nn as nn

'''
state_dict() 返回模型的状态字典包含所有参数
save保存到路径下的文件夹中
torch.save(model.state_dict(), 'model.pth')

直接保存模型全部结构和参数
torch.save(model, 'entire_model.pth')
'''
