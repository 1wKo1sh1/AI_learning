import os
import random
import numpy as np
import torch

# 设置种子使结果可复现
def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 固定哈希种子
    np.random.seed(seed)    # np种子
    random.seed(seed)   # py种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False   # 关闭cuddn加速
        torch.backends.cudnn.deterministic = True   # 设置cudnn为确定性算法

# 检查gpu，cuda是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available, Using GPU")
else:
    device = torch.device("cpu")
    print("CUDA is NOT available, Using CPU")
