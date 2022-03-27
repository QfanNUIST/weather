from collections import defaultdict
from statistics import mean
from numpy.core.fromnumeric import var
import torch
import numpy as np
from xarray import open_dataset
import pathlib, csv, glob
from torch.utils.data import Dataset, DataLoader

"""
load_data为获取vars_value 的函数
同时创建一个获取时间信息的函数，vars_name[0]
"""
def load_data(filename):
    result = defaultdict(list)
    ds = open_dataset(filename, engine='scipy')
    vars_name = [name for name in ds]
    # print(vars_name)
    #获 得 变 量 值 列 表, 每 个 变 量 值 是numpy多 维 数 组.
    #真 实 值 格 点 资 料 中 的-99999.0代 表 缺 测 值.
    for name in vars_name:
        result[name] = ds.data_vars[name].values
    return result

if __name__ == '__main__':
    examples = 
    temp = load_data('./data/traindata/example00001/grid_inputs_01.nc')
    print('temp finish')