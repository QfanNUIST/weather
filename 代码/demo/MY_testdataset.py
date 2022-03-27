"""
filename_root = "/home/fanqi/weather/data/testdata/"
csv_name = 'test_iter'
csv_save_root = '/home/fanqi/weather/demo'

[datas,stations, times] = load_csv(filename_root, csv_name, csv_save_root)
db = Gettestdata(datas, stations, times=times)
"""


"""
在my_load_csv的基础上实现以下功能
1、 取出测试的开始时间

在my_load_csv的基础上实现以下功能
1、 读取test集参数

"""

from numpy.core.fromnumeric import var
import numpy as np
import torch
from xarray import open_dataset
import os, csv, glob
from torch.utils.data import Dataset, DataLoader


data_filename = {} # 所在文件夹的名字，example00001
data_dataname = {} # 文件夹里面的名字，'12_12-36h': 0, 'grid_inputs_01.nc': 1,


"""
load_data为获取vars_value 的函数
同时创建一个获取时间信息的函数，vars_name[0]
"""
def load_data(filename):
    ds = open_dataset(filename, engine='scipy')
    vars_name = [name for name in ds]
    # print(vars_name)
    #获 得 变 量 值 列 表, 每 个 变 量 值 是numpy多 维 数 组.
    #真 实 值 格 点 资 料 中 的-99999.0代 表 缺 测 值.
    vars_values = [ds.data_vars[name].values for name in vars_name]
    return vars_values

"""获得所在文件夹的名字，example00001，方便之后用 listdir 打开文件夹"""
def get_filename(filename_root):
    for name in sorted(os.listdir(os.path.join(filename_root))):
        # print(name)
        if not os.path.isdir(os.path.join(filename_root, name)):
            # 判断是否为目录，是就导入文件名
            continue
        data_filename[name] = len(data_filename.keys())
    return data_filename

# def get_datanaem(filename_root):
#     data_filename = get_filename(filename_root)
#     for name_exam in data_filename.keys():
#         for name in sorted(os.listdir(os.path.join(filename_root, name_exam))):
#             if not os.path.isdir(os.path.join(filename_root, name_exam, name)):
#                 # 判断是否为目录，是就导入文件名
#                 data_dataname[name] = len(data_dataname.keys())
        
#     return data_dataname

# data_dataname = get_datanaem(filename_root)
# print(data_dataname)


def load_test_csv(filename_root, csv_name, csv_save_root):
    # loda_csv文件实现的功能
    # 将训练的数据grid_inputs_01.nc，的路径 '/home/fanqi/weather/data/traindata/example01840/grid_inputs_02.nc 存在csv文件中
    # 将训练的label obs__01.nc，的路径 home/fanqi/weather/data/traindata/example00003/obs_grid_rain07.nc 存在csv文件中
    # 首先建立data_path和label_label的csv文件，然后通过csv.reader（）将images_path和label读取出来
    if not os.path.exists(os.path.join(csv_save_root, csv_name)):
        data_path = []
        station_path = []
        time_path = []
        filename = get_filename(filename_root)
        # 所在文件夹的名字，example00001

        train_data_name = ['grid_inputs_01.nc','grid_inputs_02.nc','grid_inputs_03.nc','grid_inputs_04.nc', 
        'grid_inputs_05.nc', 'grid_inputs_06.nc', 'grid_inputs_07.nc', 'grid_inputs_08.nc', 'grid_inputs_09.nc']
        train_station_name = ['ji_loc_inputs_01.txt', 'ji_loc_inputs_02.txt', 'ji_loc_inputs_03.txt', 'ji_loc_inputs_04.txt', 
        'ji_loc_inputs_05.txt', 'ji_loc_inputs_06.txt', 'ji_loc_inputs_07.txt', 'ji_loc_inputs_08.txt',
        'ji_loc_inputs_09.txt']
        train_time_name = ['00_12-36h']
        for name in filename.keys():
            for train_dataname in train_data_name:
                data_path += glob.glob(os.path.join(filename_root, name, train_dataname))
                # data_path = '/home/fanqi/weather/data/traindata/example01840/grid_inputs_02.nc
            for train_stationname in train_station_name:
                station_path += glob.glob(os.path.join(filename_root, name, train_stationname))
                # label_path = home/fanqi/weather/data/traindata/example00003/ji_loc_inputs_07.txt
            for train_timename in train_time_name:
                if os.path.exists(os.path.join(filename_root, name, train_timename)):
                    time_num = int(0)
                else:
                    time_num = int(12)
                for i in range(9):
                    time_path.append(time_num)

        # print(len(data_path), data_path)
        with open(os.path.join(csv_save_root, csv_name), mode='w', newline='') as f:
            writer = csv.writer(f)
            for i in range(len(data_path)):
                data = data_path[i]
                station = station_path[i]
                time_read = time_path[i]
                writer.writerow([data, station, time_read])
            print("successfully write to csv file", csv_name)
    else:
        print("{} has been written".format(csv_name))
    
    # 从csv文件中载入 data_path, station_path
    datas, stations, times = [], [], [] 
    with open(os.path.join(csv_save_root, csv_name)) as f:
        reader = csv.reader(f)
        for row in reader:
            data, station, time = row
            datas.append(data)
            stations.append(station)
            times.append(int(time))
    assert len(datas) == len(stations) == len(times)
    return datas, stations, times



class Gettestdata(Dataset):
    def __init__(self, datas: list, stations: list, times: list):
        super(Gettestdata, self).__init__()
        self.data_path = datas
        self.station_path = stations
        self.times_path = times
    
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        time = torch.tensor(self.times_path[index])

        data = load_data(self.data_path[index])
        # 将data变成[1,c,h,w]的tensor
        for i in range(len(data)):
            data[i] = torch.tensor(data[i])
            if data[i].ndim ==3:
                data[i] = data[i].permute(2, 0, 1)
                # print(data[i].shape)
            else:
                data[i] = data[i].unsqueeze(0)
                # print(data[i].shape)
        # 最终获得的data为 全是tensor的list，接下来将其转化为整个tensor
        data = torch.cat(data, dim=0)
        print(data.shape, data.type)
        station_path = self.station_path[index]
        station = np.loadtxt(station_path)
        # station = station.squeeze(0)
        # print(station.shape)
        time = torch.tensor(self.times_path[index])
        return data, station, time


# # 测试代码
# def main():
#     filename_root = "/home/fanqi/weather/data/testdata/"
#     csv_name = 'test_iter1'
#     csv_save_root = '/home/fanqi/weather/demo'
#     [datas,stations, times] = load_test_csv(filename_root, csv_name, csv_save_root)
#     db = Gettestdata(datas, stations, times=times)
#     train_data = DataLoader(db, batch_size= 1 )
#     datas, stations, times = next(iter(train_data))
#     print(stations.type)
#     print(datas.shape)

# main()