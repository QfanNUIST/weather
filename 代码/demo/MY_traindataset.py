

"""
[datas,labels, times] = load_csv(filename_root, csv_name, csv_save_root, mode='val', rate = 0.7 )
db = Getdata(datas, labels, times=times)
"""

from numpy.core.fromnumeric import var
import torch
from xarray import open_dataset
import os, csv, glob
from torch.utils.data import Dataset, DataLoader

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


data_filename = {} # 所在文件夹的名字，example00001
data_dataname = {} # 文件夹里面的名字，'12_12-36h': 0, 'grid_inputs_01.nc': 1,

"""获得所在文件夹的名字，example00001，方便之后用 listdir 打开文件夹"""
def get_filename(filename_root):
    for name in sorted(os.listdir(os.path.join(filename_root))):
        # print(name)
        if not os.path.isdir(os.path.join(filename_root, name)):
            # 判断是否为目录，是就导入文件名
            continue
        data_filename[name] = len(data_filename.keys())
    return data_filename


def load_csv(filename_root, csv_name, csv_save_root, mode = 'train', rate = 0.7,):
    # loda_csv文件实现的功能
    # 将训练的数据grid_inputs_01.nc，的路径 '/home/fanqi/weather/data/traindata/example01840/grid_inputs_02.nc 存在csv文件中
    # 将训练的label obs__01.nc，的路径 home/fanqi/weather/data/traindata/example00003/obs_grid_rain07.nc 存在csv文件中
    # 首先建立data_path和label_label的csv文件，然后通过csv.reader（）将images_path和label读取出来
    if not os.path.exists(os.path.join(csv_save_root, csv_name)):
        data_path = []
        label_path = []
        time_path = []
        filename = get_filename(filename_root)
        # 所在文件夹的名字，example00001

        train_data_name = ['grid_inputs_01.nc','grid_inputs_02.nc','grid_inputs_03.nc','grid_inputs_04.nc', 
        'grid_inputs_05.nc', 'grid_inputs_06.nc', 'grid_inputs_07.nc', 'grid_inputs_08.nc', 'grid_inputs_09.nc']
        train_label_name = ['obs_grid_rain01.nc', 'obs_grid_rain02.nc', 'obs_grid_rain03.nc',
        'obs_grid_rain04.nc', 'obs_grid_rain05.nc', 'obs_grid_rain06.nc', 'obs_grid_rain07.nc', 
        'obs_grid_rain08.nc', 'obs_grid_rain09.nc']
        train_time_name = ['00_12-36h']
        for name in filename.keys():
            if name == 'example01632' or name == 'example01858' :
                continue
            else:
                for train_dataname in train_data_name:
                    data_path += glob.glob(os.path.join(filename_root, name, train_dataname))
                    # data_path = '/home/fanqi/weather/data/traindata/example01840/grid_inputs_02.nc
                for train_labelname in train_label_name:
                    label_path += glob.glob(os.path.join(filename_root, name, train_labelname))
                    # label_path = home/fanqi/weather/data/traindata/example00003/obs_grid_rain07.nc
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
            if mode == 'train':
                for i in range(int(rate*(len(data_path)/9))*9):
                    data = data_path[i]
                    label = label_path[i]
                    time_read = time_path[i]
                    writer.writerow([data, label, time_read])
                print("successfully write to csv file", csv_name)
            else:
                for i in range(int(rate*(len(data_path)/9))*9, len(data_path)):
                    data = data_path[i]
                    label = label_path[i]
                    time_read = time_path[i]
                    writer.writerow([data, label, time_read])
                print("successfully write to csv file", csv_name)
    else:
        print("{} has been written".format(csv_name))
    
    # 从csv文件中载入 data_path, label_path
    datas, labels, times = [], [], [] 
    with open(os.path.join(csv_save_root, csv_name)) as f:
        reader = csv.reader(f)
        for row in reader:
            data, label, time = row
            datas.append(data)
            labels.append(label)
            times.append(int(time))
    assert len(datas) == len(labels) == len(times)
    return datas, labels, times



class Getdata(Dataset):
    def __init__(self, datas: list, labels: list, times: list):
        super(Getdata, self).__init__()
        self.data_path = datas
        self.label_path = labels
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
        # print(data.shape, data.type)
        label = torch.tensor(load_data(self.label_path[index]))
        # label = label.squeeze(0)
        # print(label.shape)
        time = torch.tensor(self.times_path[index])
        return data, label, time


# # 测试代码
# def main():
#     filename_root = "/home/fanqi/weather/data/traindata/"
#     csv_name = 'train_iter'
#     # csv_name = 'val_iter'
#     csv_save_root = '/home/fanqi/weather/demo'
#     [datas,labels, times] = load_csv(filename_root, csv_name, csv_save_root, mode='train' )
#     db = Getdata(datas, labels, times=times)
#     train_data = DataLoader(db, batch_size= 1 , shuffle=True)
#     datas, labels, times = next(iter(train_data))
#     print(times)
#     print(datas.shape)

# main()