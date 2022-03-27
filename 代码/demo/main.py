import torch
from torch.utils.data import DataLoader,Dataset

from MY_traindataset import load_csv, Getdata
from MY_testdataset import load_test_csv, Gettestdata
from resnet18 import Res18
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"    #使用1号

import visdom
viz = visdom.Visdom()
viz.line([[3., 3.]], [0.], win='loss', opts=dict(title='train&valid loss', legend=['train loss', 'test_loss']))

batchsize = 180
epochs = 100
lr, weightdecay = 0.001, 1e-3

"""导入数据"""
trainset_path = "/home/fanqi/weather/data/traindata/"
testset_path = "/home/fanqi/weather/data/testdata/"
csv_save_root = '/home/fanqi/weather/demo'
trainset_csv_name = 'train_iter1.csv'
valset_csv_name = 'val_iter1.csv'
testset_csv_name = 'test_iter1.csv'
# 获取 trainset
[train_datas, train_labels, train_times] = load_csv(trainset_path, trainset_csv_name, csv_save_root, mode='train', rate = 0.7)
train_dataset = Getdata(train_datas, train_labels, train_times)
train_set = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
# 获取 valset
[val_datas, val_labels, val_times] = load_csv(trainset_path, valset_csv_name, csv_save_root, mode='val', rate = 0.7)
val_dataset = Getdata(val_datas, val_labels, val_times)
val_set = DataLoader(val_dataset, batch_size=batchsize, shuffle=True)
# 获取 test_set
[test_datas,test_stations, test_times] = load_test_csv(testset_path, testset_csv_name, csv_save_root)
test_dataset = Gettestdata(test_datas, test_stations, test_times)
test_set = DataLoader(test_dataset, batch_size = batchsize, shuffle = False)

"""搭建模型"""
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Res18().to(device=device)
# model.device = device

trainer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weightdecay)
loss = torch.nn.MSELoss()

val_loss_min = 1e20
for epoch in range(epochs):
    train_epoch_loss = []

    # 训练程序
    for step, (train_data, train_label, train_time) in enumerate(train_set):
        train_data = train_data.to(device)
        train_label = train_label.to(device)
        train_time = train_time.to(device)

        out = model(train_data)
        loss_train = loss(out, train_label)
        train_epoch_loss.append(loss_train)

        trainer.zero_grad()

        loss_train.backward()
        trainer.step()
        # print(sum(train_epoch_loss)/len(train_epoch_loss))
        if (step + 1) % 20 == 0:
            print("epoch:{}, step:{} train loss : {:.4f}".format(epoch+1, step+1, sum(train_epoch_loss)/len(train_epoch_loss)))
    train_epoch_loss =  sum(train_epoch_loss)/len(train_epoch_loss)
    print("epoch:{}, train_epoch_loss : {:.4f}".format(epoch+1, train_epoch_loss))

    # 测试程序
    model.eval()
    val_epoch_loss = []
    with torch.no_grad():
        for step, (val_data, val_label, val_time) in enumerate(val_set):
            val_data = val_data.to(device)
            val_label = val_label.to(device)
            val_time = val_time.to(device)

            out = model(val_data)
            loss_val = loss(out, val_label)
            val_epoch_loss.append(loss_val)
        val_epoch_loss = sum(val_epoch_loss)/len(val_epoch_loss)
        print("epoch:{}, val_epoch_loss : {:.4f}".format(epoch+1, val_epoch_loss))
    viz.line([[train_epoch_loss.cpu().detach().numpy(), val_epoch_loss.cpu().detach().numpy()]], [epoch], win='loss', update='append')
    if val_loss_min > val_epoch_loss:
        val_loss_min = val_epoch_loss
        print("val_loss_min:[{:.4f}]".format(val_loss_min))
        model_name = 'val_epoch_loss_min.pth'
        torch.save(model.state_dict(), model_name)  # 保存全部模型参数
    else:
        print("valid_acc_final:[{:.4f}]".format(val_epoch_loss))
        model_name = 'final_model.pth'
        torch.save(model.state_dict(), model_name)  # 保存全部模型参数

