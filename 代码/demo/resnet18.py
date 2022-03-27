"""
使用resnet实现对cifar数据的分类任务
"""
import torch
from torch import nn


class Resblk(nn.Module):
    """
    def 一层res_blk
    """

    def __init__(self, ch_in, ch_out, stride=1):
        super(Resblk, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),

        )

        self.extra = nn.Sequential(
            nn.MaxPool2d(kernel_size=stride, stride=stride, padding=0)
        )

        if ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride)
            )
        # 用来确保输入图片和输出之后的图片，channel，H，W相同

    def forward(self, x):
        out = self.conv(x)
        # print(out.shape)
        # [b , c , H,W] ==> out:[b,c,H/2,H/2]
        x = self.extra(x)
        # [b , c , H,W] ==> x:[b,c,H/2,H/2]
        # print(x.shape)

        out = x + out
        # print(out.shape)
        # 短接做残差
        return out


class Res18(nn.Module):
    """
    通过调用Resblk完成对Res18的编程
    """
    def __init__(self):
        super(Res18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(58, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
            # in:[b,3,H,W] ==> out[b,32,,H,W]
        )

        self.resblk = nn.Sequential(
            Resblk(64, 128, stride=1),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            # [b,64,,H,W] ===> [b,64,,H/4,W/4]
            Resblk(128, 64, stride=1),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            # [b,128,,H,W] ===> [b,128,,H/16,W/16]
            Resblk(64, 32, stride=1),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            # [b,256,,H,W] ===> [b,256,,H/64,W/64]
            Resblk(32, 1, stride=1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # [b,512,,H,W] ===> [b,512,,H/128,128]
        )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.conv1(x)
        # print(x.shape)
        out = self.resblk(x)
        # print(out.shape)
        return out


"""
测试代码查看输入图像的 HxW 在Res18 处理后的维度，需要将Res18的forward（）函数的print（）取消注释
"""


# def main():
#     x = torch.rand(2, 3, 224, 224)
#     model = Res18()
#     out = model(x)
#     print(out.shape)
#
#
# main()