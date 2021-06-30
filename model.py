from torch.types import Number
from util import get_onehot
from iou import box3d_iou, get_3d_box
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PointEmb(nn.Module):
    def __init__(self, in_feat, out_feat, bn):
        super().__init__()
        self.fc1 = nn.Linear(in_feat, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, out_feat)
        self.bn1 = nn.BatchNorm1d(bn)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.bn1(out)
        out = torch.cat([x, out], dim=-1)
        return out


class PCF(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.conv1 = torch.nn.Conv1d(ch, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, ch ** 2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.ch).flatten().astype(np.float32)))
            .view(1, self.ch * self.ch)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.ch, self.ch)
        return x


class SelectionDownSampling(nn.Module):
    def __init__(self, in_c, out):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, in_c, 3)
        self.conv2 = nn.Conv1d(in_c, in_c, 2)
        self.avg1 = nn.AvgPool1d(2)
        self.conv3 = nn.Conv1d(in_c, in_c, 3)
        self.conv4 = nn.Conv1d(in_c, in_c, 2)
        self.avg2 = nn.AvgPool1d(2)
        self.conv2 = nn.Conv1d(in_c, in_c, 3)
        self.conv21 = nn.Conv1d(in_c, in_c, 2)
        self.avg3 = nn.AvgPool1d(2)
        self.upsample = nn.Upsample(out)

    def forward(self, x):
        out = self.avg1(self.conv2(self.conv1(x.transpose(-1, -2))))
        out = self.avg2(self.conv4(self.conv3(out)))
        out = self.avg3(self.conv21(self.conv2(out)))
        out = self.upsample(out)
        return out.transpose(-1, -2)


class DownSampling(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, in_c, 3, padding=1)
        self.conv2 = nn.Conv1d(in_c, in_c, 3, padding=1)
        self.conv1x1 = nn.Conv1d(in_c, in_c, 1)
        self.avg2 = nn.AvgPool1d(2)

    def forward(self, x, x2):
        out1 = self.conv1(x.transpose(-1, -2)).transpose(-1, -2)
        out2 = self.conv2(out1.transpose(-1, -2)).transpose(-1, -2)
        if x2 is not None:
            outc = self.conv1x1(out1.transpose(-1, -2)).transpose(-1, -2)
            out2 += outc
        out2 = F.relu(self.avg2(out2.transpose(-1, -2)).transpose(-1, -2))
        return out1, out2


class UpSampling(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.unpool1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv1d(in_c, in_c, 3, padding=1)
        self.unpool2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv1d(in_c, in_c, 3, padding=1)

    def forward(self, x, x2):
        out1 = self.conv1(self.unpool1(x.transpose(-1, -2))).transpose(-1, -2)
        if x2 is not None:
            out1 += x2
        out2 = self.conv2(self.unpool2(out1.transpose(-1, -2))).transpose(-1, -2)
        return out1, out2


class DownUp(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.down11 = DownSampling(num_feat)
        self.up21 = UpSampling(num_feat)
        self.down12 = DownSampling(num_feat)
        self.up22 = UpSampling(num_feat)
        self.down13 = DownSampling(num_feat)
        self.up23 = UpSampling(num_feat)
        self.down14 = DownSampling(num_feat)
        self.up24 = UpSampling(num_feat)
        self.down15 = DownSampling(num_feat)
        self.up31 = UpSampling(num_feat)
        self.up32 = UpSampling(num_feat)
        self.up33 = UpSampling(num_feat)
        self.up41 = UpSampling(num_feat)
        self.up42 = UpSampling(num_feat)
        self.up51 = UpSampling(num_feat)

    def forward(self, x):
        R11, D11 = self.down11(x, None)
        R12, D12 = self.down12(D11, None)
        R13, D13 = self.down13(D12, None)
        R14, D14 = self.down14(D13, None)
        R15, D15 = self.down14(D14, None)
        R21, U21 = self.up21(R12, R11)
        R22, U22 = self.up22(R13, R12)
        R23, U23 = self.up23(R14, R13)
        R24, U24 = self.up24(R15, R14)
        R31, U31 = self.up31(R22, R21)
        R32, U32 = self.up32(R23, R22)
        R33, U33 = self.up33(R24, R23)
        R41, U41 = self.up41(R32, R31)
        R42, U42 = self.up42(R33, R32)
        return R42


class UpDown(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.down11 = UpSampling(num_feat)
        self.up21 = DownSampling(num_feat)
        self.down12 = UpSampling(num_feat)
        self.up22 = DownSampling(num_feat)
        self.down13 = UpSampling(num_feat)
        self.up23 = DownSampling(num_feat)
        self.down14 = UpSampling(num_feat)
        self.up24 = DownSampling(num_feat)
        self.down15 = UpSampling(num_feat)
        self.up31 = DownSampling(num_feat)
        self.up32 = DownSampling(num_feat)
        self.up33 = DownSampling(num_feat)
        self.up41 = DownSampling(num_feat)
        self.up42 = DownSampling(num_feat)
        self.up51 = DownSampling(num_feat)

    def forward(self, x):
        R11, D11 = self.down11(x, None)
        R12, D12 = self.down12(R11, None)
        R13, D13 = self.down13(R12, None)
        R14, D14 = self.down14(R13, None)
        R15, D15 = self.down14(R14, None)
        R21, U21 = self.up21(D11, D12)
        R22, U22 = self.up22(D12, D13)
        R23, U23 = self.up23(D13, D14)
        R24, U24 = self.up24(D14, D15)
        R31, U31 = self.up31(U21, U22)
        R32, U32 = self.up32(U22, U23)
        R33, U33 = self.up33(U23, U24)
        R41, U41 = self.up41(U31, U32)
        R42, U42 = self.up42(U32, U33)
        return R42


class Fusion(nn.Module):
    def __init__(self, feat, out_feat, instances_n):
        super().__init__()
        self.conv1 = nn.Conv1d(feat, feat, 1)
        self.fc1 = nn.Linear(feat, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, out_feat)
        self.maxpool = nn.AdaptiveAvgPool1d(instances_n)
        self.bn1 = nn.BatchNorm1d(out_feat)

    def forward(self, x):
        out = F.relu(self.conv1(x.transpose(-1, -2)))
        out = self.fc1(out.transpose(-1, -2))
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = F.relu(out).transpose(-1, -2)
        out = self.maxpool(out)
        out = self.bn1(out).transpose(-1, -2)
        return out


class ClassHead(nn.Module):
    def __init__(self, in_feat, n_class, feat):
        super().__init__()
        self.fc1 = nn.Linear(in_feat, 64)
        self.ln1 = nn.BatchNorm1d(feat)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, n_class)
        self.fc6 = nn.Linear(n_class, n_class)
        self.ln2 = nn.BatchNorm1d(feat)

    def forward(self, x):
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.fc3(self.fc2(out))
        out = self.fc6(self.ln2(self.fc5(self.fc4(out))))
        return out


class ConfHead(nn.Module):
    def __init__(self, in_feat, feat):
        super().__init__()
        self.fc1 = nn.Linear(in_feat, feat)
        self.ln1 = nn.BatchNorm1d(feat)
        self.fc2 = nn.Linear(feat, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.fc6 = nn.Linear(1, 1)
        self.ln2 = nn.BatchNorm1d(feat)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.ln1(out)
        out = F.relu(self.fc3(F.relu(self.fc2(out))))
        out = self.ln2(F.relu(self.fc6(F.relu(self.fc5(F.relu(self.fc4(out)))))))
        return out


class BoxHead(nn.Module):
    def __init__(self, in_feat, in_ins):
        super().__init__()
        self.fc1 = nn.Linear(in_feat, 32)
        self.ln1 = nn.BatchNorm1d(in_ins)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 7)
        self.fc6 = nn.Linear(7, 7)
        self.ln2 = nn.BatchNorm1d(in_ins)

    def forward(self, x):
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.fc3(self.fc2(out))
        out = self.fc6(self.ln2(self.fc5(self.fc4(out))))
        return out


class Model(nn.Module):
    def __init__(
        self,
        in_c=4,
        in_points=12000,
        hidden_feat=32,
        feat=16,
        num_instance=1024,
        num_class=8,
    ):
        super().__init__()
        self.pfc = PCF(in_c)
        self.pointemb = PointEmb(in_c, hidden_feat - in_c, in_points)
        self.select = SelectionDownSampling(hidden_feat, 1024)
        self.ud1 = UpDown(hidden_feat)
        self.du1 = DownUp(hidden_feat)
        self.fus = Fusion(hidden_feat, feat, num_instance)
        self.ud2 = UpDown(hidden_feat)
        self.du2 = DownUp(hidden_feat)
        self.fus2 = Fusion(hidden_feat, feat, num_instance)
        self.modal = Fusion(feat, feat, num_instance)
        self.cls = ClassHead(feat, num_class, num_instance)
        self.box = BoxHead(feat, num_instance)
        # self.conf = ConfHead(feat, num_instance)

    def forward(self, x: torch.Tensor):
        trans = self.pfc(x.transpose(-1, -2))
        x = torch.bmm(x, trans)
        out = self.pointemb(x)
        out = self.select(out)
        out1 = self.ud1(out)
        out2 = self.du1(out)
        _out1 = self.ud2(out)
        _out2 = self.du2(out)
        fus1 = self.fus(torch.cat([out1, out2], dim=-2))
        fus2 = self.fus2(torch.cat([_out1, _out2], dim=-2))
        out = self.modal(torch.cat([fus1, fus2], dim=-2))
        cls_pred = self.cls(out)
        box_pred = self.box(out)
        # conf_pred = self.conf(out)
        return cls_pred, box_pred

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.25,reduction=torch.mean):
        super(FocalLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, true):
        true = true
        pred = pred
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor
        # norm = torch.maximum(torch.tensor(1),torch.tensor(mask.size(0)))
        return self.reduction(loss)

class LossFN(nn.Module):
    def __init__(
        self, classes, reduction="mean", sigma=1.0, w_offset=1.5, w_cls=2, w_heading=1, w_dimension=1
    ):
        super().__init__()
        self.reduction = reduction
        self.sigma = sigma
        self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.reg_loss_fn = nn.SmoothL1Loss(reduction='none',beta=self.sigma)
        self.w_heading = w_heading
        self.w_offset = w_offset
        self.w_cls = w_cls
        self.w_dimension = w_dimension

    def forward(self, cls, offset, dim, heading, target):
        cls_loss = self.cls_loss_fn(cls,target[0])
        offset_loss = self.reg_loss_fn(offset,target[1])
        dim_loss = self.reg_loss_fn(dim,target[2])
        heading_loss = self.reg_loss_fn(heading,target[3])
        return 


        


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], "variable")
                print(n[0])
                print("Tensor with grad found:", tensor)
                print(" - gradient:", tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


if __name__ == "__main__":
    a = torch.rand(2, 1024, 10)
    a[..., 0] = torch.where(a[..., 0] > 0.5, 1, 0)
    i = torch.rand(2, 12000, 3)
    model = Model(3, 12000, 32, 16, 1024, 2)
    optim = torch.optim.AdamW(model.parameters(), 1e-2, weight_decay=0.02)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 1)
    e = 0
    print(a)
    lossFN = LossFN(2)
    while True:
        for _ in range(int(input())):
            e += 1
            print("Epoch", e)
            conf, cls, reg = model(i)
            # print(o.shape)
            # print('Class :',torch.sigmoid(o[...,1:3]).softmax(-1))
            # print('Mask :',torch.where(o[...,0]>0.5,1,0))
            loss, (conf_loss, cls_loss, reg_loss) = lossFN(conf, cls, reg, a)
            loss.backward()
            print("Conf loss :", conf_loss.item())
            print("Class Loss :", cls_loss.item())
            print("Reg loss :", reg_loss.item())
            optim.step()
            sch.step()
            optim.zero_grad()
