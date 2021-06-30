import numpy as np
import torch
from iou import box3d_iou, get_3d_box


def size_IoU(box,base):
    h = torch.minimum(box[0],base[0])
    w = torch.minimum(box[1],base[1])
    l = torch.minimum(box[2],base[2])
    intersect = h*w*l
    return intersect / (torch.prod(box) + torch.prod(base) - intersect)

def get_onehot(num, total):
    hot = [0] * total
    hot[num] = 1
    return hot


def get_label(fname):
    constant = ["Car", "Pedestrian","Cyclist"]
    label_data = []
    with open(fname, "r") as f:
        for line in f.readlines():
            if len(line) > 3:
                value = line.split()
            if value[0] in constant:
                value[0] = constant.index(value[0])
            else:
                continue
            data = []
            data.extend(get_onehot(int(value[0]),len(constant)))
            d = [float(v) for v in value[8:]]
            if d != [-1.0, -1.0, -1.0, -1000.0, -1000.0, -1000.0, -10.0]:
                data.extend(d)
            else :
                continue
            label_data.append(data)
    f.close()
    return np.asarray(label_data)

if __name__=='__main__':
    print(get_label('/home/patrick/Workspaces/LiDAR-Obj/dataset/training/label_2/007196.txt'))