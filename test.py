from box2cloud import visualize
from util import get_label
from nms import soft_nms_pytorch
import numpy as np
import torch

ID = '007108'


model = torch.load("/home/patrick/Workspaces/LiDAR-Obj/export/car_pred/model-100-e266-4.66.pth")
inp = np.fromfile(
    f"/home/patrick/Workspaces/LiDAR-Obj/dataset/training/velodyne_reduced/{ID}.bin",
    "<f4",
).reshape(-1, 4)
np.random.shuffle(inp)
inp = inp[...,:12000, :3]
points = inp.T
x,y,z = points
index = np.intersect1d(np.where(abs(x)<=60),np.where(np.logical_and(y<=60,y>=-60)))
index = np.intersect1d(np.where(np.logical_and(z<=3,z>=-3)),index)
b = points.T[index]
conf, cls, reg = model(torch.tensor(inp).unsqueeze(0))
print(reg.shape,cls.shape)
# nms = soft_nms_pytorch(
#     reg[0].view(1, -1, 7),
#     reg[0][..., :2].argmax(-1).view(1, -1),
#     thresh=0.001,
# )
label = f"/home/patrick/Workspaces/LiDAR-Obj/dataset/training/label_2/{ID}.txt"
# print("Boxes : ", nms)
inp = np.fromfile(
    f"/home/patrick/Workspaces/LiDAR-Obj/dataset/training/velodyne_reduced/{ID}.bin",
    "<f4",
).reshape(-1, 4)[...,:12000, :3]
points = inp.T
x,y,z = points
index = np.intersect1d(np.where(abs(x)<=60),np.where(np.logical_and(y<=60,y>=-60)))
index = np.intersect1d(np.where(np.logical_and(z<=3,z>=-3)),index)
b = points.T[index]
for i in range(len(cls[0])):
    classes = cls[i].argmax(-1).view(1,-1,1)
    box = torch.cat(
        [classes[i],reg[i]],
        dim=-1,
    ).detach()
    print(box.shape)
    visualize(b, box)
