import time
import numpy as np
import torch
from util import box3d_iou,get_3d_box


def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.001):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes tensor
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """
    N = dets.shape[1]
    keep = []
    indexes = torch.arange(0, N, dtype=torch.float).view(N, 1).repeat(dets.size(0),1,1)
    dets = torch.cat((dets, indexes), dim=-1)
    scores = box_scores
    print(scores)
    for j in range(dets.size(0)):
        for i in range(N):
            tscore = scores[j,i].clone()
            pos = i + 1

            if i != N - 1:
                maxscore, maxpos = torch.max(scores[j,pos:], dim=0)
                if tscore < maxscore:
                    dets[j,i], dets[j,maxpos.item() + i + 1] = dets[j,maxpos.item() + i + 1].clone(), dets[j,i].clone()
                    scores[j,i], scores[j,maxpos.item() + i + 1] = scores[j,maxpos.item() + i + 1].clone(), scores[j,i].clone()

            # IoU calculate
            iou = torch.Tensor()
            for x in range(pos,N):
                iou = torch.cat([iou,torch.tensor([box3d_iou(get_3d_box(dets[j,i,:-1].detach()),get_3d_box(dets[j,x,:-1].detach()))[0]])])


            # Gaussian decay
            weight = torch.exp(-(iou * iou) / sigma)
            scores[j,pos:] = weight * scores[j,pos:]

    # select the boxes and keep the corresponding indexes
        keep.append(dets[j,:,-1][scores[j] > thresh].int().detach().numpy().tolist())

    return keep


def speed():
    boxes = 1000*torch.rand((1, 100, 7), dtype=torch.float)
    boxscores = torch.rand((1, 100), dtype=torch.float)

    # cuda flag
    cuda = 1 if torch.cuda.is_available() else 0
    if cuda:
        boxes = boxes.cuda()
        boxscores = boxscores.cuda()

    start = time.time()
    for i in range(1000):
        soft_nms_pytorch(boxes, boxscores)
    end = time.time()
    print("Average run time: %f ms" % (end-start))


def test():
    # boxes and boxscores
    boxes = torch.rand(2,5,7)
    boxscores = torch.rand(2,5, dtype=torch.float)

    # cuda flag
    cuda = 1 if torch.cuda.is_available() else 0
    if cuda:
        boxes = boxes.cuda()
        boxscores = boxscores.cuda()

    print(soft_nms_pytorch(boxes, boxscores,thresh=0.6))


if __name__ == '__main__':
    test()
    # speed()
