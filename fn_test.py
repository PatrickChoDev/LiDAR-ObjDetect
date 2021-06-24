import torch, time
from model import Model,LossFN


def display(inp):
    if isinstance(inp, torch.Tensor):
        return list(inp.shape)
    elif isinstance(inp, tuple):
        shape = []
        for t in inp:
            shape.append(display(t))
        return shape
    return "<Unable to display>"


def checkModel(train=True):
    model = Model(3, 12000, 32)
    inp = torch.rand(2, 12000, 3, requires_grad=True)
    print(
        "Model Parameters :",
        sum(p.numel() for p in model.train().parameters() if p.requires_grad),
    )
    if train:
        with torch.no_grad():
            s = time.time()
            y = model.eval()(inp)
            e = time.time()
        print("Shape :", display(y))
        print("Time taken :", e - s)

def checkLoss():
    pred = torch.rand(4, 1024, 15, requires_grad=True)
    cls_target = torch.rand(4, 1024,8)
    reg_target = torch.rand(4, 1024, 7)
    target = torch.cat([cls_target, reg_target], dim=-1)
    # target = pred
    loss_fn = LossFN()
    loss = loss_fn(pred, target, torch.randint(0,2,(4,1024)))
    print("Loss shape :", loss.shape)
    try:
        print(f"Loss : {loss.item():.3f}")
        loss.backward()
        print("Accepted!!")
    except:

        print("Unable to do backward")
        loss.backward()


checkModel(True)
checkLoss()
