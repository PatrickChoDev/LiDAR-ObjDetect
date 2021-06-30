import torch
import os
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset import LiDARDataset
from model import LossFN, Model
from nms import soft_nms_pytorch


class Wrapper:
    def __init__(
        self,
        model,
        instance_num=512,
        num_classes=2,
        dataset="KITTI",
        batch=16,
        crop=True,
        max_point=4000,
        data_dir="./dataset",
        split_ratio=0.9,
        output="./export",
        collate_fn=True,
    ):
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
        self.model = model.to(self.device)
        self.output = output
        self.instance_num = instance_num
        self.num_classes = num_classes
        self.batch = batch
        self.dataset = LiDARDataset(data_dir, dataset, crop, max_point)
        self.train_set, self.valid_set = random_split(
            self.dataset,
            [
                int(split_ratio * len(self.dataset)),
                len(self.dataset) - int(split_ratio * len(self.dataset)),
            ],
        )
        self.train_loader = DataLoader(
            self.train_set,
            self.batch,
            True,
            collate_fn=self.padder if collate_fn else None,
        )
        self.valid_loader = DataLoader(
            self.valid_set,
            self.batch,
            False,
            collate_fn=self.padder if collate_fn else None,
        )

    def padder(self, inp):
        inputs, labels = zip(*inp)
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        try:
            targets = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=0
            )
        except:
            targets = torch.empty(inputs.size(0), 0, self.num_classes + 8)
        finally:
            targets = torch.cat(
                [
                    targets,
                    torch.zeros(
                        inputs.size(0) - targets.size(0),
                        targets.size(1),
                        self.num_classes + 8,
                    ),
                ],
                0,
            )
            targets = torch.cat(
                [
                    targets,
                    torch.zeros(
                        inputs.size(0),
                        self.instance_num - targets.size(1),
                        self.num_classes + 8,
                    ),
                ],
                1,
            )

        return inputs, targets

    def train(
        self,
        epochs,
        valid_freq,
        num_classes,
        lr=1e-3,
        start=1,
        delta=2,
        with_loss=True,
        best_loss=None,
    ):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr, weight_decay=0.01)
        self.lossFN = LossFN(num_classes, sigma=delta)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.train_set.__len__() * start
        )
        self.best_valid_loss = float("inf") if not best_loss else best_loss
        for epoch in range(start, epochs + 1):
            print(f"\tEPOECH {epoch}/{epochs}")
            running_loss = 0.0
            self.model = self.model.train()
            with tqdm(self.train_loader, position=0, leave=False) as pbar:
                for i, g in pbar:
                    self.optimizer.zero_grad()
                    conf, cls, reg = self.model(i)
                    loss, (conf_loss, cls_loss, reg_loss) = self.lossFN(
                        conf, cls, reg, g
                    )
                    # print(conf,)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    running_loss += loss.item() * i.size(0)
                    del i, conf, cls, reg, g
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        lr=f"{self.scheduler.get_last_lr()[0]:.4f}",
                    )
                pbar.close()
                print("== Training Loss ==")
                print(f"Loss : {running_loss/self.train_set.__len__()}")
                if epoch % valid_freq == 0:
                    self.valid(
                        num_classes,
                        epoch,
                        with_loss=with_loss,
                    )

    def valid(
        self,
        num_classes,
        epoch,
        with_loss=False,
        delta=2,
    ):
        valid_loss = 0.0
        self.lossFN = LossFN(num_classes, sigma=delta)
        self.model = self.model.eval()
        with torch.no_grad():
            with tqdm(self.valid_loader, position=0, leave=False) as pbar:
                for i, g in pbar:
                    conf, cls, reg = self.model(i)
                    if with_loss:
                        loss, (conf_loss, cls_loss, reg_loss) = self.lossFN(
                            conf, cls, reg, g
                        )
                        valid_loss += loss.item() * i.size(0)
                        pbar.set_postfix(
                            loss=f"{loss.item():.4f}",
                        )
            mask = torch.where(conf[0, ..., 0].sigmoid() > 0.5)
            nms = soft_nms_pytorch(
                reg[0][mask].view(1, -1, 7),
                reg[0][mask][..., :num_classes].max(-1).values.view(1, -1),
                thresh=0.5,
            )
            print("Boxes : ", nms)
            for i, ind in enumerate(nms):
                print(reg[0][mask].view(1, -1, 7)[i][ind])
            pbar.close()
        if with_loss:
            print("== Validation Loss ==")
            print(f"Loss : {valid_loss/self.valid_set.__len__()}")
            if valid_loss / self.valid_set.__len__() < self.best_valid_loss:
                self.best_valid_loss = valid_loss / self.valid_set.__len__()
                print("SAVED")
                if not os.path.isdir(self.output):
                    os.mkdir(self.output)
                torch.save(
                    self.model,
                    os.path.join(self.output
                    ,f"model-{self.instance_num}-e{epoch}-{valid_loss/self.valid_set.__len__():.2f}.pth")
                )


model = Model(3, 12000, num_instance=100, num_class=2)
model = torch.load('/home/patrick/Workspaces/LiDAR-Obj/export/car_pred/model-100-e266-4.66.pth')

wrap = Wrapper(
    model,
    100,
    2,
    "KITTI_CAR_PRED",
    4,
    data_dir="./dataset/training",
    split_ratio=0.8,
    max_point=12000,
    output="./export/car_pred",
)

# wrap.train(1000, 2, 2, 1e-2, start=1)
wrap.valid(2, 2, with_loss=False)
