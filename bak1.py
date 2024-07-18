import torch
from dataset import MNIST
from vit import ViT
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 设备

    dataset = MNIST()  # 数据集

    model = ViT().to(DEVICE)  # 模型

    try:  # 加载模型
        model.load_state_dict(torch.load("model.pth"))
    except FileNotFoundError:
        pass

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 优化器

    EPOCH = 50
    BATCH_SIZE = 64

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=10,
        persistent_workers=True,
    )  # 数据加载器

    iter_count = 0
    for epoch in range(EPOCH):
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_count % 1000 == 0:
                print(
                    "epoch:{} iter:{}, loss:{}".format(epoch, iter_count, loss.item())
                )
                torch.save(model.state_dict(), ".model.pth")
                os.replace(".model.pth", "model.pth")
            iter_count += 1


if __name__ == "__main__":
    main()
