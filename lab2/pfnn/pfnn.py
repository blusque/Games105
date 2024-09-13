import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
from pfnn_dataset import PFNNDataset
from pfnn_model import PFNN
from tqdm import tqdm, trange

input_dims = 234
output_dims = 277
batch_size = 32
lr = 5e-5
betas = [0.9, 0.999]
epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    seed = random.randint(1, 10000)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    print("##### Training Seed: {} #####\n".format(seed))

    print("==> Building Model...")
    print("input_dims: {}".format(input_dims))
    print("output_dims: {}".format(output_dims))
    print("device = {}".format(device))
    model = PFNN(input_dims, output_dims, device).to(device)
    print("===== Model Done =====\n")

    print("==> Loading Dataset...")
    dataset = PFNNDataset()
    print("dataset len: {}".format(len(dataset)))
    print("batch size: {}".format(batch_size))
    print("shuffle: {}".format(True))
    dloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print("===== Dataset Done =====\n")

    print("==> Initializing Optimizer...")
    # print("optimizer: {}".format(Adam))
    print("learning rate: {}".format(lr))
    print("betas: {}".format(betas))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    print("===== Optimizer Done =====\n")

    print("==> Training...")
    for e in range(1, epochs + 1):
        train(e, dloader, model, optimizer)
        if e % 10 == 0:
            save_model(model, "pfnn_model_{}.pth".format(e))
    print("===== Training Done =====")
    pass


def train(epoch, dloader, model, optimizer):
    bar = tqdm(enumerate(dloader, 1), leave=True, total=len(dloader),
               desc="Epoch: {}, Iterations: {}".format(epoch, len(dloader)))
    total_loss = 0
    mse_loss = 0
    regu_loss = 0
    if epoch >= 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr / 2
    for data, _ in zip(dloader, bar):
        x = data[0]
        x = x.to(device)
        y = data[1]
        y = y.to(device)
        phase = data[2]
        phase = phase.to(device)

        # TODO: the dimisions of r is incorrect, find out why and fix it
        r, beta = model(x, phase)

        # print("r size: ", r.size())
        mse = torch.mean((r - y) ** 2)
        regulation = 0.01 * torch.abs(beta).mean()
        loss = mse + regulation
        loss.backward()

        optimizer.step()

        total_loss += loss.item() / 120
        mse_loss += mse.item() / 120
        regu_loss += regulation.item() / 120
        bar.set_postfix({"Loss": total_loss, "MSE": mse_loss, "RE": regu_loss})
    pass


def save_model(model: PFNN, file):
    torch.save(model, file)
    pass


def validate():
    pass


if __name__ == '__main__':
    main()
