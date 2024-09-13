import torch
import torch.nn as nn
from torch.nn import functional as F


def phase_function(phase, alpha):
    assert len(alpha) == 4

    coeff = 4 * phase / (2 * torch.pi)
    mu = coeff % 1
    k1 = torch.floor(coeff)
    k1 = k1 % 4
    k0 = k1 - 1
    k0 = k0 % 4
    k2 = k1 + 1
    k2 = k2 % 4
    k3 = k1 + 2
    k3 = k3 % 4

    def cubic_spline(a, b, c, d, mu):
        return b + \
            mu * (0.5 * c - 0.5 * a) + \
            mu ** 2 * (a - 2.5 * b + 2 * c - 0.5 * d) + \
            mu ** 3 * (1.5 * b - 1.5 * c + 0.5 * d - 0.5 * a)

    params = cubic_spline(
        alpha[0], alpha[1],
        alpha[2], alpha[3], mu)
    return params


class PFNN(nn.Module):
    def __init__(self, input_dims, output_dims, device):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.f = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(input_dims, 512),
                nn.Linear(512, 512),
                nn.Linear(512, output_dims)
            ]),
            nn.ModuleList([
                nn.Linear(input_dims, 512),
                nn.Linear(512, 512),
                nn.Linear(512, output_dims)
            ]),
            nn.ModuleList([
                nn.Linear(input_dims, 512),
                nn.Linear(512, 512),
                nn.Linear(512, output_dims)
            ]),
            nn.ModuleList([
                nn.Linear(input_dims, 512),
                nn.Linear(512, 512),
                nn.Linear(512, output_dims)
            ])
        ])

        self.dropout1 = nn.Dropout(0.7)
        self.dropout2 = nn.Dropout(0.7)
        self.dropout3 = nn.Dropout(0.7)

    def forward(self, x, p):
        assert (x.dtype == torch.float32)
        # print(x.shape)
        bsize = x.shape[0]
        r = range(bsize)
        coeff = 4 * p / (2 * torch.pi)
        mu = coeff % 1.0
        k1 = torch.floor(coeff)
        k1 = (k1 % 4).long().flatten()
        k0 = k1 - 1
        k0 = (k0 % 4).long().flatten()
        k2 = k1 + 1
        k2 = (k2 % 4).long().flatten()
        k3 = k1 + 2
        k3 = (k3 % 4).long().flatten()

        def cubic_spline(a, b, c, d, mu):
            result = b + \
                mu * (0.5 * c - 0.5 * a) + \
                mu ** 2 * (a - 2.5 * b + 2 * c - 0.5 * d) + \
                mu ** 3 * (1.5 * b - 1.5 * c + 0.5 * d - 0.5 * a)
            # result.astype = torch.float32
            # print("mu type: ", mu.dtype)
            # print("result dtype: ", result.dtype)
            return result

        w = []

        H1 = []
        w0 = []
        for module in self.f:
            H1.append(module[0](x).unsqueeze(dim=0))
            for params in module[0].parameters():
                # print("params size: ", params.flatten().size())
                w0.append(params.flatten())
        H1 = torch.concat(H1, dim=0)
        w0 = torch.concat(w0, dim=0)
        H10 = H1[k0, r]
        H11 = H1[k1, r]
        H12 = H1[k2, r]
        H13 = H1[k3, r]
        # print("dtype: ", H1.dtype)
        # r1 = cubic_spline(H10, H11, H12, H13, mu)
        # print("dtype: ", r1.dtype)
        mid0 = cubic_spline(H10, H11, H12, H13, mu)
        mid0 = self.dropout1(mid0)
        mid0 = torch.relu(mid0)
        # print("mid0 dtype: ", mid0.dtype)

        H2 = []
        w1 = []
        # print("mid0 dtype: ", mid0.dtype)
        for module in self.f:
            H2.append(module[1](mid0).unsqueeze(dim=0))
            for params in module[1].parameters():
                w1.append(params.flatten())
        H2 = torch.concat(H2, dim=0)
        w1 = torch.concat(w1, dim=0)
        H20 = H2[k0, r]
        H21 = H2[k1, r]
        H22 = H2[k2, r]
        H23 = H2[k3, r]
        mid1 = torch.relu(cubic_spline(H20, H21, H22, H23, mu))
        # mid1 = self.dropout2(mid1)

        Y = []
        w2 = []
        for module in self.f:
            Y.append(module[2](mid1).unsqueeze(dim=0))
            for params in module[2].parameters():
                w2.append(params.flatten())
        Y = torch.concat(Y, dim=0)
        w2 = torch.concat(w2, dim=0)
        Y0 = Y[k0, r]
        Y1 = Y[k1, r]
        Y2 = Y[k2, r]
        Y3 = Y[k3, r]

        w = [w0, w1, w2]
        w = torch.concat(w, dim=0)
        y = cubic_spline(Y0, Y1, Y2, Y3, mu)
        # y = self.dropout3(y)

        return y, w
