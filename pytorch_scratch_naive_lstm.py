import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from time import time

volatile = True

def linear(X, W, bias):
    product = torch.mm(X, W)
    bias = bias.repeat(product.size()[0], 1)
    return product + bias

def sigmoid(x):
    return .5 * (F.tanh(.5 * x) + 1)

def gaussian(shape):
    num = np.random.normal(size=shape)
    return num.astype(np.float32)

D = 1024

WX_SHAPE = (7, D)
Wxi = gaussian(shape=WX_SHAPE)
Wxf = gaussian(shape=WX_SHAPE)
Wxo = gaussian(shape=WX_SHAPE)
Wxg = gaussian(shape=WX_SHAPE)

# PyTorch tensor
Wxi = torch.from_numpy(Wxi).cuda()
Wxf = torch.from_numpy(Wxf).cuda()
Wxo = torch.from_numpy(Wxo).cuda()
Wxg = torch.from_numpy(Wxg).cuda()
Wxi = Variable(Wxi, volatile=volatile)
Wxf = Variable(Wxf, volatile=volatile)
Wxo = Variable(Wxo, volatile=volatile)
Wxg = Variable(Wxg, volatile=volatile)

bX_SHAPE = (D,)
bxi = np.zeros(shape=bX_SHAPE, dtype=np.float32)
bxf = np.zeros(shape=bX_SHAPE, dtype=np.float32)
bxo = np.zeros(shape=bX_SHAPE, dtype=np.float32)
bxg = np.zeros(shape=bX_SHAPE, dtype=np.float32)

# PyTorch tensor
bxi = torch.from_numpy(bxi).cuda()
bxf = torch.from_numpy(bxf).cuda()
bxo = torch.from_numpy(bxo).cuda()
bxg = torch.from_numpy(bxg).cuda()
bxi = Variable(bxi, volatile=volatile)
bxf = Variable(bxf, volatile=volatile)
bxo = Variable(bxo, volatile=volatile)
bxg = Variable(bxg, volatile=volatile)

WH_SHAPE = (D, D)
Whi = gaussian(shape=WH_SHAPE)
Whf = gaussian(shape=WH_SHAPE)
Who = gaussian(shape=WH_SHAPE)
Whg = gaussian(shape=WH_SHAPE)

# PyTorch tensor
Whi = torch.from_numpy(Whi).cuda()
Whf = torch.from_numpy(Whf).cuda()
Who = torch.from_numpy(Who).cuda()
Whg = torch.from_numpy(Whg).cuda()
Whi = Variable(Whi, volatile=volatile)
Whf = Variable(Whf, volatile=volatile)
Who = Variable(Who, volatile=volatile)
Whg = Variable(Whg, volatile=volatile)


bH_SHAPE = (D,)
bhi = np.zeros(shape=bH_SHAPE, dtype=np.float32)
bhf = np.zeros(shape=bH_SHAPE, dtype=np.float32)
bho = np.zeros(shape=bH_SHAPE, dtype=np.float32)
bhg = np.zeros(shape=bH_SHAPE, dtype=np.float32)

# PyTorch tensor
bhi = torch.from_numpy(bhi).cuda()
bhf = torch.from_numpy(bhf).cuda()
bho = torch.from_numpy(bho).cuda()
bhg = torch.from_numpy(bhg).cuda()
bhi = Variable(bhi, volatile=volatile)
bhf = Variable(bhf, volatile=volatile)
bho = Variable(bho, volatile=volatile)
bhg = Variable(bhg, volatile=volatile)

W = gaussian(shape=(D, 10))
b = np.zeros(shape=(10,), dtype=np.float32)

# PyTorch tensor
W = torch.from_numpy(W).cuda()
b = torch.from_numpy(b).cuda()
W = Variable(W, volatile=volatile)
b = Variable(b, volatile=volatile)

N = 128
X = gaussian(shape=(N, 784 // 7, 7))

# PyTorch tensor
X = torch.from_numpy(X).cuda()
X = Variable(X, volatile=volatile)

t0 = time()
for index in range(10):
    h = Variable(torch.zeros((N, D)).cuda(), volatile=volatile)
    c = Variable(torch.zeros((N, D)).cuda(), volatile=volatile)

    for i in range(784 // 7):
        x_i = X[:, i, :]
        i = sigmoid(linear(x_i, Wxi, bxi) + linear(h, Whi, bhi))
        f = sigmoid(linear(x_i, Wxf, bxf) + linear(h, Whf, bhf))
        o = sigmoid(linear(x_i, Wxo, bxo) + linear(h, Who, bho))
        g = F.tanh(linear(x_i, Wxg, bxg) + linear(h, Whg, bhg))
        c = f * c + i * g
        h = o * F.tanh(c)
    tmp = linear(h, W, b).data.cpu().numpy()
    print((time() - t0) / (index + 1))
