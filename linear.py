import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_layers, in_features, out_features):
        super(MLP, self).__init__()
        self._n_layers = n_layers
        setattr(self, '_in', nn.Linear(in_features, out_features))
        for i in range(n_layers):
            linear = nn.Linear(out_features, out_features)
            setattr(self, 'layer%d' % i, linear)
        setattr(self, '_out', nn.Linear(out_features, 1))

    def forward(self, data):
        data = self._in(data)
        for i in range(self._n_layers):
            linear = getattr(self, 'layer%d' % i)
            data = linear(data)
        data = self._out(data)
        return data

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--in_features', type=int, default=1)
parser.add_argument('--n_iterations', type=int, default=100)
parser.add_argument('--n_layers', type=int)
parser.add_argument('--out_features', type=int, default=1)
args = parser.parse_args()

model = MLP(args.n_layers, args.in_features, args.out_features)
criterion = nn.MSELoss()

from torch.optim import SGD
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

model.cuda()
model.train()

from torch.autograd import Variable
import torch as th
data = Variable(th.rand(args.batch_size, args.in_features).cuda())
targets = Variable(th.zeros(args.batch_size).cuda())

from time import time

ft = 0
bt = 0

for _ in range(args.n_iterations):
    t0 = time()
    out = model(data)
    ft += time() - t0

    loss = criterion(out, targets)

    t0 = time()
    loss.backward()
    bt += time() - t0

ft /= args.n_iterations
bt /= args.n_iterations

print ft, bt

arg_dict = vars(args)
keys = sorted(arg_dict.keys())
identifier = '-'.join('%s-%d' % (k, arg_dict[k]) for k in keys)
path = 'info/%s' % identifier

import cPickle as pickle
pickle.dump((ft, bt), open(path, 'w'))
