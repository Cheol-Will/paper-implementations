import argparse, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)

CUR_DIR = os.getcwd()
parser = argparse.ArgumentParser()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.covn1 = nn.Conv2d(3, 3, 3, padding = 1)
        self.covn2 = nn.Conv2d(3, 3, 3, padding = 1)

    def forward(self, x):
        if self.training:
            if torch.bernoulli(torch.tensor(0.5)) == 1:
                # local drop-path
                b = self.covn2(self.covn1(x))
            else:
                # global drop-path
                b = self.covn1(x)
            return b
        else:
            # whole network
            b = self.covn2(self.covn1(x))
            return b

    def join(self, *feature_maps):
        f_stacked = torch.stack(feature_maps, )
        f_join = torch.mean(f_stacked, dim = 0)
        return f_join

class FractalBlock(nn.Module):

    def __init__(self, column, in_channels, drop_prop):
        super(FractalBlock, self).__init__()
        self.column = column
        self.in_channels = in_channels
        self.layers = self.make_layer()
        self.join_idx = self.make_join_idx()
        self.drop_prop = drop_prop

    def make_layer(self):
        layers=  nn.ModuleList()

        for col in range(self.column):
            layers.append(nn.ModuleList(
                [nn.Conv2d(self.in_channels, self.in_channels, 3, padding = 1) for _ in range(2**(col))]
            ))

        return layers

    def make_join_idx(self):
        C = self.column
        num_joins = 2**(C-2)
        join_idx =[[C-2, C-1] for _ in range(num_joins)]

        for c in range(C-2-1, -1, -1):
            for j in range(num_joins):
                if (j+1) % (2**(C-2 - c)) == 0:
                    join_idx[j].insert(0, c)
        return join_idx

    def forward(self, x):

        if self.training: # train mode
            # choose local or global drop-path randomly (50%)
            if np.random.binomial(1, 0) == 1:
                # local drop path : pick one subpath 
                out = x.clone()

            else:
                # global drop path : pick one single column and perform forward (dont care join layer)
                c = torch.randint(low = 0, high = 4, size = (1, )).item()
                out = x.clone()
                for layer in self.layers[c]:
                    out = layer(out)

        else: # eval mode
            # calculate output layer with all layer
            out_layers = [x.clone() for _ in range(self.column)]
            # for join_idx in range(2**(self.columns-2)):
            for layer_idx in range(2**(self.column-1)):
                print(f"layer_idx: {layer_idx}")
                for c in range(self.column):
                    if (layer_idx+1) % 2**(self.column-1-c) == 0:
                        out_layers[c] = self.layers[c][(layer_idx+1) // 2**(self.column-1-c) - 1](out_layers[c])

                # after appropriate conv in each column, perform join when it's needed
                if (layer_idx+1) % 2 == 0:
                    # after even-numbered layer, always perform join with corresponding columns and update
                    temp = self.join(*[out_layers[idx] for idx in self.join_idx[layer_idx//2]])
                    for idx in self.join_idx[layer_idx//2]:
                        out_layers[idx] = temp
            # when using all columns, out_layers has exactly the same elements
            out = out_layers[0]
        # for 
        return out

        
    def join(self, *feature_maps):
        # stack all feature maps and averaging
        f_stacked = torch.stack(feature_maps)
        f_join = torch.mean(f_stacked, dim = 0)
        return f_join



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FractalBlock(4, 3, 0.1)
    model.to(device) 

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

    # print("train mode", "-"*50)
    # model.train()
    # for _ in range(5):
    #     X = torch.zeros((3, 4, 4), device = device)
    #     pred = model(X)
    #     print(pred.shape)
    #     print(pred[0])

    print("eval mode", "-"*30)
    model.eval()
    X = torch.zeros((3, 4, 4), device = device)
    pred = model(X)
    print(pred)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)