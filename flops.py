import argparse
import torch

from train import build_model
from models.convmixer import ConvMixer, ConvMixerBlock
from models.densenet import DenseNet, DenseBlock
from models.fractalnet import FractalNet, FractalBlock
from models.resnet import ResNet, ResBlock
from models.preactresnet import PreActResNet, PreActResBlock
from models.stochastic_depth import StDepth, StDepthBlock
from models.mlpmixer import MLPMixer, MixerBlock
from models.vision_transformer import VisionTransformer, Encoder

from pthflops import count_ops


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ResNet", 
                        choices=["ResNet", "PreActResNet", "DenseNet", "StDepth", "ConvMixer", "MLPMixer", "ViT", "FractalNet"])
    args = parser.parse_args()
    model = build_model(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X = torch.rand(1, 3, 32, 32).to(device)
    print(count_ops(model, X))