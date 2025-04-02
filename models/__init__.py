from .convmixer import ConvMixer, ConvMixerBlock
from .densenet import DenseNet, DenseBlock
from .fractalnet import FractalNet, FractalBlock
from .resnet import ResNet, ResBlock
from .preactresnet import PreActResNet, PreActResBlock
from .stochastic_depth import StDepth, StDepthBlock
from .mlpmixer import MLPMixer, MixerBlock
from .vision_transformer import VisionTransformer, VisionTransformerBlock


def build_model(args):

    if args.model == "ResNet":         
        model = ResNet(ResBlock, channel_list=[16, 32, 64], num_blocks_list=[18, 18, 18], dataset="cifar-10")
    elif args.model == "PreActResNet":
        model = PreActResNet(PreActResBlock, channel_list=[16, 32, 64], num_blocks_list=[18, 18, 18], dataset="cifar-10")
    elif args.model == "DenseNet":
        model = DenseNet(DenseBlock, num_blocks_list=[13, 13, 13], growth_rate=12, in_channels=16, compression_rate=1, num_classes=10)
    elif args.model == "StDepth":
        model = StDepth(StDepthBlock, channel_list=[16, 32, 64], num_blocks_list=[18, 18, 18], num_classes=10)
    elif args.model == "FractalNet":
        model = FractalNet(col=3, channel_list=[64, 128, 256, 512, 512])
    elif args.model == "MLPMixer":
        model = MLPMixer(image_size=(32, 32), hidden_dim=512, patch_size=4, c_hidden=512*4, s_hidden=256, depth=12, num_classes=10)
    elif args.model == "ViT":
        model = VisionTransformer(image_size=(32, 32), hidden_dim=384, num_heads=6, mlp_dim=64*4, patch_size=4, depth=8, num_classes=10)
    elif args.model == "ConvMixer":
        model = ConvMixer(hidden_dim=256, depth=16, patch_size=1, kernel_size=8, num_classes=10)
    else: 
        print("Check if the model name is correct")
        return  

    return model
