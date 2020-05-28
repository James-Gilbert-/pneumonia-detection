from torch import nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck
from torch.hub import load_state_dict_from_url

class MedNet(models.ResNet):
    """ Simple transfer learning for a medical image task instead of CIFAR """

    def __init__(self, num_classes):
        super(MedNet, self).__init__(block=Bottleneck, layers=[3, 8, 36, 3])
        block = 0
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet152-b121ed2d.pth",
                                              progress=True)
        self.load_state_dict(state_dict)
        for child in self.children():
            block+=1
            if block < 7:
                for param in child.parameters():
                    param.requires_grad=False

        # instead of many classes, there are three classes, so a trainable, fully
        # connected layer is added
        nf = self.fc.in_features
        self.fc = nn.Sequential(nn.Linear(nf, num_classes,bias=True))