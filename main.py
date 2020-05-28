import os
from argparse import ArgumentParser, Namespace
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from model import MedResNet
from utils import custom_dataset
from utils import data_utils

classes = ["NORMAL", "BACTERIAL", "VIRAL"]

global device
if torch.cuda.is_available():
    device = torch.device("cuda:0")


def block(in_chan, out_chan, ksize, padding, dropout=None):
    """Creates a standard block comprised of a convolutional layer, batch norm, dropout, and LeakyReLU activation"""
    layers = [nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=ksize, padding=padding),
              nn.BatchNorm2d(out_chan)]
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.LeakyReLU())

    return layers


def fc(in_features, out_features, dropout=None):
    """Creates a fully connected layer with dropout"""

    fc_layer = [nn.Linear(in_features, out_features)]
    if dropout:
        fc_layer.append(nn.Dropout(dropout))
    return fc_layer


class MedNet(nn.Module):
    """A basic convolutional net for medical image classification. Unused in training. """

    def __init__(self, hparams):
        super(MedNet, self).__init__()

        lc = 64
        self.hparams = hparams

        self.net = nn.ModuleList()
        self.net.extend(
            block(1, lc, 3, 1) +
            block(lc, lc * 2, 3, 1, dropout=0.1) +
            block(lc * 2, lc, 3, 1, dropout=0.1))

        self.fcl = nn.ModuleList()
        self.fcl.extend(fc(hparams.size * hparams.size * lc, lc, 0.1))
        self.fcl.extend(fc(lc, 3))

    def forward(self, x):

        for layer in self.net:
            x = layer(x)
        features = np.prod(x.size()[1:])
        x = x.view(-1, features)
        for fc_l in self.fcl:
            x = fc_l(x)
        return x


def xray_dataloader(transform_list, data_params, split='train', length=None):
    """Loads the X-ray train set into a data loader"""

    data_dir = os.path.join(data_params.data_path, split)
    dataset_xray = custom_dataset.XrayDataset(data_dir, transform_list)
    if split == 'test':
        sampler_ = torch.utils.data.RandomSampler(
            dataset_xray, replacement=True, num_samples=length)
        xray_loader = torch.utils.data.DataLoader(
            dataset=dataset_xray, sampler=sampler_, **data_params.data_cfg)
    else:
        xray_loader = torch.utils.data.DataLoader(
            dataset=dataset_xray, shuffle=True, **data_params.data_cfg)
    return xray_loader


def train(hp: Namespace):
    """
    This method trains the model with the given hyperparameters, data path,
    checkpoint directory,log directory, and name of the run for the tensorboard
    """
    hp = SimpleNamespace(**vars(hp))
    data_loader_config = {
        "batch_size": hp.batch_size,
        "num_workers": 4,
        "pin_memory": True
    }
    hp.data_cfg = data_loader_config

    # the training transforms also add random affine for improved generalization
    # normalization values are for ResNet
    tr_list_train = [transforms.RandomResizedCrop(hp.size),
                     transforms.RandomAffine(0.001),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])]
    tr_list = [transforms.Resize((hp.size, hp.size)),
               transforms.CenterCrop(hp.size),
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])]

    # load the model
    num_classes = 3
    model = MedResNet.MedNet(num_classes).to(device)

    # load the test and train sets
    train_dataloader = xray_dataloader(
        transform_list=tr_list_train,
        split='train', data_params=hp)
    train_len = len(train_dataloader)
    val_dataloader = xray_dataloader(
        transform_list=tr_list,
        data_params=hp,
        split='test',
        length=train_len)

    # loss, optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=hp.lr)

    # initializing structures for logging the training and validation losses
    losses_train = deque(maxlen=100)
    losses_val = deque(maxlen=100)
    writer = SummaryWriter(log_dir=os.path.join(hp.tf_logs, hp.run_name))
    correct = 0
    total = 0
    class_correct = list(0. for _ in range(3))
    class_total = list(0. for _ in range(3))
    for epoch in range(hp.epochs):
        for idx, (train_imgset, val_imgset) in enumerate(zip(train_dataloader, val_dataloader)):
            optimizer.zero_grad()

            imgsets = train_imgset['image'].to(device)
            labels = train_imgset['label'].to(device)
            outputs = model(imgsets)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses_train.append(loss.item())

            with torch.no_grad():

                val_imgsets, val_labels = val_imgset['image'].to(device), val_imgset['label'].to(device)
                output_val = model(val_imgsets)
                _, predicted = torch.max(output_val.data, 1)

                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

                losses_val.append(criterion(output_val, val_labels).item())
                c = (predicted == val_labels).squeeze()

                for i in range(len(val_labels)):
                    label_n = val_labels[i]
                    class_correct[label_n] += c[i].item()
                    class_total[label_n] += 1

            if idx > 0 and idx % hp.losscount == 0:
                for i in range(3):
                    print('Accuracy of %5s : %2d %%' % (
                        classes[i], 100 * class_correct[i] / class_total[i]))
                print(
                    f"EPOCH {epoch} idx {idx} / {len(train_dataloader)}\
                    Train Loss: {sum(losses_train) / hp.losscount} ||| Val Loss: {sum(losses_val) / hp.losscount}")

                if idx % hp.losscount == 0:
                    print(f'Accuracy of the network: %d %%' % (
                            100 * correct / total))
                writer.add_scalars('Loss', {
                    "Train Loss": sum(losses_train) / hp.losscount, "Val Loss": sum(losses_val) / hp.losscount},
                                   epoch * len(train_dataloader) + idx)

                if hp.data_viz:
                    writer.add_figure('predicted vs actual',
                                      data_utils.plot_classes_preds(model, imgsets, labels=labels, classes=classes),
                                      global_step=epoch * len(train_dataloader) + idx)

                writer.flush()

        # checkpoints are saved in the given directory under a specific run name
        path = f'{hp.checkpoints}/{hp.run_name}_e_{epoch}.pth'
        torch.save(model.state_dict(), path)


def test(hp: Namespace):
    """Simple function to calculate summary stats from a trained model"""
    hp = SimpleNamespace(**vars(hp))
    data_loader_config = {
        "batch_size": hp.batch_size,
        "num_workers": 4,
        "pin_memory": True
    }
    hp.data_cfg = data_loader_config
    # load the model
    num_classes = 3
    model = MedResNet.MedNet(num_classes)
    model_dict = model.state_dict()
    state_dict = torch.load(hp.saved_model)

    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)

    # load and transform the data
    tr_list = [transforms.Resize((hp.size, hp.size)),
               transforms.CenterCrop(hp.size),
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])]
    validation_data = xray_dataloader(transform_list=tr_list, data_params=hp, split='test')


    with torch.no_grad():
        y_true = []
        y_pred = []
        labels = [0,1,2]

        for iter, batch in enumerate(validation_data):

            output = model(batch['image'])
            _, predicted = torch.max(output.data, 1)
            y_true.extend(batch['label'])
            y_pred.extend(predicted)


        f1 = f1_score(y_true,y_pred,labels,average='weighted')
        print(f"f1 score is {f1}")


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a transfer learner with specified hparams, data, '
                                        'and directories')
    parser.add_argument("--data_path", type=str, help="path where dataset is stored")
    parser.add_argument("--run_name", type=str, default="run", help="name of the training run")
    parser.add_argument("--batch_size", type=int, default=5, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--losscount", type=int, default=3, help="number of batches to train before writing to logs")
    parser.add_argument("--checkpoints", type=str, default='./model/checkpoints', help="model checkpoint directory")
    parser.add_argument("--tf_logs", type=str, default='./logs', help="training logs directory for tfboard")
    parser.add_argument("--size", type=int, default=224,
                        help="size of the train image, given as length of one side")
    parser.add_argument("--data_viz", type=bool, default=True,
                        help="enable data visualization")
    parser.add_argument("--stage", type=str, default="train")
    parser.add_argument("--saved_model", type=str, default=None)

    params = parser.parse_args()
    if params.stage == "train":
        train(params)
    else:
        test(params)
