import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def matplotlib_imshow(img, one_channel=True):
    """ Taken from https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html """
    if one_channel:
        img = img.mean(dim=0)
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    """Taken from https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    """
    output = net.eval()(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)

    if images.size()[0] > 1:
        preds = np.squeeze(preds_tensor.cpu().numpy())
        retprobs = [F.softmax(el, dim=0)[i].item() for (i, el) in zip(preds, output)]
    else:
        preds = preds_tensor.cpu().numpy()
        retprobs = F.softmax(output)
    return preds, retprobs


def plot_classes_preds(net, images, labels, classes=None):
    """
    Taken from https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    Generates Matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    """
    if classes is None:
        classes = ["NORMAL", "BACTERIAL", "VIRAL"]
    preds, probs = images_to_probs(net, images)
    lbs = labels.cpu()
    images = images.cpu()
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(10, 20))
    for idx in np.arange(len(labels)):
        ax = fig.add_subplot(1, len(labels), idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        class_pred = classes[preds[idx]]

        p1 = probs[idx] * 100.0
        class_labels = classes[lbs[idx]]
        ax.set_title(f"{class_pred}, {p1:.2f}%\n(label: {class_labels})",
                     color=("green" if preds[idx] == lbs[idx].item() else "red"))
    return fig
