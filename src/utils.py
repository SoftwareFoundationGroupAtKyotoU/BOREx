import io
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet152_Weights

from os import path as p

path_synset = p.abspath(p.join(p.dirname(__file__), "../synset_words.txt"))
if not p.exists(path_synset):
    print("File not found: {p}")
    exit(-1)

labels = np.loadtxt(path_synset, str, delimiter="\t")
def get_class_name(num:int) -> str:
    """Get class name from label number
    
    Args:
        num(int)    : Label number
    Returns:
        str : Class name
    """
    return ' '.join(labels[num].split(',')[0].split()[1:])

def get_top_n(model:nn.DataParallel, img:torch.Tensor, n:int):
    """Predict with classification model and return top N labels

    Args:
        model       (nn.DataParallel)   : ResNet model for classification
        img         (torch.Tensor)      : Input image
        n           (int)               : The number to get from the top
    Returns:
        label_list  (list(int))         : Top N labels
        prob_list   (list(np.float32))  : Probability for each label
    """
    prob = model(img)
    sorted_prob, idx = torch.sort(prob,descending = True)
    label_list = []
    for i in range(n):
        predicted_label = idx[0][i].cpu()
        label_list.append(predicted_label.item())
    return label_list, sorted_prob[0][0:10].to('cpu').detach().numpy().copy()

def get_model() -> nn.DataParallel:
    """Get model for classification

    Returns:
        nn.DataParallel: ResNet model
    """
    # Load black box model for explanations
    model = nn.Sequential(models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1), nn.Softmax(dim=1)).eval().cuda()
    for param in model.parameters():
        param.requires_grad = False
    
    # To use multiple GPUs
    model = nn.DataParallel(model)

    return model

def get_model_gcam() -> nn.DataParallel:
    """Get model for Grad-CAM

    Returns:
        nn.DataParallel: ResNet model
    """
    return nn.DataParallel(models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1).eval().cuda())

def sigmoid(a,x):
    """Sigmoid function
    
    Args:
        a
        x
    Returns:
        Any
    """
    return 1 / (1 + torch.exp(-a * x))

def auc(arr):
    """Normalized Area Under Curve of the array
    
    Args:
        arr(np.ndarray)
    Returns:
        float
    """
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

def voc_mean(arr):
    """
    Args:
        arr(np.ndarray)
    Returns:
        float
    """
    return arr.sum() / arr.shape[0]

def normalization_saliencymap(sal, mask_size):
    """Normalize to account for mask size
    
    Args:
        sal         (np.ndarray)    : Saliency map
        mask_size   (list(int))
    Returns:
        np.ndarray  : new Saliency map
    """
    sal = np.expand_dims(sal, 0)
    kernel_size = (mask_size[1] * 2 + 1, mask_size[0] * 2 + 1)
    
    sal = torch.from_numpy(sal.astype(np.float32)).clone()
    m = nn.AvgPool2d(kernel_size, stride=1, padding=(mask_size[1],mask_size[0]), count_include_pad=False)
    new_sal = m(sal)
    new_sal = new_sal.to('cpu').detach().numpy().copy()
    new_sal = np.squeeze(new_sal)
    
    return new_sal

def tensor_image(img:torch.Tensor):
    """Plots image from torch.Tensor
    
    Args:
        img(torch.Tensor)
    Returns:
        torch.Tensor
    """
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.clip(std * img.numpy().transpose((1, 2, 0)) + mean, 0, 1)
    return img

def fig2img(fig:plt.Figure, dpi=300) -> Image:
    """Figure to image with specified dpi
    
    Args:
        fig (plt.Figure)
        dpi (int, optional)
    Returns:
        Image
    """
    bf = io.BytesIO()
    fig.savefig(bf, format="jpg", dpi=dpi)
    bf.seek(0)
    img_arr = np.frombuffer(bf.getvalue(), dtype=np.uint8)
    bf.close()
    img = Image.fromarray(cv2.cvtColor(cv2.imdecode(img_arr, 1), cv2.COLOR_BGR2RGB))
    
    return img
