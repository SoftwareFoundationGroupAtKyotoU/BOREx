import abc
import yaml
from enum import Enum
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize

import utils

# Constant values
PE          = 0.001 
n_classes   = 1000      #total class number
C0 = 0.
C2 = 0.2510
C5 = 0.5020
C7 = 0.7529
color_dict  = {
    'aeroplane'     : [C5, C0, C0],
    'boat'          : [C0, C0, C5],
    'bicycle'       : [C0, C5, C0],
    'bus'           : [C0, C5, C5],
    'car'           : [C5, C5, C5],
    'train'         : [C5, C7, C0],
    'motorbike'     : [C2, C5, C5],
    'tvmonitor'     : [C0, C2, C5],
    'diningtable'   : [C7, C5, C0],
    'sofa'          : [C0, C7, C0],
    'bottle'        : [C5, C0, C5],
    'chair'         : [C7, C0, C0],
    'pottedplant'   : [C0, C2, C0],
    'bird'          : [C5, C5, C0],
    'cat'           : [C2, C0, C0],
    'cow'           : [C2, C5, C0],
    'dog'           : [C2, C0, C5],
    'horse'         : [C7, C0, C5],
    'sheep'         : [C5, C2, C0]   
}
## Figure
FIG_SAL         = None
FIG_PLOT        = "plot"
FIG_DELETION    = "deletion"
FIG_INSERTION   = "insertion"
FIG_RECALL      = "recall"
FIG_PRECISION   = "precision"
FIG_F           = "f"
TARGET_FIG      = [ # Figures to draw
    FIG_SAL,
    FIG_PLOT
]

class SaliencyMethod(Enum):
    """Method for generating initial saliency map"""
    RISE    = 0
    GCAMPP  = 1
    GCAM    = 2

class MeanType(Enum):
    """Mean type for acquisition function"""
    NORMAL  = 0
    POW     = 1
    ABS     = 2

class IConfig:
    """Config interface class"""
    @abc.abstractmethod
    def __init__(self):
        pass

    def save_yaml(self, path:str):
        """Save yaml

        Args:
            path(str)
        """
        data = dict()                               # Create dictionary
        for key, value in self.__dict__.items():    # Add Key,Value
            data[key] = value
        with open(path, "wt") as f:                 # Write file
            yaml.dump(data, f)

    @classmethod
    def load_yaml(cls, path:str):
        """Load yaml

        Args:
            path(str)
        """
        conf = cls()                        # Create instance
        with open(path, "rt") as f:         # Read file
            data = yaml.safe_load(f)
        for key, value in data.items():     # Set attributes
            conf.__setattr__(key, value)
        return conf

class Result:
    """BOREx result

    Attributes:
        sal         (np.ndarray)            : Saliency map
        deletion    (np.float64)            : Normalized Area Under Curve of the deletion
        insertion   (np.float64)            : Normalized Area Under Curve of the insertion
        recall      (np.float64, optional)  : VOC mean of the recall
        precision   (np.float64, optional)  : VOC mean of the precision
        f_value     (np.float64, optional)  : VOC mean of the f-value
    """
    def __init__(self, sal, deletion, insertion, recall:float=None, precision:float=None, f_value:float=None):
        self.sal        = sal
        self.deletion   = deletion
        self.insertion  = insertion
        self.recall     = recall
        self.precision  = precision
        self.f_value    = f_value

class Model:
    """ResNet model

    Attributes:
        size        (int)               : ResNet image size
        model       (nn.DataParallel)   : ResNet model for classification
        model_gcam  (nn.DataParallel)   : ResNet model for Grad-CAM
    """
    def __init__(self, size=224):
        """
        Args:
            size    (int, optional) : ResNet input size
        """
        self.size       = size
        self.model      = utils.get_model()
        self.model_gcam = utils.get_model_gcam()

    def get_confidence(self, image:torch.Tensor, class_idx:int):
        """Get confidence value

        Args:
            image       (torch.Tensor)  : Image tensor
            class_idx   (int)           : Class index
        Returns:
            float: Confidence value
        """
        
        return self.model(image)[0][class_idx]
    
    def get_confidence_array(self, image:torch.Tensor):
        """Get confidence array

        Args:
            image       (torch.Tensor)  : Image tensor
        Returns:
            list(float): Confidence value array
        """

        return self.model(image)[0]

    def get_top_n(self, img, n:int=10):
        """Predict with classification model and return top N labels

        Args:
            img         (torch.Tensor)  : Input image
            n           (int, optional) : The number to get from the top
        Returns:
            label_list  (list(int))         : Top N labels
            prob_list   (list(np.float32))  : Probability for each label
        """
        return utils.get_top_n(self.model, img, n)

class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters', leave=False):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = N
        self.p1 = p1

    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)

        p = []
        for i in range(0, N, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))

        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks[0:N,:,:].view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1

        return sal
    
class PN_RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(PN_RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters', leave=False):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1


    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)

        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        p = []
        for i in range(0, N, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))

        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W) - self.p1)
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1 / (1 - self.p1)

        return sal
    
class RISEBatch(RISE):
    def forward(self, x):
        # Apply array of filters to the image
        N = self.N
        B, C, H, W = x.size()
        stack = torch.mul(self.masks.view(N, 1, H, W), x.data.view(B * C, H, W))
        stack = stack.view(B * N, C, H, W)
        stack = stack

        #p = nn.Softmax(dim=1)(model(stack)) in batches
        p = []
        for i in range(0, N*B, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N*B)]))
        p = torch.cat(p)
        CL = p.size(1)
        p = p.view(N, B, CL)
        sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, H * W))
        sal = sal.view(B, CL, H, W)
        return sal


class GradCAM(object):
    """Calculate GradCAM salinecy map"""

    def __init__(self, arch: torch.nn.Module, target_layer: torch.nn.Module):
        self.model_arch = arch

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
        def forward_hook(module, input, output):
            self.activations['value'] = output

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


class GradCAMpp(GradCAM):
    """Calculate GradCAM++ salinecy map"""

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value'] # dS/dA
        activations = self.activations['value'] # A
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

        return saliency_map, logit