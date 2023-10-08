import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import torch
from sklearn.gaussian_process import GaussianProcessRegressor

# To ignore sklearn warning
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, os.path.dirname(__file__))
from define import *
from saliency import *

class BorexConfig(IConfig):
    """BOREx config

    Attributes:
        output_dir      (str)           : Directory to output result
        iter_num        (int)           : Number of iterations for Bayesian optimization
        normalize       (bool)          : Whether normalize for mask size when generating saliency map or not
        sig_gain        (int)           : Strength of the sigmoid function when computing saliency
        aq_mean         (MeanType)      :
            If 'NORMAL', use the mean used in the acquisition function as it is

            If 'POW', use the square of the mean

            If 'ABS', use the absolute value of the mean
        ratio           (int)           : Ratio of mean to variance in the acquisition function
        mask_size_list  (list(int))     : List of mask sizes
        jitter          (list(float))   : Jitter for not sticking to grid
        magnification   (float)         : Magnification to normalize at preprocessing
        is_save         (bool)          : Whether save Figures or not
        is_show         (bool)          : Whether show Saliency map or not
    """
    def __init__(self, output_dir="", iter_num=50, normalize=False, sig_gain=0, aq_mean=MeanType.NORMAL, ratio=10, mask_size_list=[30, 45, 60, 75], jitter=[1 , 1, 0.5], magnification=0.1, is_save=True, is_show=False):
        self.output_dir     = output_dir
        self.iter_num       = iter_num
        self.normalize      = normalize
        self.sig_gain       = sig_gain
        self.__aq_mean      = int(aq_mean.value)
        self.ratio          = ratio
        self.mask_size_list = mask_size_list
        self.jitter         = jitter
        self.magnification  = magnification
        self.is_save        = is_save
        self.is_show        = is_show

    @property
    def aq_mean(self) -> MeanType:
        return MeanType(self.__aq_mean)
    @aq_mean.setter
    def aq_mean(self, v: MeanType):
        self.__aq_mean = int(v.value)

class Borex:
    """Borex

    Attributes:
        config  (BorexConfig)   : BOREx config
        model   (Model)         : ResNet
    """
    def __init__(self, config:BorexConfig, img_size=224):
        self.config = config
        self.model  = Model(size=img_size)

    @ignore_warnings(category=ConvergenceWarning)
    def run(self, image:torch.Tensor, image_seg:torch.Tensor, sal_map:SaliencyMap, gp:GaussianProcessRegressor, voc_class:str = None):
        """Run BOREx, and get result.

        Args:
            image       (torch.Tensor)              : Input image
            image_seg   (torch.Tensor)              : Segmentation image
            sal_map     (SaliencyMap)               : Initial saliency map
            gp          (GaussianProcessRegressor)  : Gaussian Process Regressor
            voc_class   (str, optional)             : Class name in Pascal VOC
        Returns:
            Result  : BOREx result
        """
        # var
        mask_size_list  = self.config.mask_size_list
        size            = self.model.size
        sig_gain        = self.config.sig_gain
        aq_mean         = self.config.aq_mean
        ratio           = self.config.ratio
        is_save         = self.config.is_save
        is_show         = self.config.is_show
        sal             = sal_map.sal
        class_idx       = sal_map.class_idx

        # Save & Show initial saliency map
        path_ini_sal = os.path.join(self.config.output_dir, f"{class_idx}_borex_{get_class_name(class_idx)}_init.jpg")
        fig = sal_map.render_fig(image, get_class_name(class_idx))
        if is_save:
            fig2img(fig).save(path_ini_sal)
        plt.close(fig)

        # Normalize sal
        sal_mean = np.mean(np.abs(sal))
        sal *= self.config.magnification/sal_mean

        duplicated_sal = np.tile(sal, (len(mask_size_list),1,1))
        flat_sal = duplicated_sal.T.flatten()

        # Create sampling grid(position, and blanking window size)
        x,y,s = np.meshgrid(np.linspace(0,size-1,size),np.linspace(0,size-1,size),np.array(mask_size_list))
        X = np.vstack((x.ravel(), y.ravel(), s.ravel())).T

        # Storage for samples
        Xm = []
        fm = []

        # First sample
        max_point = np.argmax(sal).astype(int)
        x = max_point %  size
        y = max_point // size
        state = [y, x, mask_size_list[0]]
        Xm.append(state)

        for i in range(self.config.iter_num):
            # Blank image at sample locationf
            im_o = image.clone()    # outside
            im_i = image.clone()    # inside

            #　Mask the outside of the rectangle
            im_o[:, :, 0:min(size,state[0]+state[2])    , 0:max(state[1]-state[2],0)]       = 0 # upper left
            im_o[:, :, min(size,state[0]+state[2]):size , 0:min(size,state[1]+state[2])]    = 0 # upper right
            im_o[:, :, max(state[0]-state[2],0):size    , min(size,state[1]+state[2]):size] = 0 # bottom right
            im_o[:, :, 0:max(state[0]-state[2],0)       , max(state[1]-state[2],0):size]    = 0 # bottom left
        
            #　Mask the inside of the rectangle
            im_i[:, :, max(state[0]-state[2],0):min(size,state[0]+state[2]), max(state[1]-state[2],0):min(size,state[1]+state[2])] = 0

            # Predict by classification model
            logit_i = self.model.get_confidence(im_i, class_idx)
            logit_o = self.model.get_confidence(im_o, class_idx)

            # Applying the sigmoid function if sig_gain > 0
            score = logit_o-logit_i
            if sig_gain > 0:
                score = 2 * (sigmoid(sig_gain, score) - 0.5)
            
            # Storage saliency
            fm.append(score - sal[state[0]][state[1]])

            # Fit surrgate model
            fm_np = []
            for tensor in fm:
                fm_np.append(tensor.cpu().numpy().copy())
            gp.fit(np.array(Xm), np.array(fm_np))

            # Calculate the mean and variance
            mu, sig = gp.predict(X, return_std=True)

            # Calculate aquisition function
            mu += flat_sal
            if aq_mean == MeanType.ABS:
                mu_aq = np.abs(mu)
            elif aq_mean == MeanType.POW:
                mu_aq = np.sqrt(mu**2)
            else:
                mu_aq = mu
            aq_val = ratio * sig.reshape(-1, 1) + mu_aq.reshape(-1, 1)

            # Choose next blanking point, tradeing off exploration and exploitation
            aq_bin = np.argmax(aq_val)
            state = X[aq_bin].astype(int)
            Xm.append(state+np.random.randn(3,)*self.config.jitter) # Add some jitter so we aren't stuck to grid

            ov = mu.reshape(size, size, len(mask_size_list))
        
        if self.config.normalize:
            new_ov = []
            for k in range(len(mask_size_list)):
                sal = normalization_saliencymap(ov[:,:,k], [mask_size_list[k],mask_size_list[k]])
                new_ov.append(sal)
            new_ov = np.mean(np.array(new_ov), axis=0)
            
            sal = new_ov.T
        else:
            sal = ov.T
        
        # Calculate & Plot
        calculator  = MetricsCalculator(self.model, class_idx, image, image_seg, im_i, ov, Xm, sig, ratio, sal, voc_class=voc_class)
        if is_save:
            calculator.save_figs(self.config.output_dir)
        calculator.close_figs()

        insertion   = calculator.insertion
        deletion    = calculator.deletion
        recall      = calculator.recall
        precision   = calculator.precision
        f_value     = calculator.f_value
        
        return Result(sal, deletion, insertion, recall, precision, f_value)
