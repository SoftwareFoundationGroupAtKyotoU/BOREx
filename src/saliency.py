import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

from define import *
from utils import *

class SaliencyMap:
    """Saliency map wrapper

    Attributes:
        sal         (np.ndarray)        : Saliency map
        class_idx   (int)               : Class index
        method      (SaliencyMethod)    : Method for generating initial saliency map
    """
    def __init__(self, sal:np.ndarray, class_idx:int, method:SaliencyMethod):
        self.sal        = sal
        self.class_idx  = class_idx
        self.method     = method

    @staticmethod
    def generate(borex, img:torch.Tensor, class_idx:int, method=SaliencyMethod.RISE, rise_num=100):
        """Generate initial saliency map

        Args:
            borex       (Borex)
            img         (torch.Tensor)              : If RISE image, else image_gcam
            class_idx   (int)
            method      (SaliencyMethod, optional)
            rise_num    (int, optional)
        Returns:
            SaliencyMap: Saliency map
        """
        # Borex
        model   = borex.model
        aq_mean = borex.config.aq_mean

        # Generate saliency map
        sal = None
        if   method == SaliencyMethod.RISE:
            sal = SaliencyMap.__generate_rise(img, aq_mean, model, class_idx, rise_num)
        elif method == SaliencyMethod.GCAMPP:
            sal = SaliencyMap.__generate_gcam_pp(img, model, class_idx)
        elif method == SaliencyMethod.GCAM:
            sal = SaliencyMap.__generate_gcam(img, model, class_idx)

        return SaliencyMap(sal, class_idx, method)
    
    @staticmethod
    def __generate_rise(img:torch.Tensor, aq_mean:MeanType, model:Model, class_idx:int, rise_num:int) -> np.ndarray:
        """Generate initial saliency map by RISE

        Args:
            img         (torch.Tensor)
            aq_mean     (MeanType)
            model       (Model)
            class_idx   (int)
            rise_num    (int)
        Returns:
            np.ndarray: Saliency map
        """
        # Generate masks
        if aq_mean == MeanType.NORMAL:
            explainer = RISE(model.model, (model.size, model.size))
        else:
            explainer = PN_RISE(model.model, (model.size, model.size), gpu_batch=100)
        explainer.generate_masks(N=rise_num, s=8, p1=0.3)
        
        # RISE
        sal = explainer(img.cuda())[class_idx].cpu().numpy()
        return sal
    
    @staticmethod
    def __generate_gcam_pp(img_gcam:torch.Tensor, model:Model, class_idx:int) -> np.ndarray:
        """Generate initial saliency map by GCAM++

        Args:
            img_gcam    (torch.Tensor)
            model       (Model)
            class_idx   (int)
        Returns:
            np.ndarray: Saliency map
        """        
        # Generate Grad-CAM++ model
        gradcam_pp = GradCAMpp(model.model_gcam, model.model_gcam.module.layer4)

        # Generate saliency map
        normed_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_gcam)[None]
        mask_pp, _ = gradcam_pp(normed_img, class_idx=class_idx)
        sal = mask_pp.cpu().numpy()[0][0]
        return sal
    
    @staticmethod
    def __generate_gcam(img_gcam:torch.Tensor, model:Model, class_idx:int) -> np.ndarray:
        """Generate initial saliency map by GCAM

        Args:
            img_gcam    (torch.Tensor)
            model       (Model)
            class_idx   (int)
        Returns:
            np.ndarray: Saliency map
        """
        # Generate Grad-CAM model
        gradcam = GradCAM(model.model_gcam, model.model_gcam.module.layer4)
        
        # Generate saliency map
        normed_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_gcam)[None]
        mask, _ = gradcam(normed_img, class_idx=class_idx)
        sal = mask.cpu().numpy()[0][0]
        return sal

    def render_fig(self, image:torch.Tensor, title:str):
        """Render initial saliency map figure

        Args:
            image       (torch.Tensor)
            title       (str)
        Returns:
            plt.Figure: Figure
        """
        fig = plt.figure()
        # fig = plt.figure(num=title)   # with window title for plt.show()
        sub = fig.add_subplot()
        sub.imshow(tensor_image(image[0]))
        sub.axis("off")
        sub.set_title(title)
        im = sub.imshow(self.sal, cmap="jet", alpha=0.5)
        fig.colorbar(im)
        return fig

class TensorReader:
    """Provide read functions with specified size
    
    Attributes:
        read_tensor             (transforms.Compose)
        read_tensor_for_seg     (transforms.Compose)
        read_tensor_for_gcam    (transforms.Compose)
    """
    def __init__(self, size=224):
        # Function that opens image from disk, normalizes it and converts to tensor (for input image)
        self.read_tensor = transforms.Compose([
            lambda x: Image.open(x),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            lambda x: torch.unsqueeze(x, 0)
        ])
        # Function that opens image from disk, normalizes it and converts to tensor (for segmentation image)
        self.read_tensor_for_seg = transforms.Compose([
            lambda x: Image.open(x),
            lambda x: x.convert('RGB'),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            lambda x: torch.unsqueeze(x, 0)
        ])
        # Function that opens image from disk, normalizes it and converts to tensor (for gradcam)
        self.read_tensor_for_gcam = transforms.Compose([
            lambda x: Image.open(x),
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

class MetricsCalculator:
    """Calculator for metric of deletion/insertion and recall/precision/f-value
    
    Args:
        deletion    (np.float64)
        insertion   (np.float64)
        recall      (np.float64)
        precision   (np.float64)
        f_value     (np.float64)
    """
    def __init__(self, model:Model, class_idx:int, img:torch.Tensor, img_seg:torch.Tensor, im_i:torch.Tensor, ov:np.ndarray, Xm:list, sig:np.ndarray, ratio:int, sal:np.ndarray, substrate_fn=torch.zeros_like, voc_class:str=None):
        self.__model        = model.model
        self.__size         = model.size
        self.__class_idx    = class_idx
        self.__substrate_fn = substrate_fn

        # Plot
        self.__fig_plot     = self.render_plot(img, im_i, ov, Xm, sig, ratio)

        # Saliency map
        self.__fig_salmap   = self.render_saliency_map(img, sal)

        # Calculate deletion, insertion
        d, self.__fig_deletion    = self.calc_deletion(img, sal, draw=True)
        i, self.__fig_insertion   = self.calc_insertion(img, sal, draw=True)
        self.deletion   = auc(d)
        self.insertion  = auc(i)

        # Calculate recall, precision, f-value
        r, self.__fig_recall      = self.calc_recall(img_seg, sal, voc_class, draw=True)
        p, self.__fig_precision   = self.calc_precision(img_seg, sal, voc_class, draw=True)
        f, self.__fig_f_value     = self.calc_f_value(r, p, draw=True)
        self.recall     = voc_mean(r)
        self.precision  = voc_mean(p)
        self.f_value    = voc_mean(f)

        self.dic_fig = {
            self.__fig_salmap     : FIG_SAL,
            self.__fig_plot       : FIG_PLOT,
            self.__fig_deletion   : FIG_DELETION,
            self.__fig_insertion  : FIG_INSERTION,
            self.__fig_recall     : FIG_RECALL,
            self.__fig_precision  : FIG_PRECISION,
            self.__fig_f_value    : FIG_F,
        }

    def save_figs(self, output_dir:str):
        """Save all figures to output_dir

        args:
            output_dir  (str)
        """
        self.__output_dir = output_dir
        for fig, name in self.dic_fig.items():
            fig2img(fig).save(self.__get_fname(name))
    
    def close_figs(self):
        """Close all figures"""
        for fig, name in self.dic_fig.items():
            if name in TARGET_FIG:
                continue
            plt.close(fig)
    
    def render_plot(self, image:torch.Tensor, im_i:torch.Tensor, ov:np.ndarray, Xm:list, sig:np.ndarray, ratio:int):
        """Render plot figure
        
        Args:
            image   (torch.Tensor)
            im_i    (torch.Tensor)
            ov      (np.ndarray)
            Xm      (list)
            sig     (np.ndarray)
            ratio   (int)
        Returns:
            plt.Figure
        """
        ov = np.mean(ov, axis=-1)
        fig = plt.figure(figsize=(12, 4), num=f"{get_class_name(self.__class_idx)} Plot" )
        
        sub1 = plt.subplot(131)
        sub1.axis("off")
        sub1.imshow(tensor_image(im_i[0]))
        sub1.plot(np.array(Xm)[:-2,1],np.array(Xm)[:-2,0],'*')
        sub1.plot(np.array(Xm)[-2,1],np.array(Xm)[-2,0],'r*')

        sub2 = plt.subplot(132)
        sub2.axis('off')
        sub2.imshow(tensor_image(image[0]))
        sub2.imshow(ov.T, cmap='jet', alpha=0.5)
        sub2.plot(np.array(Xm)[:-2,1],np.array(Xm)[:-2,0],'*')
        sub2.plot(np.array(Xm)[-2,1],np.array(Xm)[-2,0],'r*')
        sub2.set_title("Mean sensitivity")

        sxy = np.sum(sig.reshape(self.__size, self.__size, -1), -1)

        sub3 = plt.subplot(133)
        sub3.imshow(ratio*sxy.T+ov.T)
        sub3.plot(np.array(Xm)[:-2,1],np.array(Xm)[:-2,0],'*')
        sub3.plot(np.array(Xm)[-2,1],np.array(Xm)[-2,0],'r*')
        sub3.set_title('Acquisition function')

        return fig

    def render_saliency_map(self, img:torch.Tensor, sal:np.ndarray):
        """Render saliency map figure
        
        Args:
            img (torch.Tensor)
            sal (np.ndarray)
        Returns:
            plt.Figure
        """
        fig = plt.figure(num=f"{get_class_name(self.__class_idx)} Saliency map")
        sub = fig.add_subplot()
        sub.imshow(tensor_image(img[0]))
        im = sub.imshow(sal, cmap="jet")
        sub.axis("off")
        fig.colorbar(im)

        return fig

    def calc_deletion(self, img:torch.Tensor, sal:np.ndarray, draw=False):
        """Calculate deletion & Render deletion figure

        Args;
            img         (torch.Tensor)
            sal         (np.ndarray)
            path        (str, optional)
        Returns:
            np.ndarray
            plt.Figure
        """
        self.start      = img.clone()
        self.finish     = self.__substrate_fn(img)
        self.__title    = "Deletion game"
        self.__ylabel   = "Pixels deleted"
        return self.__calc_metric(sal, draw)

    def calc_insertion(self, img:torch.Tensor, sal:np.ndarray, draw=False):
        """Calculate insertion & Render insertion figure

        Args;
            img         (torch.Tensor)
            sal         (np.ndarray)
            path        (str, optional)
        Returns:
            np.ndarray
            plt.Figure
        """
        self.start      = self.__substrate_fn(img)
        self.finish     = img.clone()
        self.__title    = "Insertion game"
        self.__ylabel   = "Pixels inserted"
        return self.__calc_metric(sal, draw)

    def __calc_metric(self, sal:np.ndarray, draw:bool):
        """Calculate value & Render figure
        
        Args:
            sal     (np.ndarray)
            draw    (bool)
        Returns:
            np.ndarray
            plt.Figure
        """
        HW      = self.__size**2
        n_steps = (HW + self.__size - 1) // self.__size

        scores = np.empty(n_steps+1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(sal.reshape(-1, HW), axis=1), axis=-1)
        for i in range(n_steps+1):
            pred = self.__model(self.start.cuda())
            scores[i] = pred[0, self.__class_idx]
            
            if i < n_steps:
                coords = salient_order[:, self.__size*i:self.__size*(i+1)]
                self.start.cpu().numpy().reshape(1, 3, HW)[0, :, coords] = self.finish.cpu().numpy().reshape(1, 3, HW)[0, :, coords]
            elif draw:
                # Render
                fig = plt.figure(figsize=(10, 5), num=f"{get_class_name(self.__class_idx)} {self.__title}")
                
                figp = fig.add_subplot(121)
                figp.set_title(f"{self.__ylabel} {100*i/n_steps:.1f}%, P={scores[i]:.4f}")
                figp.axis("off")
                figp.imshow(tensor_image(self.start[0]))

                figg = fig.add_subplot(122)
                figg.plot(np.arange(i+1)/n_steps, scores[:i+1])
                figg.set_xlim(-0.1, 1.1)
                figg.set_ylim(0, 1.05)
                figg.fill_between(np.arange(i+1)/n_steps, 0, scores[:i+1], alpha=0.4)
                figg.set_title(self.__title)
                figg.set_xlabel(self.__ylabel)
                figg.set_ylabel(get_class_name(self.__class_idx))

                return scores, fig

        return scores, None
    
    def calc_recall(self, img_seg:torch.Tensor, sal:np.ndarray, voc_class:int, draw=False):
        """Calculate recall & Render recall figure
        
        Args:
            img_seg     (torch.Tensor)
            sal         (np.ndarray)
            voc_class   (int)
            draw        (bool, optional)
        Returns:
            np.ndarray
            plt.Figure
        """
        class_color = color_dict[voc_class]
        n_steps = self.__size**2    # HW

        seg = img_seg.clone()

        # 値を計算
        scores = np.empty(n_steps)
        total_obj_px = 0
        for y in range(self.__size): # Height
            for x in range(self.__size): # Width
                if abs(img_seg[0][0][y][x].item() - class_color[0]) < PE  and abs(img_seg[0][1][y][x].item() - class_color[1]) < PE and abs(img_seg[0][2][y][x].item() - class_color[2]) < PE:
                    total_obj_px += 1
        
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(sal.reshape(-1, n_steps), axis=1), axis=-1)

        obj_px = 0
        for i in range(n_steps):
            coords = salient_order[:, i]
            color = seg.cpu().numpy().reshape(1, 3, n_steps)[0, :, coords]

            if abs(color[0][0] - class_color[0]) < PE and abs(color[0][1] - class_color[1]) < PE and abs(color[0][2] - class_color[2]) < PE:
                obj_px += 1
            if total_obj_px > 0:
                scores[i] = obj_px / total_obj_px
            else:
                scores[i] = 0

        if draw:
            self.__title    = "Recall"
            self.__xlabel   = "Pixels inserted"
            self.__ylabel   = "recall"
            fig = self.__render_values(np.arange(n_steps)/n_steps, scores)
            return scores, fig

        return scores, None
    
    def calc_precision(self, img_seg:torch.Tensor, sal:np.ndarray, voc_class:int, draw=False):
        """Calculate precision & Render precision figure
        
        Args:
            img_seg     (torch.Tensor)
            sal         (np.ndarray)
            voc_class   (int)
            draw        (bool, optional)
        Returns:
            np.ndarray
            plt.Figure
        """
        class_color = color_dict[voc_class]
        n_steps = self.__size**2    # HW

        seg = img_seg.clone()

        scores = np.empty(n_steps)

        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(sal.reshape(-1, n_steps), axis=1), axis=-1)
        
        obj_px = 0
        for i in range(n_steps):
            coords = salient_order[:, i]
            color = seg.cpu().numpy().reshape(1, 3, n_steps)[0, :, coords]

            if abs(color[0][0] - class_color[0])<PE and abs(color[0][1]-class_color[1])<PE and abs(color[0][2]-class_color[2])<PE:
                obj_px += 1
            scores[i] = obj_px / (i+1)

        if draw:
            self.__title    = "Precision"
            self.__xlabel   = "Pixels inserted"
            self.__ylabel   = "precision"
            fig = self.__render_values(np.arange(n_steps)/n_steps, scores)
            return scores, fig

        return scores, None

    def calc_f_value(self, recall:np.ndarray, precision:np.ndarray, draw=False):
        """Calculate f-value & Render f-value figure
        
        Args:
            recall      (np.ndarray)
            precision   (np.ndarray)
            draw        (bool, optional)
        Returns:
            np.ndarray
            plt.Figure
        """
        n_steps = len(recall)
        scores = recall * precision

        if draw:
            self.__title    = "F value"
            self.__xlabel   = "Pixels inserted"
            self.__ylabel   = "f value"
            fig = self.__render_values(np.arange(n_steps)/n_steps, scores)
            return scores, fig
        
        return scores, None

    def __render_values(self, pins, scores) -> plt.Figure:
        """Render figure by x-axis & y-axis values"""
        fig = plt.figure(figsize=(5, 5), num=f"{get_class_name(self.__class_idx)} {self.__title}")
        sub = fig.add_subplot(111)
        sub.plot(pins, scores)
        sub.set_xlim(-0.1, 1.1)
        sub.set_ylim(0, 1.05)
        sub.fill_between(pins, 0, scores, alpha=0.4)
        sub.set_title(self.__title)
        sub.set_xlabel(self.__xlabel)
        sub.set_ylabel(self.__ylabel)

        return fig

    def __get_fname(self, suffix:str=None) -> str:
        """Get filename from properties"""
        class_name = get_class_name(self.__class_idx)
        if suffix == None:
            return os.path.join(self.__output_dir, f"{self.__class_idx}_borex_{class_name}.jpg")
        return os.path.join(self.__output_dir, f"{self.__class_idx}_borex_{class_name}_{suffix}.jpg")