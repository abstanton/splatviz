"""
Widget that segments an object using sam2. It takes in the 3d points of the object and the rendered orbital views and segments the object using sam2.
It takes the segmented object in each view and uses it to filter the gaussian splat.
"""
import cv2
import os
import numpy as np
import torch
from imgui_bundle import imgui
import PIL.Image

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from splatviz_utils.cam_utils import world_point_to_pixel_and_depth
from widgets.widget import Widget
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


SAM2_DIRECTORY = "../sam2"

class ObjectSegmentationWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Object Segmentation")
        
        checkpoint = os.path.join(SAM2_DIRECTORY, "checkpoints/sam2.1_hiera_large.pt")
        model_cfg = os.path.join(SAM2_DIRECTORY, "configs/sam2.1/sam2.1_hiera_l.yaml")
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        self.masks = []

    def _get_orbital_views(self):
        views = getattr(self.viz.store, "orbital_views", None)
        if views is None:
            return []
        return views

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            if imgui_utils.button("Run object segmentation", width=viz.button_large_w):
                self._run_segmentation()

    def _run_segmentation(self):
        orbital_views = self._get_orbital_views()
        if len(orbital_views) < 2:
            return

        self.masks = []
        for i, view in enumerate(orbital_views):
            image = view["image"]
            self.predictor.set_image(image)
            points_2d_and_status = view["points_2d_and_status"]
            points_2d = torch.tensor([p[:2] for p in points_2d_and_status], device="cuda")
            point_labels = torch.ones_like(points_2d[:,0])
            res = self.predictor.predict(points_2d, point_labels)
            mask = res[0][2]
            self.masks.append(mask)


          
        