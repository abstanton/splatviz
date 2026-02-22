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
from gaussian_splatting.scene.gaussian_model import GaussianModel



SAM2_DIRECTORY = "../sam2"

class ObjectSegmentationWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Object Segmentation")
        
        checkpoint = os.path.join(SAM2_DIRECTORY, "checkpoints/sam2.1_hiera_large.pt")
        model_cfg = os.path.join(SAM2_DIRECTORY, "configs/sam2.1/sam2.1_hiera_l.yaml")
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        self.masks = []
        self.gaussian_model = None
        self.gs_point_mask_counts = None
        self.gs_point_mask_threshold = 10
        self.depth_threshold = 0.05
        self.curr_ply_file_path = ""

    def _get_orbital_views(self):
        views = getattr(self.viz.store, "orbital_views", None)
        if views is None:
            return []
        return views

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        ply_file_path = viz.args.ply_file_paths[0]
        if ply_file_path != self.curr_ply_file_path:
            self.gaussian_model = self._load_model(ply_file_path)
            self.curr_ply_file_path = ply_file_path

        if show:
            if imgui_utils.button("Run object segmentation", width=viz.button_large_w):
                self._run_segmentation()
            if imgui_utils.button("Clear segmentation", width=viz.button_large_w):
                self.gs_point_mask_counts = None
            label("Threshold views", viz.label_w)
            _, self.gs_point_mask_threshold = imgui.input_int("##gs_point_mask_threshold", self.gs_point_mask_threshold, 1, 32)
            label("Depth threshold", viz.label_w)
            _, self.depth_threshold = imgui.input_float("##depth_threshold", self.depth_threshold, 0.01, 2.0)
        
        self.viz.args.points_mask_count = self.gs_point_mask_counts
        self.viz.args.points_mask_threshold = self.gs_point_mask_threshold

    def _load_model(self, ply_file_path):
        if ply_file_path.endswith(".ply"):
            model = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)
            model.load_ply(ply_file_path)
        else:
            raise NotImplementedError("Select a .ply or .yml file.")
        return model

    def _run_segmentation(self):
        orbital_views = self._get_orbital_views()
        if len(orbital_views) < 2:
            return
        if self.gaussian_model is None:
            return

        points = self.gaussian_model.get_xyz.cuda()
        gs_point_mask_counts = torch.zeros_like(points[...,0])     

        for i, view in enumerate(orbital_views):
            if view is None:
                continue

            depth_map = view["depth_map"].cuda()
            if depth_map.dim() == 3:
                depth_map = depth_map[0]

            image = view["image"]
            if view["ratio"] < 0.1:
                continue

            self.predictor.set_image(image)
            points_2d_and_status = view["points_2d_and_status"]
            points_2d = torch.tensor([p[:2] for p in points_2d_and_status if p[2]], device="cuda")
            point_labels = torch.ones_like(points_2d[:,0])

            with torch.inference_mode():
                res = self.predictor.predict(points_2d, point_labels)
            mask = torch.tensor(res[0][2], device="cuda")

            cam_params = view["cam_params"].cuda()
            fov = view["fov"]
            resolution = view["resolution"]

            gs_points_2d = world_point_to_pixel_and_depth(points, cam_params, fov, resolution, resolution)
            gs_points_x = gs_points_2d[0].long()
            gs_points_y = gs_points_2d[1].long()
            gs_point_depths = gs_points_2d[2]

            gs_points_mask = (gs_points_x > 0) & (gs_points_x < resolution) & (gs_points_y > 0) & (gs_points_y < resolution)
            mask_values = mask[gs_points_y[gs_points_mask], gs_points_x[gs_points_mask]] > 0.5

            depth_values = depth_map[gs_points_y[gs_points_mask], gs_points_x[gs_points_mask]]
            depth_mask = torch.abs(depth_values - gs_point_depths[gs_points_mask]) < self.depth_threshold

            final_mask = depth_mask & mask_values
            gs_point_mask_counts[gs_points_mask] += final_mask > 0.5

        self.gs_point_mask_counts = gs_point_mask_counts
            


          
        