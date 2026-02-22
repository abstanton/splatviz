import os
import numpy as np
import torch
from imgui_bundle import imgui

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from widgets.widget import Widget


def unproject_pixel_to_world(px, py, depth, cam_params, fov_deg, resolution):
    """Unproject a pixel and depth to 3D world position.
    cam_params: 4x4 camera-to-world matrix. depth: view-space z.
    """
    fov_rad = fov_deg / 360 * 2 * np.pi
    tan_fov = np.tan(fov_rad / 2)
    # Normalized device coords: x in [-1,1], y in [-1,1]
    ndc_x = 2 * px / resolution - 1
    ndc_y = 1 - 2 * py / resolution
    # View space (symmetric fov, principal point at center)
    z_v = float(depth)
    x_v = ndc_x * z_v * tan_fov
    y_v = ndc_y * z_v * tan_fov
    view_pos = torch.tensor([x_v, y_v, z_v, 1.0], dtype=cam_params.dtype)
    world = (cam_params @ view_pos)[:3]
    return world.numpy()


class Pick3DWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Pick 3D")
        self.enabled = False
        self.points = []  # list of (x, y, z) numpy arrays
        self.save_path = "./picked_points.txt"

    def _image_display_rect(self):
        """Compute the on-screen rect of the rendered image (same as splatviz draw)."""
        viz = self.viz
        if "image" not in viz.result:
            return None
        img = viz.result.image
        # img is (H, W, C) numpy
        tex_h, tex_w = img.shape[0], img.shape[1]
        max_w = viz.content_width - viz.pane_w
        max_h = viz.content_height
        zoom = min(max_w / max(tex_w, 1), max_h / max(tex_h, 1))
        width = zoom * tex_w
        height = zoom * tex_h
        left = viz.pane_w + max_w / 2 - width / 2
        top = max_h / 2 - height / 2
        return left, top, width, height, tex_w, tex_h

    def _pixel_from_mouse(self, mx, my):
        """Return (px, py) in image pixel coords, or None if outside image."""
        rect = self._image_display_rect()
        if rect is None:
            return None
        left, top, width, height, tex_w, tex_h = rect
        if mx < left or mx >= left + width or my < top or my >= top + height:
            return None
        zoom = width / tex_w
        px = (mx - left) / zoom
        py = (my - top) / zoom
        return px, py

    def _sample_depth(self, px, py):
        """Sample depth at pixel (px, py). Returns scalar or None if invalid."""
        if "depth_map" not in self.viz.result:
            return None
        depth_map = self.viz.result.depth_map
        res = self.viz.result.get("depth_resolution")
        if res is None and depth_map.dim() >= 2:
            res = depth_map.shape[-1]
        if res is None:
            return None
        # Clamp to valid range
        px = int(np.clip(px, 0, res - 1))
        py = int(np.clip(py, 0, res - 1))
        if depth_map.dim() == 3:
            d = depth_map[0, py, px].item()
        else:
            d = depth_map[py, px].item()
        # Ignore background (often 0 or very large)
        if d <= 0 or not np.isfinite(d):
            return None
        return d

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            _changed, self.enabled = imgui.checkbox("Enable pick (click in image)", self.enabled)
            imgui.same_line()
            if imgui_utils.button("Clear points"):
                self.points.clear()

            label("Save path", viz.label_w)
            _, self.save_path = imgui.input_text("##pick_save_path", self.save_path)
            if imgui_utils.button("Save to file", width=viz.button_w):
                self._save_points()

            imgui.text(f"Points: {len(self.points)}")
            if self.points:
                show_list = self.points[-10:]
                start = len(self.points) - len(show_list)
                with imgui_utils.item_width(viz.pane_w - 80):
                    for i, pt in enumerate(show_list):
                        imgui.text(f"  #{start + i + 1}: ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})")

        viz.args.return_depth = self.enabled

        if not self.enabled:
            return

        # Click handling: only when we have depth and image
        if "depth_map" not in viz.result or "image" not in viz.result:
            return
        if "depth_cam_params" not in viz.result or "depth_fov" not in viz.result:
            return

        if not imgui.is_mouse_clicked(0):  # left click
            return

        io = imgui.get_io()
        mx, my = io.mouse_pos.x, io.mouse_pos.y
        pixel = self._pixel_from_mouse(mx, my)
        if pixel is None:
            return

        px, py = pixel
        depth = self._sample_depth(px, py)
        if depth is None:
            return

        cam_params = viz.result.depth_cam_params
        fov = viz.result.depth_fov
        res = viz.result.get("depth_resolution")
        if res is None:
            res = viz.result.depth_map.shape[-1]
        world = unproject_pixel_to_world(px, py, depth, cam_params, fov, res)
        self.points.append(world)

    def _save_points(self):
        if not self.points:
            return
        try:
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            with open(self.save_path, "w") as f:
                for pt in self.points:
                    f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
            self.viz.result.message = f"Saved {len(self.points)} points to {self.save_path}"
        except Exception as e:
            self.viz.result.error = str(e)
