"""
Widget that takes picked 3D points (on one object), generates orbital views that
frame all points with ROI expansion, renders each view, and flags views where
any point's rendered depth differs from expected (occlusion or out of frame).
The viewport is never changed: all test renders use temporary args.
"""
import os
import numpy as np
import torch
from imgui_bundle import imgui
import PIL.Image

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from splatviz_utils.cam_utils import get_origin, get_forward_vector, create_cam2world_matrix, world_point_to_pixel_and_depth
from widgets.widget import Widget



def sample_depth_at(depth_map, px, py, resolution):
    """Sample depth buffer at pixel (px, py). depth_map may be (1,H,W) or (H,W)."""
    px = int(np.clip(px, 0, resolution - 1))
    py = int(np.clip(py, 0, resolution - 1))
    if depth_map.dim() == 3:
        return depth_map[0, py, px].item()
    return depth_map[py, px].item()


def _compute_roi_center_and_radius(points):
    """Centroid and radius that contains all points (max distance from centroid)."""
    pts = np.asarray(points, dtype=np.float64)
    center = np.mean(pts, axis=0)
    radii = np.linalg.norm(pts - center, axis=1)
    radius = float(np.max(radii))
    if radius <= 0:
        radius = 1e-6
    return center, radius


class OrbitalValidationWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Orbital validation")
        self.depth_threshold = 0.05
        self.roi_expansion = 1.5  # orbit so object has this margin (e.g. 1.5 = 50% margin)
        self.num_yaw = 8
        self.num_pitch = 3
        self.results = []  # list of {view_idx, valid, failed_points: [(idx, diff)], yaw, pitch}
        self.views_save_dir = "./orbital_views"

    def _get_picked_points(self):
        pts = getattr(self.viz.store, "picked_points", None)
        if pts is None:
            return []
        return list(pts)

    def _get_up_vector(self):
        up = getattr(self.viz.args, "up_vector", None)
        if up is not None:
            return torch.as_tensor(up, device="cuda") if not isinstance(up, torch.Tensor) else up.to("cuda")
        return torch.tensor([0.0, 1.0, 0.0], device="cuda")

    def _make_orbital_cam(self, lookat_center, orbit_radius, yaw, pitch, up_vector):
        """Orbital camera looking at lookat_center from (yaw, pitch) on sphere of radius orbit_radius."""
        lookat = torch.tensor(lookat_center, dtype=torch.float32, device="cuda")
        cam_origin = get_origin(yaw, pitch, orbit_radius, lookat, up_vector, device="cuda")
        forward = get_forward_vector(lookat, yaw + np.pi / 2, pitch + np.pi / 2, orbit_radius, up_vector, cam_origin)
        cam2world = create_cam2world_matrix(forward.unsqueeze(0), cam_origin.unsqueeze(0), up_vector.unsqueeze(0))
        return cam2world[0]

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            label("Depth threshold", viz.label_w)
            _, self.depth_threshold = imgui.input_float("##depth_threshold", self.depth_threshold, 0.001, 0.1)

            label("ROI expansion", viz.label_w)
            _, self.roi_expansion = imgui.input_float("##roi_expansion", self.roi_expansion, 0.1, 5.0)

            label("Yaw steps", viz.label_w)
            _, self.num_yaw = imgui.input_int("##num_yaw", self.num_yaw, 1, 32)

            label("Pitch steps", viz.label_w)
            _, self.num_pitch = imgui.input_int("##num_pitch", self.num_pitch, 1, 8)

            points = self._get_picked_points()
            imgui.text(f"Picked points: {len(points)}")

            if imgui_utils.button("Run orbital validation", width=viz.button_large_w):
                self._run_validation()

            if len(self.results) > 0:
                label("Save to folder", viz.label_w)
                _, self.views_save_dir = imgui.input_text("##views_save_dir", self.views_save_dir)
                if imgui_utils.button("Write views to disk", width=viz.button_large_w):
                    self._write_views_to_disk()

            if self.results:
                ok_count = sum(1 for r in self.results if r["ratio"] > 0.5)
                imgui.text(f"Views: {ok_count}/{len(self.results)} OK")
                for r in self.results:
                    status = "OK" if r["ratio"] > 0.5 else "FLAGGED"

                    if status=="OK":
                        color = (0, 1, 0, 1)
                    else:
                        color = (1, 0, 0, 1)
                    imgui.text_colored(color, f"  View {r['view_idx'] + 1}: {status}")

        viz.store.orbital_views = self.results

    def _run_validation(self):
        """Generate orbital views that contain all points (with ROI expansion); validate depth for each point per view."""
        points = self._get_picked_points()
        if not points:
            self.viz.result.message = "No picked points. Use Pick 3D first."
            self.results = []
            return

        center, object_radius = _compute_roi_center_and_radius(points)
        orbit_radius = object_radius * float(self.roi_expansion)
        if orbit_radius <= 0:
            orbit_radius = 1.0

        base_args = dict(self.viz.args)
        base_args["return_depth"] = True

        up_vector = self._get_up_vector()
        resolution = int(getattr(self.viz.args, "resolution", 1024))
        fov = float(getattr(self.viz.args, "fov", 60))
        threshold = float(self.depth_threshold)

        renderer = self.viz.renderer.renderer
        self.results = []

        num_yaw = max(1, int(self.num_yaw))
        num_pitch = max(1, int(self.num_pitch))

        pitch_vals = np.linspace(np.pi / 2 - 1e-3, np.pi - 1e-3, num_pitch, endpoint=False) if num_pitch > 1 else [np.pi / 2 - 1e-3]
        yaw_vals = np.linspace(0, 2 * np.pi, num_yaw, endpoint=False)

        view_idx = 0
        for pitch in pitch_vals:
            for yaw in yaw_vals:
                cam_params = self._make_orbital_cam(center, orbit_radius, yaw, pitch, up_vector)
                test_args = dict(base_args)
                test_args["cam_params"] = cam_params
                test_args["return_depth"] = True

                with torch.inference_mode():
                    result = renderer.render(**test_args)

                if "depth_map" not in result or "error" in result:
                    entry = None
                    self.results.append(entry)
                    view_idx += 1
                    continue

                depth_map = result["depth_map"]
                res = result.get("depth_resolution", resolution)
                cam_params_cpu = result.get("depth_cam_params", cam_params.cpu())
                fov_actual = result.get("depth_fov", fov)

                points_2d_and_status = []
                for idx, point in enumerate(points):
                    point = torch.tensor(point).unsqueeze(0)
                    proj = world_point_to_pixel_and_depth(point, cam_params_cpu, fov_actual, resolution, resolution)
                    if proj is None:
                        points_2d_and_status.append((None, None, False))
                        continue
                    px, py, expected_depth = proj[0].item(), proj[1].item(), proj[2].item()
                    rendered_depth = sample_depth_at(depth_map, px, py, res)
                    diff = abs(rendered_depth - expected_depth)
                    if diff > threshold:
                        points_2d_and_status.append((px, py, False))
                    else:
                        points_2d_and_status.append((px, py, True))

                entry = {
                    "view_idx": view_idx,
                    "points_2d_and_status": points_2d_and_status,
                    "cam_params": cam_params,
                    "fov": fov_actual,
                    "resolution": resolution,
                    "ratio": len([p for p in points_2d_and_status if p[2]]) / len(points_2d_and_status),
                    "image": result["image"],
                    "depth_map": result["depth_map"].detach()
                }

                self.results.append(entry)
                view_idx += 1

        self.viz.result.message = f"Validated {view_idx} view(s), all containing {len(points)} point(s). {len(self.results)} views saved for later processing."

    def _write_views_to_disk(self):
        """Write results to the configured directory for later processing."""
        if not self.results:
            self.viz.result.message = "No saved views. Run orbital validation first."
            return
        try:
            os.makedirs(self.views_save_dir, exist_ok=True)
            for v in self.results:
                img = v.get("image")
                if img is None:
                    continue
                view_idx = v["view_idx"]

                ratio = v["ratio"]
              
                name = f"view_{view_idx:04d}_ratio{ratio:.2f}.png"
                path = os.path.join(self.views_save_dir, name)
                if img.ndim == 2:
                    pil_img = PIL.Image.fromarray(img, "L")
                elif img.shape[-1] == 3:
                    pil_img = PIL.Image.fromarray(img, "RGB")
                else:
                    pil_img = PIL.Image.fromarray(img[:, :, :3], "RGB")
                pil_img.save(path)
            self.viz.result.message = f"Wrote {len([v for v in self.results if v.get('image') is not None])} images to {self.views_save_dir}"
        except Exception as e:
            self.viz.result.error = str(e)