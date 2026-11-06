"""GUI for selecting image calibration points and deriving metric coordinates."""

import argparse
import json
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parents[2]
FRAME_DIR = ROOT / "data/ua_detrac/raw/DETRAC/DETRAC-Images/DETRAC-Images"
OUTPUT_DIR = ROOT / "data/ua_detrac/world_points"


sys.path.insert(0, str(ROOT / "src"))
from trafficvision.geometry.calibration_points import (  # noqa: E402
    check_distance_residuals,
    metric_coordinates,
    rectangle_coordinates,
    validate_correspondences,
)


class CalibrationGUI:
    def __init__(self, root, frame_dir=FRAME_DIR, output_dir=OUTPUT_DIR):
        self.root = root
        self.frame_dir, self.output_dir = Path(frame_dir), Path(output_dir)
        root.title("UA-DETRAC calibration point labeler")
        root.geometry("1600x1000")
        root.minsize(1250, 760)
        if not self.frame_dir.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục frame: {self.frame_dir}")
        self.cameras = {
            folder.name: sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            for folder in sorted(self.frame_dir.iterdir())
            if folder.is_dir()
        }
        self.cameras = {key: value for key, value in self.cameras.items() if value}
        self.camera = tk.StringVar(value=next(iter(self.cameras), ""))
        self.camera_choice = tk.StringVar()
        self.distance = tk.StringVar()
        self.pair = tk.StringVar()
        self.geometry_mode = tk.StringVar(value="Hình chữ nhật")
        self.rectangle_length = tk.StringVar()
        self.rectangle_width = tk.StringVar()
        self.max_error = tk.StringVar(value="0.25")
        self.allow_extrapolation = tk.BooleanVar(value=False)
        self.status = tk.StringVar(
            value="Chọn các điểm trên mặt đường, sau đó nhập khoảng cách thực tế giữa từng cặp."
        )
        self.points = []
        self.distances = {}
        self.photo = None
        self.scale = 1.0
        self.offset = (0, 0)
        self.world = None
        self._solution_signature = None

        body = ttk.Frame(root, padding=10)
        body.pack(fill="both", expand=True)
        toolbar = ttk.Frame(body)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Chọn camera / sequence:").pack(side="left")
        self.camera_dropdown = ttk.Combobox(
            toolbar, textvariable=self.camera_choice, state="readonly", width=34
        )
        self.camera_dropdown.pack(side="left", padx=7)
        self.camera_dropdown.bind("<<ComboboxSelected>>", self.select_camera)
        ttk.Label(
            toolbar,
            text="Click các điểm đường có khoảng cách đo thực. Vùng đệm chỉ dùng khi có tọa độ ngoại suy đã xác định.",
        ).pack(side="left", padx=12)

        content = ttk.Frame(body)
        content.pack(fill="both", expand=True, pady=8)
        self.refresh_camera_list()

        self.canvas = tk.Canvas(content, bg="#343a40", width=1200, height=800, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Configure>", lambda _event: self.render())

        side = ttk.Frame(content, width=280)
        side.pack(side="right", fill="y", padx=(12, 0))
        ttk.Label(side, text="Điểm ảnh đã chọn", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.point_list = tk.Listbox(side, height=12, width=35)
        self.point_list.pack(fill="x", pady=5)
        buttons = ttk.Frame(side)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Undo điểm cuối", command=self.undo).pack(side="left")
        ttk.Button(buttons, text="Xóa tất cả", command=self.clear).pack(side="left", padx=4)

        ttk.Separator(side).pack(fill="x", pady=10)
        ttk.Label(side, text="Giả định hình học", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.geometry_box = ttk.Combobox(
            side,
            textvariable=self.geometry_mode,
            values=("Hình chữ nhật", "Hình khác (khoảng cách các cặp)"),
            state="readonly",
        )
        self.geometry_box.pack(fill="x", pady=5)
        self.geometry_box.bind("<<ComboboxSelected>>", self.change_geometry_mode)

        self.rectangle_frame = ttk.Frame(side)
        ttk.Label(
            self.rectangle_frame,
            text="Click 4 góc đầu theo thứ tự quanh hình. Điểm bổ sung cần khoảng cách tới P1, P2, P3.",
            wraplength=260,
        ).pack(anchor="w")
        ttk.Label(self.rectangle_frame, text="Chiều dài P1–P2 (m)").pack(anchor="w", pady=(5, 0))
        ttk.Entry(self.rectangle_frame, textvariable=self.rectangle_length).pack(fill="x")
        ttk.Label(self.rectangle_frame, text="Chiều rộng P2–P3 (m)").pack(anchor="w", pady=(5, 0))
        ttk.Entry(self.rectangle_frame, textvariable=self.rectangle_width).pack(fill="x")

        self.distances_frame = ttk.Frame(side)
        ttk.Label(
            self.distances_frame, text="Khoảng cách giữa 2 điểm (m)", font=("TkDefaultFont", 11, "bold")
        ).pack(anchor="w")
        self.pair_box = ttk.Combobox(self.distances_frame, textvariable=self.pair, state="readonly", width=28)
        self.pair_box.pack(fill="x", pady=5)
        ttk.Entry(self.distances_frame, textvariable=self.distance).pack(fill="x")
        ttk.Button(self.distances_frame, text="Lưu khoảng cách", command=self.add_distance).pack(
            fill="x", pady=5
        )
        self.distance_list = tk.Listbox(self.distances_frame, height=8, width=35)
        self.distance_list.pack(fill="x", pady=5)
        self.rectangle_frame.pack(fill="x", pady=4)

        ttk.Label(side, text="Sai số khoảng cách / fit tối đa (m)").pack(anchor="w")
        ttk.Entry(side, textvariable=self.max_error).pack(fill="x")
        ttk.Checkbutton(
            side,
            text="Cho phép điểm ngoại suy ngoài ảnh",
            variable=self.allow_extrapolation,
            command=self.invalidate_solution,
        ).pack(anchor="w")
        ttk.Button(side, text="Tính world points", command=self.solve).pack(fill="x", pady=(10, 4))
        ttk.Button(side, text="Lưu image/world points", command=self.save).pack(fill="x")
        self.world_list = tk.Listbox(side, height=10, width=35)
        self.world_list.pack(fill="x", pady=5)

        for variable in (self.rectangle_length, self.rectangle_width, self.max_error, self.distance):
            variable.trace_add("write", self.invalidate_solution)
        ttk.Label(body, textvariable=self.status, wraplength=1250).pack(fill="x", pady=(4, 0))
        if self.cameras:
            self.camera_choice.set(self.camera_dropdown.cget("values")[0])
            self.load_camera()

    def has_saved_points(self, camera):
        image_path = self.output_dir / f"image_points_{camera}.json"
        world_path = self.output_dir / f"world_points_{camera}.json"
        try:
            image_points = json.loads(image_path.read_text(encoding="utf-8"))
            world_points = json.loads(world_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        try:
            metadata_path = self.output_dir / f"calibration_points_{camera}.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            if not isinstance(metadata, dict):
                return False
            validate_correspondences(image_points, world_points, metadata.get("max_error_m", 0.5))
        except (ValueError, TypeError, OSError, np.linalg.LinAlgError):
            return False
        return True

    def refresh_camera_list(self):
        choices = [f"{camera}{' --done' if self.has_saved_points(camera) else ''}" for camera in self.cameras]
        self.camera_dropdown.configure(values=choices)
        current = self.camera.get()
        self.camera_choice.set(next((item for item in choices if item.split()[0] == current), ""))

    def select_camera(self, _event=None):
        choice = self.camera_choice.get().strip()
        if not choice:
            return
        selected = choice.split()[0]
        if selected == self.camera.get():
            return
        self.camera.set(selected)
        self.load_camera()

    def change_geometry_mode(self, _event=None):
        self.update_geometry_controls()
        self.world = None
        self.refresh_lists()

    def update_geometry_controls(self):
        if self.geometry_mode.get() == "Hình chữ nhật":
            self.distances_frame.pack_forget()
            self.rectangle_frame.pack(fill="x", pady=4)
            if len(self.points) > 4:
                self.distances_frame.pack(fill="x", pady=4)
        else:
            self.rectangle_frame.pack_forget()
            self.distances_frame.pack(fill="x", pady=4)

    def load_camera(self):
        self.points.clear()
        self.distances.clear()
        self.world = None
        self.rectangle_length.set("")
        self.rectangle_width.set("")
        files = self.cameras.get(self.camera.get())
        if not files:
            messagebox.showerror("Camera", "Camera này không có ảnh.")
            return
        self.frame_path = files[0]
        frame = cv2.imread(str(self.frame_path))
        if frame is None:
            self.image = None
            self.canvas.delete("all")
            self.refresh_lists()
            messagebox.showerror("Frame", f"Không đọc được ảnh: {self.frame_path}")
            return
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
        self.refresh_lists()
        self.render()
        self.status.set(
            f"Frame: {self.frame_path.name} ({self.width}×{self.height}). Click các điểm trên mặt đường."
        )

    def render(self):
        if getattr(self, "image", None) is None:
            return
        cw, ch = max(self.canvas.winfo_width(), 10), max(self.canvas.winfo_height(), 10)
        # Keep a 25% margin around the frame so projected road points can be
        # selected beyond the image boundary and retain negative/out-of-frame px.
        self.scale = min(cw / (self.width * 1.5), ch / (self.height * 1.5))
        shown = cv2.resize(
            self.image, (max(1, round(self.width * self.scale)), max(1, round(self.height * self.scale)))
        )
        pil = Image.fromarray(shown)
        self.photo = ImageTk.PhotoImage(pil)
        self.offset = ((cw - pil.width) // 2, (ch - pil.height) // 2)
        self.canvas.delete("all")
        self.canvas.create_image(*self.offset, image=self.photo, anchor="nw")
        self.canvas.create_rectangle(
            self.offset[0],
            self.offset[1],
            self.offset[0] + pil.width,
            self.offset[1] + pil.height,
            outline="#ffffff",
            width=2,
        )
        for i, (x, y) in enumerate(self.points, 1):
            px, py = self.offset[0] + x * self.scale, self.offset[1] + y * self.scale
            self.canvas.create_oval(px - 5, py - 5, px + 5, py + 5, fill="#00e676", outline="white")
            self.canvas.create_text(
                px + 12, py - 12, text=str(i), fill="yellow", font=("TkDefaultFont", 12, "bold")
            )

    def on_click(self, event):
        if getattr(self, "image", None) is None:
            return
        x = round((event.x - self.offset[0]) / self.scale)
        y = round((event.y - self.offset[1]) / self.scale)
        if not self.allow_extrapolation.get() and not (0 <= x < self.width and 0 <= y < self.height):
            self.status.set("Điểm ngoài ảnh bị bỏ qua. Chỉ bật ngoại suy khi đã biết tọa độ hình học.")
            return
        margin = 0.25
        if (
            -margin * self.width <= x <= (1 + margin) * self.width
            and -margin * self.height <= y <= (1 + margin) * self.height
        ):
            self.points.append([x, y])
            self.world = None
            self.refresh_lists()
            self.render()
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                self.status.set(
                    f"P{len(self.points)} nằm ngoài ảnh: ({x}, {y}) px. Kiểm tra tọa độ ngoại suy thực tế."
                )

    def refresh_lists(self):
        if hasattr(self, "rectangle_frame"):
            self.update_geometry_controls()
        self.point_list.delete(0, "end")
        for i, (x, y) in enumerate(self.points, 1):
            self.point_list.insert("end", f"P{i}: ({x}, {y}) px")
        pairs = [(i, j) for i in range(len(self.points)) for j in range(i + 1, len(self.points))]
        labels = [f"P{i+1} ↔ P{j+1}" for i, j in pairs]
        self.pair_box_values = labels
        # Set values through the widget's associated combobox.
        self._set_pair_values(labels)
        self.distance_list.delete(0, "end")
        for (i, j), value in sorted(self.distances.items()):
            self.distance_list.insert("end", f"P{i+1}–P{j+1}: {value:g} m")
        if hasattr(self, "world_list"):
            self.world_list.delete(0, "end")
            if self.world is not None:
                for i, (x, y) in enumerate(self.world, 1):
                    self.world_list.insert("end", f"P{i}: ({x:.3f}, {y:.3f}) m")

    def _set_pair_values(self, labels):
        # Store widget explicitly on first creation in __init__.
        if hasattr(self, "pair_box"):
            self.pair_box.configure(values=labels)
            if not labels:
                self.pair.set("")
            elif self.pair.get() not in labels:
                self.pair.set(labels[0])

    def add_distance(self):
        try:
            first, second = [int(value[1:]) - 1 for value in self.pair.get().split(" ↔ ")]
            value = float(self.distance.get())
            if not (0 <= first < second < len(self.points)) or not np.isfinite(value) or value <= 0:
                raise ValueError
        except (ValueError, IndexError):
            messagebox.showerror("Khoảng cách", "Chọn cặp điểm và nhập khoảng cách dương bằng mét.")
            return
        self.distances[(first, second)] = value
        self.distance.set("")
        self.world = None
        self.refresh_lists()

    def invalidate_solution(self, *_args):
        self.world = None
        self._solution_signature = None
        if hasattr(self, "world_list"):
            self.world_list.delete(0, "end")
        self.status.set("Dữ liệu đã đổi; cần tính lại world points trước khi lưu.")

    def input_signature(self):
        return (
            self.camera.get(),
            tuple(map(tuple, self.points)),
            tuple(sorted(self.distances.items())),
            self.geometry_mode.get(),
            self.rectangle_length.get(),
            self.rectangle_width.get(),
            self.max_error.get(),
            self.allow_extrapolation.get(),
        )

    def solve(self):
        self.invalidate_solution()
        try:
            if self.distance.get().strip():
                raise ValueError("Nhấn 'Lưu khoảng cách' hoặc xóa giá trị đang nhập trước khi tính.")
            if not self.allow_extrapolation.get() and any(
                not (0 <= x < self.width and 0 <= y < self.height) for x, y in self.points
            ):
                raise ValueError("Còn điểm ngoài ảnh; cần bật ngoại suy có chủ đích hoặc chọn lại điểm.")
            tolerance = float(self.max_error.get())
            if self.geometry_mode.get() == "Hình chữ nhật":
                world = rectangle_coordinates(
                    self.points,
                    float(self.rectangle_length.get()),
                    float(self.rectangle_width.get()),
                    self.distances,
                    tolerance,
                )
            else:
                world = metric_coordinates(len(self.points), self.distances, tolerance)
            homography = validate_correspondences(self.points, world, tolerance)
            residual = check_distance_residuals(world, self.distances, tolerance)
        except (ValueError, np.linalg.LinAlgError) as exc:
            messagebox.showerror("Calibration chưa hợp lệ", str(exc))
            return
        self.world = world
        self._solution_signature = self.input_signature()
        self.refresh_lists()
        self.status.set(
            f"Đã tính {len(world)} điểm. Sai lệch khoảng cách tối đa: {residual:.3f} m; "
            f"fit homography trung bình: {homography.reprojection_error_m:.3f} m. "
            "Cần kiểm tra trên điểm đo độc lập trước khi báo vận tốc thực."
        )

    def save(self):
        if self.world is None or self._solution_signature != self.input_signature():
            messagebox.showerror("Chưa tính", "Nhấn 'Tính world points' trước khi lưu.")
            return
        try:
            validate_correspondences(self.points, self.world, float(self.max_error.get()))
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.invalidate_solution()
            messagebox.showerror("Calibration chưa hợp lệ", str(exc))
            return
        camera = self.camera.get()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.output_dir / f"image_points_{camera}.json"
        world_path = self.output_dir / f"world_points_{camera}.json"
        if (world_path.exists() or image_path.exists()) and not messagebox.askyesno(
            "Ghi đè tọa độ?", f"Tệp tọa độ của {camera} đã tồn tại. Bạn có muốn thay thế không?"
        ):
            return
        image_path.write_text(json.dumps(self.points, indent=2) + "\n", encoding="utf-8")
        world_path.write_text(json.dumps(self.world.tolist(), indent=2) + "\n", encoding="utf-8")
        metadata = dict(
            camera_id=camera,
            frame_size=[self.width, self.height],
            frame=str(self.frame_path),
            geometry_mode=self.geometry_mode.get(),
            world_unit="meter",
            max_error_m=float(self.max_error.get()),
            measured_distances=[
                dict(first=i + 1, second=j + 1, meters=d) for (i, j), d in sorted(self.distances.items())
            ],
            rectangle_length=self.rectangle_length.get(),
            rectangle_width=self.rectangle_width.get(),
            extrapolated_points=[
                i + 1
                for i, (x, y) in enumerate(self.points)
                if not (0 <= x < self.width and 0 <= y < self.height)
            ],
        )
        (self.output_dir / f"calibration_points_{camera}.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        self.status.set(f"Đã lưu {len(self.points)} cặp điểm: {image_path.name}, {world_path.name}")
        self.refresh_camera_list()

    def undo(self):
        if self.points:
            self.points.pop()
            self.distances = {(i, j): d for (i, j), d in self.distances.items() if j < len(self.points)}
            self.world = None
            self.refresh_lists()
            self.render()

    def clear(self):
        self.points.clear()
        self.distances.clear()
        self.world = None
        self.refresh_lists()
        self.render()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, default=FRAME_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    root = tk.Tk()
    gui = CalibrationGUI(root, args.frames_dir, args.output_dir)
    # Keep the instance alive for Tk callbacks.
    root.calibration_gui = gui
    root.mainloop()


if __name__ == "__main__":
    main()
