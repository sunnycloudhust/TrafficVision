"""GUI for selecting image calibration points and deriving metric coordinates."""

import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parents[2]
FRAME_DIR = ROOT / "data/ua_detrac/raw/DETRAC/DETRAC-Images/DETRAC-Images"
OUTPUT_DIR = ROOT / "data/ua_detrac/world_points"


def metric_coordinates(point_count, measured_distances):
    """Embed a complete set of pairwise distances in 2D, anchored at point 1 and 2."""
    if point_count < 4:
        raise ValueError("Homography cần ít nhất 4 điểm.")
    expected = point_count * (point_count - 1) // 2
    if len(measured_distances) != expected:
        raise ValueError(f"Cần nhập khoảng cách cho đủ {expected} cặp điểm.")
    distances = np.zeros((point_count, point_count), dtype=float)
    for (first, second), distance in measured_distances.items():
        if not (0 <= first < point_count and 0 <= second < point_count) or first == second:
            raise ValueError("Cặp điểm khoảng cách không hợp lệ.")
        if distance <= 0 or not np.isfinite(distance):
            raise ValueError("Khoảng cách phải là số mét dương.")
        distances[first, second] = distances[second, first] = distance

    centering = np.eye(point_count) - np.ones((point_count, point_count)) / point_count
    gram = -0.5 * centering @ (distances**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:2]
    if eigenvalues[order[1]] <= 1e-9:
        raise ValueError("Các khoảng cách không xác định được mặt phẳng 2D.")
    coords = eigenvectors[:, order] * np.sqrt(np.maximum(eigenvalues[order], 0))
    coords -= coords[0]
    baseline = coords[1]
    angle = np.arctan2(baseline[1], baseline[0])
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    coords = coords @ rotation.T
    if coords[1, 0] < 0:
        coords[:, 0] *= -1
    coords[0] = [0.0, 0.0]
    return coords


def locate_from_three_anchors(anchors, distances):
    """Locate one 2D point from distances to three non-collinear anchors."""
    anchors = np.asarray(anchors, dtype=float)
    distances = np.asarray(distances, dtype=float)
    if anchors.shape != (3, 2) or distances.shape != (3,):
        raise ValueError("Cần đúng 3 điểm mốc và 3 khoảng cách.")
    if not np.isfinite(anchors).all() or not np.isfinite(distances).all() or np.any(distances <= 0):
        raise ValueError("Các khoảng cách phải là số mét dương.")
    matrix = 2 * (anchors[1:] - anchors[0])
    if np.linalg.matrix_rank(matrix) < 2:
        raise ValueError("Ba góc mốc phải không thẳng hàng.")
    rhs = (
        np.sum(anchors[1:] ** 2, axis=1)
        - np.sum(anchors[0] ** 2)
        + distances[0] ** 2
        - distances[1:] ** 2
    )
    point = np.linalg.solve(matrix, rhs)
    errors = np.abs(np.linalg.norm(point - anchors, axis=1) - distances)
    return point, float(errors.max())


class CalibrationGUI:
    def __init__(self, root):
        self.root = root
        root.title("UA-DETRAC calibration point labeler")
        root.geometry("1600x1000")
        root.minsize(1250, 760)
        if not FRAME_DIR.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục frame: {FRAME_DIR}")
        self.cameras = {
            folder.name: sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            for folder in sorted(FRAME_DIR.iterdir()) if folder.is_dir()
        }
        self.cameras = {key: value for key, value in self.cameras.items() if value}
        self.camera = tk.StringVar(value=next(iter(self.cameras), ""))
        self.camera_choice = tk.StringVar()
        self.distance = tk.StringVar()
        self.pair = tk.StringVar()
        self.geometry_mode = tk.StringVar(value="Hình chữ nhật")
        self.rectangle_length = tk.StringVar()
        self.rectangle_width = tk.StringVar()
        self.status = tk.StringVar(value="Chọn các điểm trên mặt đường, sau đó nhập khoảng cách thực tế giữa từng cặp.")
        self.points = []
        self.distances = {}
        self.photo = None
        self.scale = 1.0
        self.offset = (0, 0)
        self.world = None

        body = ttk.Frame(root, padding=10)
        body.pack(fill="both", expand=True)
        toolbar = ttk.Frame(body)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Chọn camera / sequence:").pack(side="left")
        self.camera_dropdown = ttk.Combobox(toolbar, textvariable=self.camera_choice, state="readonly", width=34)
        self.camera_dropdown.pack(side="left", padx=7)
        self.camera_dropdown.bind("<<ComboboxSelected>>", self.select_camera)
        ttk.Label(
            toolbar,
            text="Có thể chọn điểm trong ảnh hoặc vùng đệm xung quanh ảnh (tọa độ ngoài khung sẽ âm / vượt kích thước ảnh).",
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
        ttk.Label(self.distances_frame, text="Khoảng cách giữa 2 điểm (m)",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.pair_box = ttk.Combobox(self.distances_frame, textvariable=self.pair, state="readonly", width=28)
        self.pair_box.pack(fill="x", pady=5)
        ttk.Entry(self.distances_frame, textvariable=self.distance).pack(fill="x")
        ttk.Button(self.distances_frame, text="Lưu khoảng cách", command=self.add_distance).pack(fill="x", pady=5)
        self.distance_list = tk.Listbox(self.distances_frame, height=8, width=35)
        self.distance_list.pack(fill="x", pady=5)
        self.rectangle_frame.pack(fill="x", pady=4)

        ttk.Button(side, text="Tính world points", command=self.solve).pack(fill="x", pady=(10, 4))
        ttk.Button(side, text="Lưu image/world points", command=self.save).pack(fill="x")
        self.world_list = tk.Listbox(side, height=10, width=35)
        self.world_list.pack(fill="x", pady=5)

        ttk.Label(body, textvariable=self.status, wraplength=1250).pack(fill="x", pady=(4, 0))
        if self.cameras:
            self.camera_choice.set(self.camera_dropdown.cget("values")[0])
            self.load_camera()

    def has_saved_points(self, camera):
        image_path = OUTPUT_DIR / f"image_points_{camera}.json"
        world_path = OUTPUT_DIR / f"world_points_{camera}.json"
        try:
            image_points = json.loads(image_path.read_text(encoding="utf-8"))
            world_points = json.loads(world_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            isinstance(image_points, list)
            and isinstance(world_points, list)
            and len(image_points) >= 4
            and len(image_points) == len(world_points)
        )

    def refresh_camera_list(self):
        choices = [
            f"{camera}{' --done' if self.has_saved_points(camera) else ''}"
            for camera in self.cameras
        ]
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
        self.image = cv2.cvtColor(cv2.imread(str(self.frame_path)), cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
        self.refresh_lists()
        self.render()
        self.status.set(f"Frame: {self.frame_path.name} ({self.width}×{self.height}). Click các điểm trên mặt đường.")

    def render(self):
        if not hasattr(self, "image"):
            return
        cw, ch = max(self.canvas.winfo_width(), 10), max(self.canvas.winfo_height(), 10)
        # Keep a 25% margin around the frame so projected road points can be
        # selected beyond the image boundary and retain negative/out-of-frame px.
        self.scale = min(cw / (self.width * 1.5), ch / (self.height * 1.5))
        shown = cv2.resize(self.image, (round(self.width * self.scale), round(self.height * self.scale)))
        pil = Image.fromarray(shown)
        self.photo = ImageTk.PhotoImage(pil)
        self.offset = ((cw - pil.width) // 2, (ch - pil.height) // 2)
        self.canvas.delete("all")
        self.canvas.create_image(*self.offset, image=self.photo, anchor="nw")
        self.canvas.create_rectangle(
            self.offset[0], self.offset[1], self.offset[0] + pil.width,
            self.offset[1] + pil.height, outline="#ffffff", width=2,
        )
        for i, (x, y) in enumerate(self.points, 1):
            px, py = self.offset[0] + x * self.scale, self.offset[1] + y * self.scale
            self.canvas.create_oval(px-5, py-5, px+5, py+5, fill="#00e676", outline="white")
            self.canvas.create_text(px+12, py-12, text=str(i), fill="yellow", font=("TkDefaultFont", 12, "bold"))

    def on_click(self, event):
        x = round((event.x - self.offset[0]) / self.scale)
        y = round((event.y - self.offset[1]) / self.scale)
        margin = 0.25
        if -margin * self.width <= x <= (1 + margin) * self.width and -margin * self.height <= y <= (1 + margin) * self.height:
            self.points.append([x, y])
            self.world = None
            self.refresh_lists()
            self.render()
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                self.status.set(f"P{len(self.points)} nằm ngoài ảnh: ({x}, {y}) px. Kiểm tra tọa độ ngoại suy thực tế.")

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
            if labels and self.pair.get() not in labels:
                self.pair.set(labels[0])

    def add_distance(self):
        try:
            first, second = [int(value[1:]) - 1 for value in self.pair.get().split(" ↔ ")]
            value = float(self.distance.get())
            if not np.isfinite(value) or value <= 0:
                raise ValueError
        except (ValueError, IndexError):
            messagebox.showerror("Khoảng cách", "Chọn cặp điểm và nhập khoảng cách dương bằng mét.")
            return
        self.distances[(first, second)] = value
        self.distance.set("")
        self.world = None
        self.refresh_lists()

    def solve(self):
        try:
            image_points = np.asarray(self.points, dtype=float)
            if (
                len(np.unique(image_points, axis=0)) != len(image_points)
                or np.linalg.matrix_rank(image_points - image_points.mean(axis=0)) < 2
            ):
                raise ValueError("Image points bị trùng hoặc thẳng hàng; hãy chọn điểm phủ một vùng 2D.")
            if self.geometry_mode.get() == "Hình chữ nhật":
                if len(self.points) < 4:
                    raise ValueError("Hình chữ nhật cần 4 góc đầu tiên, click theo thứ tự quanh hình.")
                length = float(self.rectangle_length.get())
                width = float(self.rectangle_width.get())
                if not np.isfinite(length) or not np.isfinite(width) or length <= 0 or width <= 0:
                    raise ValueError("Nhập chiều dài và chiều rộng dương bằng mét.")
                corners = np.asarray(
                    [[0.0, 0.0], [length, 0.0], [length, width], [0.0, width]],
                    dtype=float,
                )
                self.world = corners.tolist()
                anchor_points = corners[:3]
                max_error = 0.0
                for point_index in range(4, len(self.points)):
                    required = [(anchor_index, point_index) for anchor_index in range(3)]
                    missing = [f"P{a+1}–P{point_index+1}" for a, b in required if (a, b) not in self.distances]
                    if missing:
                        raise ValueError(
                            f"Nhập khoảng cách từ P{point_index+1} tới 3 góc P1, P2, P3: "
                            + ", ".join(missing)
                        )
                    measured = [self.distances[pair] for pair in required]
                    world_point, error = locate_from_three_anchors(anchor_points, measured)
                    self.world.append(world_point.tolist())
                    max_error = max(max_error, error)
                self.world = np.asarray(self.world, dtype=float)
            else:
                self.world = metric_coordinates(len(self.points), self.distances)
            if np.linalg.matrix_rank(self.world - self.world.mean(axis=0)) < 2:
                raise ValueError("Các điểm thực tế gần như thẳng hàng; cần phủ một vùng 2D.")
        except ValueError as exc:
            self.world = None
            messagebox.showerror("Chưa đủ dữ liệu", str(exc))
            return
        self.refresh_lists()
        if self.geometry_mode.get() == "Hình chữ nhật":
            if len(self.points) > 4:
                self.status.set(
                    f"World points đã dựng theo hình chữ nhật; mỗi điểm thêm dùng 3 khoảng cách tới "
                    f"P1–P3. Sai lệch ba cạnh mốc lớn nhất: {max_error:.3f} m."
                )
            else:
                self.status.set(
                    "Đã gán 4 world points theo hình chữ nhật: P1=(0,0), P2=(dài,0), "
                    "P3=(dài,rộng), P4=(0,rộng)."
                )
        else:
            residuals = [
                abs(float(np.linalg.norm(self.world[first] - self.world[second])) - measured)
                for (first, second), measured in self.distances.items()
            ]
            max_error = max(residuals, default=0.0)
            self.status.set(
                f"World points đã dựng (P1=(0,0), P2 trên +X). Sai lệch lớn nhất giữa "
                f"khoảng cách nhập và dựng lại: {max_error:.3f} m."
            )

    def save(self):
        if self.world is None:
            messagebox.showerror("Chưa tính", "Nhấn 'Tính world points' trước khi lưu.")
            return
        camera = self.camera.get()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        image_path = OUTPUT_DIR / f"image_points_{camera}.json"
        world_path = OUTPUT_DIR / f"world_points_{camera}.json"
        if world_path.exists() and not messagebox.askyesno(
            "Ghi đè world points?", f"{world_path.name} đã tồn tại. Bạn có muốn thay thế không?"
        ):
            return
        image_path.write_text(json.dumps(self.points, indent=2) + "\n", encoding="utf-8")
        world_path.write_text(json.dumps(self.world.tolist(), indent=2) + "\n", encoding="utf-8")
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
    root = tk.Tk()
    gui = CalibrationGUI(root)
    # Keep the instance alive for Tk callbacks.
    root.calibration_gui = gui
    root.mainloop()


if __name__ == "__main__":
    main()
