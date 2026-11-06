"""Run actual Tk callbacks when a display is available; geometry tests are headless."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import yaml
from trafficvision.geometry.calibration import load_calibration

tk = pytest.importorskip("tkinter")
spec = importlib.util.spec_from_file_location(
    "calibration_tool", Path(__file__).parents[1] / "data/data-labelling/calibration_tool.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def gui(tmp_path, monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk needs a display (use xvfb-run on headless Linux)")
    root.withdraw()
    frames = tmp_path / "frames" / "MVI_test"
    frames.mkdir(parents=True)
    Image.new("RGB", (400, 300)).save(frames / "img00001.jpg")
    errors = []
    monkeypatch.setattr(module.messagebox, "showerror", lambda title, message: errors.append(message))
    monkeypatch.setattr(module.messagebox, "askyesno", lambda *args: True)
    app = module.CalibrationGUI(root, frames.parent, tmp_path / "points")
    root.update_idletasks()
    app.errors = errors
    try:
        yield app
    finally:
        root.destroy()


def solve_rectangle(gui):
    gui.points = [[10, 10], [110, 10], [110, 210], [10, 210]]
    gui.rectangle_length.set("10")
    gui.rectangle_width.set("20")
    gui.solve()
    assert gui.world is not None and not gui.errors


def test_solve_save_and_done_checks_actual_geometry(gui):
    solve_rectangle(gui)
    gui.save()
    assert gui.has_saved_points("MVI_test")
    world = json.loads((gui.output_dir / "world_points_MVI_test.json").read_text())
    np.testing.assert_allclose(world, [[0, 0], [10, 0], [10, 20], [0, 20]])
    metadata = json.loads((gui.output_dir / "calibration_points_MVI_test.json").read_text())
    assert metadata["max_error_m"] == 0.25 and metadata["world_unit"] == "meter"
    (gui.output_dir / "world_points_MVI_test.json").write_text(json.dumps([["bad"]] * 4))
    assert not gui.has_saved_points("MVI_test")


@pytest.mark.parametrize("name", ["rectangle_length", "rectangle_width", "max_error"])
def test_editing_dimensions_invalidates_solution_before_save(gui, name):
    solve_rectangle(gui)
    getattr(gui, name).set("12")
    assert gui.world is None
    gui.save()
    assert not gui.output_dir.exists() and gui.errors


def test_inconsistent_distances_and_empty_solve_do_not_save(gui):
    gui.solve()
    assert gui.world is None and gui.errors
    gui.errors.clear()
    solve_rectangle(gui)
    gui.distances[(0, 1)] = 1000
    gui.solve()
    gui.save()
    assert gui.world is None and gui.errors and not gui.output_dir.exists()


def test_extrapolation_requires_explicit_opt_in(gui):
    event = SimpleNamespace(x=gui.offset[0] - gui.scale * 20, y=gui.offset[1] + gui.scale * 20)
    gui.on_click(event)
    assert not gui.points
    gui.allow_extrapolation.set(True)
    gui.on_click(event)
    assert gui.points[0][0] < 0


def test_gui_export_is_consumed_by_calibration_cli(gui, tmp_path):
    solve_rectangle(gui)
    gui.save()
    output = tmp_path / "camera.yaml"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_camera.py",
            "--image",
            str(gui.frame_path),
            "--image-points",
            str(gui.output_dir / "image_points_MVI_test.json"),
            "--world-points",
            str(gui.output_dir / "world_points_MVI_test.json"),
            "--camera-id",
            "MVI_test",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    config = yaml.safe_load(output.read_text())
    assert config["frame_size"] == [400, 300] and config["world_unit"] == "meter"
    h = load_calibration(config)
    np.testing.assert_allclose(h.project([60.0, 110.0]), [5.0, 10.0], atol=1e-6)


def test_corrupt_metadata_is_not_done(gui):
    solve_rectangle(gui)
    gui.save()
    (gui.output_dir / "calibration_points_MVI_test.json").write_text("[]")
    assert not gui.has_saved_points("MVI_test")


def test_pending_distance_is_not_silently_ignored(gui):
    solve_rectangle(gui)
    gui.distance.set("1000")
    assert gui.world is None
    gui.solve()
    assert gui.world is None and "Lưu khoảng cách" in gui.errors[-1]


def test_clearing_points_drops_stale_pair_selection(gui):
    solve_rectangle(gui)
    gui.clear()
    assert gui.pair.get() == ""
    gui.distance.set("10")
    gui.add_distance()
    assert not gui.distances


def test_existing_outside_points_require_extrapolation_permission(gui):
    solve_rectangle(gui)
    gui.points[0] = [-10, 10]
    gui.solve()
    assert gui.world is None and "ngoài ảnh" in gui.errors[-1]
