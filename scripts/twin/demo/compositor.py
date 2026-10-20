"""Composite the scene view + decision/reason tree panel and encode the demo mp4.

Offline (CPU, no GPU lock): reads the scene frames captured in Isaac + the per-frame
decision log, renders each panel, composites them side by side (scene left, panel right)
and encodes a 1080p H.264 clip by REUSING the mission's ``twin.restock.recording.Recorder``
encoder (the exact VFR libx264/yuv420p/crf20/faststart pipeline + its decode verification).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from twin.restock.recording import Recorder
from . import panel as panel_mod


def composite_frame(scene_img, panel_img, *, out_size, panel_w, bg=panel_mod.BG):
    """Scene scaled into the left region (letterboxed to aspect), panel pasted right."""
    W, H = out_size
    region_w = W - panel_w
    canvas = Image.new('RGB', (W, H), bg)
    sw, sh = scene_img.size
    scale = min(region_w / sw, H / sh)
    nw, nh = max(2, int(sw * scale)), max(2, int(sh * scale))
    scene = scene_img.resize((nw, nh), Image.LANCZOS)
    ox = (region_w - nw) // 2
    oy = (H - nh) // 2
    canvas.paste(scene, (ox, oy))
    if panel_img.size != (panel_w, H):
        panel_img = panel_img.resize((panel_w, H), Image.LANCZOS)
    canvas.paste(panel_img, (region_w, 0))
    return canvas


def _even(n):
    return n - (n % 2)


def build_video(scene_frames_dir, decision_log_path, out_dir, *, ffmpeg,
                out_size=(1920, 1080), panel_w=480, fps=10):
    """Composite every logged frame and encode ``out_dir/walkthrough.mp4``.

    ``decision_log_path`` is the JSON written by the GPU capture: {meta, frames:[entry...]}
    where each entry carries ``scene_file`` (relative to ``scene_frames_dir``) and the
    panel ``entry`` payload (tree/header/chips). Returns the encode timing dict.
    """
    out_size = (_even(out_size[0]), _even(out_size[1]))
    scene_frames_dir = Path(scene_frames_dir)
    log = json.loads(Path(decision_log_path).read_text())
    entries = log['frames']
    out_dir = Path(out_dir)
    frames_dir = out_dir / 'composite-frames'
    rec = Recorder(frames_dir, fps=fps)
    for i, entry in enumerate(entries):
        scene_path = scene_frames_dir / entry['scene_file']
        scene_img = Image.open(scene_path).convert('RGB')
        panel_img = panel_mod.render_panel(entry['panel'], size=(panel_w, out_size[1]))
        comp = composite_frame(scene_img, panel_img, out_size=out_size, panel_w=panel_w)
        rgb = np.asarray(comp, dtype=np.uint8)
        # Recorder requires even HxW; out_size is forced even above.
        rec.add(rgb, render_id=f'demo-composite-{i:06d}', simulation_time=(i + 1) / fps)
    mp4 = out_dir / 'walkthrough.mp4'
    if mp4.exists():
        mp4.unlink()
    for side in ('.ffconcat', '.timing.json'):
        p = mp4.with_suffix(side)
        if p.exists():
            p.unlink()
    return rec.encode(mp4, ffmpeg=ffmpeg)
