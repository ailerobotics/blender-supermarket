"""Assemble the captain-facing demo cut of one mission run (CPU only, no GPU, run dir read-only).

Input: the run dir's recorded overhead walkthrough_chase frames (mission-frames/), the per-tick
nav-trace (which stage is active, base pose, coupled) and mission.json (stage results + the
workaround labels the lanes record). Output: a 1280x720 30 fps H.264 mp4 laid out as

    +----------------------------------+-----------+
    |  recorded scene (960x720)        |  stage    |
    |                                  |  ladder,  |
    |  [time-lapse badge]   [sim clock]|  route    |
    |  == stage caption + labels ==    |  map      |
    +----------------------------------+-----------+

with a title card, one caption per stage for all seven stages (performed stages over their own
footage; shortcut-synthesized / not-reached stages as honest cards), every workaround label on screen,
slow stretches (e.g. the couple dock alignment) time-lapsed, and a closing result card.

Honesty rules: only recorded frames are shown, never interpolated; a frame is only ever HELD (a
still) for a stage too short to own a frame, and the badge then says "still". Speed-ups are always
on screen. Shortcut stages are never shown as performed.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import run_evidence as E

OUT_W, OUT_H = 1280, 720
SCENE_W, SCENE_H = 960, 720
PANEL_W = OUT_W - SCENE_W
FPS = 30

BASE_SPEED = 4.0            # footage speed while the robot moves (3 fps capture -> 12 fps shown)
SLOW_SPEED = 16.0           # creeping / settling stretches (dock alignment, waits)
MAX_STAGE_S = 20.0          # cap on one stage's on-screen footage; faster time-lapse beyond it
MOVING_LIN = 0.05           # m/s
MOVING_YAW = 0.08           # rad/s

TITLE_S, STAGE_CARD_S, END_S, STILL_S = 4.0, 2.5, 6.0, 2.0
MIN_STAGE_S = 2.5           # every performed stage stays on screen at least this long (caption legible)
ARM_STAGES = ('collect', 'transfer', 'verify')   # the base is parked; the arm moves (not in the trace)

BG = (18, 22, 30)
PANEL_BG = (26, 31, 42)
FG = (235, 238, 242)
DIM = (140, 148, 160)
ACCENT = (90, 170, 255)
AMBER = (255, 190, 60)
GREEN = (90, 200, 120)
RED = (235, 85, 85)

_STATUS_STYLE = {'succeeded': ('done', GREEN), 'shortcut': ('synth', AMBER), 'failed': ('FAILED', RED),
                 'unavailable': ('n/a', RED), 'started': ('...', ACCENT), 'missing': ('-', DIM)}


# ---------------- fonts ----------------
_FONT_DIRS = ('/usr/share/fonts/truetype/dejavu', '/usr/share/fonts/dejavu', '/usr/share/fonts/TTF')
_font_cache = {}


def font(size, bold=False):
    from PIL import ImageFont
    key = (size, bold)
    if key not in _font_cache:
        name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
        for d in _FONT_DIRS:
            p = Path(d) / name
            if p.is_file():
                _font_cache[key] = ImageFont.truetype(str(p), size)
                break
        else:
            try:
                _font_cache[key] = ImageFont.load_default(size=size)
            except TypeError:            # Pillow < 10.1
                _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def _wrap(draw, text, fnt, width):
    words, lines, cur = str(text).split(), [], ''
    for w in words:
        trial = (cur + ' ' + w).strip()
        if draw.textlength(trial, font=fnt) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _text_block(draw, xy, text, fnt, fill, width, spacing=4):
    x, y = xy
    for line in _wrap(draw, text, fnt, width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + spacing
    return y


# ---------------- the edit plan ----------------
@dataclass
class Shot:
    """One entry of the edit decision list: a recorded frame (or a rendered card) and how long it shows."""
    kind: str                     # 'frame' | 'still' | 'card'
    duration: float               # seconds on screen
    stage: str | None = None
    frame: E.Frame | None = None
    speed: float = 1.0
    card: str | None = None       # card id when kind == 'card'


@dataclass
class Plan:
    run: E.Run
    status: dict
    labels: list
    windows: dict
    t_uncouple: float | None
    shots: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    views: dict = field(default_factory=dict)    # extra camera name -> [Frame]

    def stage_labels(self, stage):
        return [l for l in self.labels if l.stage == stage]

    def run_labels(self):
        return [l for l in self.labels if l.stage is None]


def _moving(row):
    return row is not None and (row['lin'] >= MOVING_LIN or abs(row.get('rw', 0.0)) >= MOVING_YAW)


def _smoothed_motion(run, frames, window=1.0):
    """Per frame: did the base move within +-window s? (debounces the speed so it does not flicker)."""
    raw = [_moving(run.row_at(f.t)) for f in frames]
    times = [f.t for f in frames]
    out = []
    j0 = 0
    for i, t in enumerate(times):
        while times[j0] < t - window:
            j0 += 1
        j1 = i
        while j1 + 1 < len(times) and times[j1 + 1] <= t + window:
            j1 += 1
        out.append(any(raw[j0:j1 + 1]))
    return out


def footage_shots(run, stage, frames, next_t=None):
    """Shots for one stage's recorded frames: motion-adaptive time-lapse (base moving -> BASE_SPEED,
    creeping/settling -> SLOW_SPEED; arm stages always BASE_SPEED since the trace cannot see the arm),
    scaled up so no stage exceeds MAX_STAGE_S, and held (labelled STILL) up to MIN_STAGE_S."""
    if not frames:
        return []
    moving = [True] * len(frames) if stage in ARM_STAGES else _smoothed_motion(run, frames)
    dts = []
    for i, f in enumerate(frames):
        nxt = frames[i + 1].t if i + 1 < len(frames) else (next_t if next_t is not None else f.t + 1 / 3)
        dts.append(max(1e-3, min(2.0, nxt - f.t)))
    speeds = [BASE_SPEED if m else SLOW_SPEED for m in moving]
    total = sum(dt / s for dt, s in zip(dts, speeds))
    scale = max(1.0, total / MAX_STAGE_S)
    shots = [Shot('frame', dt / (s * scale), stage, f, s * scale) for f, dt, s in zip(frames, dts, speeds)]
    shown = sum(s.duration for s in shots)
    if shown < MIN_STAGE_S:
        shots.append(Shot('still', MIN_STAGE_S - shown, stage, frames[-1], 0.0))
    return shots


def build_plan(run, *, checks=None, views=None):
    status = E.stage_status(run)
    plan = Plan(run=run, status=status, labels=E.workaround_labels(run), windows=E.stage_windows(run),
                t_uncouple=E.uncouple_time(run), checks=list(checks or []), views=dict(views or {}))
    shots = [Shot('card', TITLE_S, card='title')]
    by_stage = E.frames_by_stage(run)
    last_frame = None
    unreached = []
    for stage in E.STAGES:
        st = status[stage]
        if st == 'shortcut':
            shots.append(Shot('card', STAGE_CARD_S, stage, card='shortcut'))
            continue
        if st == 'missing':                   # never begun per the event log: not reached
            unreached.append(stage)
            continue
        frames = by_stage.get(stage) or []
        if frames:
            after = [f.t for f in run.frames if f.t > frames[-1].t]
            shots.extend(footage_shots(run, stage, frames, after[0] if after else None))
            last_frame = frames[-1]
        else:
            # Too short to own a recorded frame (e.g. park latches in one tick): HOLD the latest real
            # frame at or before the stage, labelled 'still', so the caption still gets its moment.
            lo = plan.windows.get(stage, (None, None))[0]
            prior = [f for f in run.frames if lo is None or f.t <= lo + 0.5]
            hold = prior[-1] if prior else last_frame
            if hold is not None:
                shots.append(Shot('still', STILL_S, stage, hold, 0.0))
            else:
                shots.append(Shot('card', STAGE_CARD_S, stage, card='stage'))
    if unreached:
        shots.append(Shot('card', STAGE_CARD_S + 0.5, None, card='unreached'))
    shots.append(Shot('card', END_S, card='end'))
    plan.shots = shots
    plan.unreached = unreached
    return plan


# ---------------- rendering ----------------
class Renderer:
    def __init__(self, plan):
        from PIL import Image
        self.plan = plan
        self.run = plan.run
        self.model = E.cart_model(self.run)
        self._Image = Image
        self._scene_cache = {}
        self._band_cache = {}
        self._panel_cache = {}
        self._map = None

    # ---- static pieces ----
    def _map_geometry(self):
        if self._map is not None:
            return self._map
        run, m = self.run, self.model
        pts = [(r['x'], r['y']) for r in run.trace[::5]] or [(0.0, 0.0)]
        extra = [m.bay_center, m.initial_cart[:2]]
        xs = [p[0] for p in pts + extra]
        ys = [p[1] for p in pts + extra]
        x0, x1, y0, y1 = min(xs) - 1.0, max(xs) + 1.0, min(ys) - 1.0, max(ys) + 1.0
        box = (12, 0, PANEL_W - 12, 190)
        bw, bh = box[2] - box[0], box[3] - box[1]
        scale = min(bw / max(x1 - x0, 1e-3), bh / max(y1 - y0, 1e-3))
        ox = box[0] + (bw - (x1 - x0) * scale) / 2
        oy = box[1] + (bh - (y1 - y0) * scale) / 2

        def to_px(x, y):
            return (ox + (x - x0) * scale, oy + (y1 - y) * scale)
        path = [to_px(r['x'], r['y']) for r in run.trace[::10]]
        times = [r['t'] for r in run.trace[::10]]
        self._map = (to_px, scale, path, times)
        return self._map

    def _panel(self, stage):
        """The right-hand panel (stage ladder + map base) for the active stage, cached per stage."""
        if stage in self._panel_cache:
            img, off = self._panel_cache[stage]
            return img.copy(), off
        from PIL import ImageDraw
        img = self._Image.new('RGB', (PANEL_W, OUT_H), PANEL_BG)
        d = ImageDraw.Draw(img)
        d.text((16, 14), 'RESTOCK MISSION', font=font(20, True), fill=FG)
        d.text((16, 40), f'{self.run.name}  ·  Isaac Sim', font=font(14), fill=DIM)
        y = 72
        now = E.STAGES.index(stage) if stage in E.STAGES else len(E.STAGES)
        for i, s in enumerate(E.STAGES):
            st = self.plan.status[s]
            # Progress up to the shown moment: earlier stages show their outcome, the shown stage is
            # 'now', later ones are pending (a shortcut-synthesized stage always says so).
            if i < now or st == 'shortcut':
                word, colour = _STATUS_STYLE[st]
            elif i == now:
                word, colour = 'now', ACCENT
            else:
                word, colour = _STATUS_STYLE['missing']
            active = s == stage
            if active:
                d.rounded_rectangle((8, y - 4, PANEL_W - 8, y + 26), radius=6, fill=(44, 58, 84))
            d.text((16, y), f'{i + 1}', font=font(16, True), fill=ACCENT if active else DIM)
            d.text((40, y), E.TITLES[s], font=font(16, active), fill=FG if active or st != 'missing' else DIM)
            tw = d.textlength(word, font=font(13, True))
            d.text((PANEL_W - 18 - tw, y + 2), word, font=font(13, True), fill=colour)
            y += 32
            for lab in self.plan.stage_labels(s):
                y = _text_block(d, (40, y - 4), lab.text, font(11), AMBER, PANEL_W - 56, 2) + 4
        run_labels = self.plan.run_labels()
        if run_labels:
            y += 4
            for lab in run_labels:
                y = _text_block(d, (16, y), lab.text, font(11), AMBER, PANEL_W - 32, 2) + 2
        map_top = max(y + 10, OUT_H - 250)
        d.text((16, map_top), 'route (top view, measured base path)', font=font(12), fill=DIM)
        to_px, scale, path, _ = self._map_geometry()
        off = map_top + 22
        moved = [(px, py + off) for px, py in path]
        if len(moved) > 1:
            d.line(moved, fill=(70, 78, 92), width=2)
        bx, by = to_px(*self.model.bay_center)
        hw, hh = self.model.bay_size[0] / 2 * scale, self.model.bay_size[1] / 2 * scale
        d.rectangle((bx - hw, by - hh + off, bx + hw, by + hh + off), outline=(200, 200, 210), width=1)
        d.text((bx + hw + 3, by - 7 + off), 'bay', font=font(11), fill=DIM)
        self._panel_cache[stage] = (img, off)
        return img.copy(), off

    def _band(self, stage, uncoupled):
        key = (stage, uncoupled)
        if key in self._band_cache:
            return self._band_cache[key]
        from PIL import ImageDraw
        band_h = 112
        band = self._Image.new('RGBA', (SCENE_W, band_h), (12, 16, 24, 205))
        d = ImageDraw.Draw(band)
        idx = E.STAGES.index(stage) + 1
        d.text((18, 10), f'{idx}/7', font=font(26, True), fill=ACCENT + (255,))
        d.text((80, 10), E.TITLES[stage].upper(), font=font(26, True), fill=FG + (255,))
        caption = E.CAPTIONS[stage]
        if uncoupled:
            caption = {'park': 'Parked at the transfer bay; tow hitch released (uncoupled at the parking spot)',
                       'return': 'Uncoupled at the parking spot; returning to the dock without the cart'}.get(
                stage, caption + ' (cart uncoupled)')
        y = _text_block(d, (20, 48), caption, font(17), FG + (255,), SCENE_W - 40, 3)
        labels = self.plan.stage_labels(stage)
        if labels:
            text = 'WORKAROUND: ' + '; '.join(l.text for l in labels)
            _text_block(d, (20, y + 2), text, font(14, True), AMBER + (255,), SCENE_W - 40, 2)
        self._band_cache[key] = band
        return band

    def _scene(self, frame):
        if frame.index in self._scene_cache:
            return self._scene_cache[frame.index]
        with self._Image.open(frame.file) as im:
            img = im.convert('RGB')
        if img.size != (SCENE_W, SCENE_H):
            img = img.resize((SCENE_W, SCENE_H), self._Image.BILINEAR)
        self._scene_cache = {frame.index: img}      # one-entry cache: shots are in time order
        return img

    def _pip(self, stage, t):
        """Optional inset from an extra recorded camera that suits the stage (none recorded today)."""
        prefs = {'collect': ('source', 'arm', 'wrist'), 'transfer': ('shelf', 'bay', 'arm', 'wrist'),
                 'verify': ('shelf', 'bay', 'arm')}.get(stage, ())
        for want in prefs:
            for name, frames in self.plan.views.items():
                if want in name.lower() and frames:
                    best = min(frames, key=lambda f: abs(f.t - t))
                    if abs(best.t - t) <= 1.0:
                        return name, best
        return None

    # ---- frames ----
    def footage(self, shot):
        from PIL import ImageDraw
        run, frame, stage = self.run, shot.frame, shot.stage
        t = frame.t
        uncoupled = self.plan.t_uncouple is not None and t >= self.plan.t_uncouple
        canvas = self._Image.new('RGB', (OUT_W, OUT_H), BG)
        canvas.paste(self._scene(frame), (0, 0))
        pip = self._pip(stage, t)
        if pip is not None:
            name, pf = pip
            with self._Image.open(pf.file) as im:
                inset = im.convert('RGB').resize((320, 240), self._Image.BILINEAR)
            canvas.paste(inset, (SCENE_W - 330, 10))
            ImageDraw.Draw(canvas).text((SCENE_W - 326, 252), name, font=font(12, True), fill=FG)
        band = self._band(stage, uncoupled)
        base = canvas.crop((0, SCENE_H - band.size[1], SCENE_W, SCENE_H)).convert('RGBA')
        canvas.paste(self._Image.alpha_composite(base, band).convert('RGB'), (0, SCENE_H - band.size[1]))
        d = ImageDraw.Draw(canvas)
        # top-left: speed badge; top-right: sim clock + coupling.
        if shot.kind == 'still':
            badge, colour = 'STILL (stage shorter than one capture)', AMBER
        elif shot.speed >= 8:
            badge, colour = f'x{shot.speed:.0f} TIME-LAPSE', AMBER
        else:
            badge, colour = f'x{shot.speed:.0f}', FG
        tw = d.textlength(badge, font=font(16, True))
        d.rounded_rectangle((10, 10, 26 + tw, 38), radius=6, fill=(12, 16, 24))
        d.text((18, 14), badge, font=font(16, True), fill=colour)
        row = run.row_at(t)
        coupled = 'coupled' if (row and row['coupled']) else 'uncoupled'
        clock = f'sim t = {t:6.1f} s   ·   {coupled}'
        cw = d.textlength(clock, font=font(15))
        if pip is None:
            d.rounded_rectangle((SCENE_W - 26 - cw, 10, SCENE_W - 10, 38), radius=6, fill=(12, 16, 24))
            d.text((SCENE_W - 18 - cw, 15), clock, font=font(15), fill=FG)
        else:
            d.text((18, 46), clock, font=font(15), fill=FG)
        # right panel with the live robot marker on the route map.
        panel, off = self._panel(stage)
        pd = ImageDraw.Draw(panel)
        to_px, scale, path, times = self._map_geometry()
        done = [(px, py + off) for (px, py), tt in zip(path, times) if tt <= t]
        if len(done) > 1:
            pd.line(done, fill=ACCENT, width=3)
        if row is not None:
            x, y = to_px(row['x'], row['y'])
            y += off
            hx, hy = math.cos(row['yaw']) * 9, -math.sin(row['yaw']) * 9
            pd.ellipse((x - 5, y - 5, x + 5, y + 5), fill=FG)
            pd.line((x, y, x + hx, y + hy), fill=FG, width=2)
            cart = E.estimated_cart_pose(run, self.model, t)
            cx, cy = to_px(cart[0], cart[1])
            pd.rectangle((cx - 4, cy - 4 + off, cx + 4, cy + 4 + off), outline=AMBER, width=2)
        canvas.paste(panel, (SCENE_W, 0))
        return canvas

    def card(self, shot):
        from PIL import ImageDraw
        img = self._Image.new('RGB', (OUT_W, OUT_H), BG)
        d = ImageDraw.Draw(img)
        run, plan = self.run, self.plan
        W = OUT_W - 160
        if shot.card == 'title':
            d.text((80, 90), 'Mobile restock mission', font=font(48, True), fill=FG)
            d.text((80, 156), 'Isaac Sim walkthrough · mobile UR10 + rollwagen', font=font(24), fill=DIM)
            d.text((80, 210), f'{run.name}', font=font(22, True), fill=ACCENT)
            y = 260
            summary = ('Seven stages: collect -> couple -> outbound tow -> park & uncouple -> transfer -> '
                       'verify -> return. The carton rides on the rollwagen deck from collect to transfer.')
            y = _text_block(d, (80, y), summary, font(20), FG, W, 6) + 16
            labels = plan.labels
            if labels:
                d.text((80, y), 'Workarounds in this run (also shown on screen per stage):',
                       font=font(18, True), fill=AMBER)
                y += 30
                for lab in labels:
                    who = f'{lab.stage}: ' if lab.stage else ''
                    y = _text_block(d, (100, y), who + lab.text, font(18), AMBER, W - 20, 4) + 2
                y += 10
            note = (f'Footage is the recorded overhead chase camera; time-lapse x{BASE_SPEED:.0f} while moving, '
                    f'faster through slow stretches (always shown top-left). Nothing is interpolated.')
            _text_block(d, (80, max(y, OUT_H - 120)), note, font(16), DIM, W, 4)
        elif shot.card in ('shortcut', 'stage'):
            stage = shot.stage
            idx = E.STAGES.index(stage) + 1
            d.text((80, 200), f'{idx}/7', font=font(40, True), fill=ACCENT)
            d.text((180, 200), E.TITLES[stage].upper(), font=font(40, True), fill=FG)
            y = _text_block(d, (80, 270), E.CAPTIONS[stage], font(24), FG, W, 6) + 20
            if shot.card == 'shortcut':
                shortcut = run.mission.get('shortcut') or {}
                text = (f'NOT PERFORMED IN THIS RUN: start-phase shortcut (start_at={shortcut.get("start_at")}). '
                        'The simulator placed the carton on the rollwagen deck and posed the robot; this stage '
                        'was synthesized, not actuated.')
                _text_block(d, (80, y), text, font(22, True), AMBER, W, 6)
        elif shot.card == 'unreached':
            d.text((80, 200), 'Not reached in this run', font=font(36, True), fill=RED)
            y = 260
            for stage in getattr(plan, 'unreached', []):
                idx = E.STAGES.index(stage) + 1
                y = _text_block(d, (80, y), f'{idx}/7  {E.TITLES[stage]} -- {E.CAPTIONS[stage]}',
                                font(22), FG, W, 6) + 8
        elif shot.card == 'end':
            ok = run.mission.get('state') == 'complete'
            d.text((80, 70), E.result_line(run), font=font(44, True), fill=GREEN if ok else RED)
            y = 130
            if not ok:
                y = _text_block(d, (80, y), E.short_reason(run.mission.get('reason'), 160), font(18), FG, W, 4) + 10
            for i, s in enumerate(E.STAGES):
                word, colour = _STATUS_STYLE[plan.status[s]]
                lo_hi = plan.windows.get(s)
                dur = f'{lo_hi[1] - lo_hi[0]:.1f} s sim' if lo_hi else ''
                d.text((80, y), f'{i + 1}. {E.TITLES[s]}', font=font(20), fill=FG)
                d.text((420, y), word, font=font(20, True), fill=colour)
                d.text((560, y), dur, font=font(18), fill=DIM)
                y += 30
            y += 8
            for c in plan.checks:
                colour = {'PASS': GREEN, 'WARN': AMBER}.get(c.result, RED)
                d.text((80, y), f'{c.name}', font=font(16), fill=FG)
                d.text((300, y), c.result, font=font(16, True), fill=colour)
                y = _text_block(d, (420, y), c.evidence, font(13), DIM, OUT_W - 500, 2) + 4
            if plan.labels:
                y += 6
                txt = 'Workarounds: ' + '; '.join((f'{l.stage}: ' if l.stage else '') + l.text for l in plan.labels)
                y = _text_block(d, (80, y), txt, font(15, True), AMBER, W, 3)
            pv = run.mission.get('physics_verified')
            if pv is not None:
                _text_block(d, (80, OUT_H - 50), f'physics_verified={pv} (mission.json) -- mission success alone '
                            'does not verify video, contacts or autonomy.', font(14), DIM, W, 2)
        return img


# ---------------- encoding ----------------
def find_ffmpeg(explicit=None):
    for cand in (explicit, os.environ.get('FFMPEG'), shutil.which('ffmpeg')):
        if cand and Path(cand).is_file():
            return str(cand)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def discover_views(run):
    """Extra Recorder-format camera dirs in the run (``<name>/frames.json``) besides mission-frames."""
    views = {}
    for manifest in sorted(run.run_dir.glob('*/frames.json')):
        if manifest.parent.name == 'mission-frames':
            continue
        try:
            data = json.loads(manifest.read_text())
            views[manifest.parent.name] = [
                E.Frame(i, manifest.parent / f['file'], float(f['simulation_time']),
                        float(f['simulation_time']) - run.clock_offset)
                for i, f in enumerate(data.get('frames', []))]
        except (ValueError, KeyError, TypeError):
            continue
    return views


def render_video(plan, output, *, ffmpeg, crf=20, preset='veryfast'):
    """Encode the plan to ``output`` (refuses to overwrite). Returns the cut summary dict."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    renderer = Renderer(plan)
    cmd = [ffmpeg, '-nostdin', '-n', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{OUT_W}x{OUT_H}', '-r', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
           '-preset', preset, '-crf', str(crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    emitted, acc, shown = 0, 0.0, 0
    edl = []
    try:
        for shot in plan.shots:
            start = round(acc * FPS)
            acc += shot.duration
            n = round(acc * FPS) - start
            if n <= 0:
                continue                       # time-lapse: this capture falls between output frames
            img = renderer.card(shot) if shot.kind == 'card' else renderer.footage(shot)
            data = img.tobytes()
            for _ in range(n):
                proc.stdin.write(data)
            emitted += n
            shown += 1
            edl.append({'kind': shot.kind, 'stage': shot.stage, 'card': shot.card,
                        'frame': shot.frame.file.name if shot.frame else None,
                        'sim_t': round(shot.frame.t, 3) if shot.frame else None,
                        'speed': round(shot.speed, 2), 'out_frames': n})
        proc.stdin.close()
        if proc.wait(timeout=600) != 0:
            raise RuntimeError(f'ffmpeg exited {proc.returncode}')
    except BaseException:
        proc.kill()
        raise
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError('encoder produced no video')
    summary = {'video': str(output), 'run': str(plan.run.run_dir), 'fps': FPS, 'size': [OUT_W, OUT_H],
               'output_frames': emitted, 'duration_s': round(emitted / FPS, 2),
               'recorded_frames_used': sum(1 for e in edl if e['kind'] == 'frame'),
               'recorded_frames_total': len(plan.run.frames), 'clock_offset_s': plan.run.clock_offset,
               'clock_source': plan.run.clock_source, 'result': E.result_line(plan.run),
               'stage_status': plan.status,
               'labels': [{'stage': l.stage, 'key': l.key, 'value': l.value} for l in plan.labels],
               'checks': [{'name': c.name, 'result': c.result, 'evidence': c.evidence} for c in plan.checks],
               'edl': edl}
    output.with_suffix('.cut.json').write_text(json.dumps(summary, indent=1, default=str) + '\n')
    return summary
