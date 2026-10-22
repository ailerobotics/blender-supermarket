"""Render the decision/reason tree panel for one frame as a PIL image.

The panel mirrors the live ``bt_view.html`` colour language (status dots + per-node
feedback) but as a raster surface that composites into the video. It draws the full
PR153 tree (seven stage subtrees + their step children), highlights the active node,
colours every node by status and prints each decision's reason -- a value MEASURED from
the scene that frame -- beside it.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

# bt_view.html palette.
BG = (13, 17, 23)
PANEL = (22, 27, 34)
BORDER = (48, 54, 61)
TEXT = (201, 209, 217)
MUTED = (139, 148, 158)
ACCENT = (88, 166, 255)
STATUS_RGB = {'success': (46, 160, 67), 'running': (210, 153, 34),
              'failure': (218, 54, 51), 'idle': (72, 79, 88)}

_FONT_DIR = '/usr/share/fonts/truetype/dejavu'


def _font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    try:
        return ImageFont.truetype(f'{_FONT_DIR}/{name}', size)
    except Exception:
        return ImageFont.load_default()


def _truncate(draw, text, font, max_w):
    if draw.textlength(text, font=font) <= max_w:
        return text
    ell = '…'
    while text and draw.textlength(text + ell, font=font) > max_w:
        text = text[:-1]
    return text + ell


def render_panel(entry, size=(480, 1080)):
    """Render one frame's panel. ``entry`` keys:

    * ``tree``   -- serialised node dict (from tree_model.annotate)
    * ``header`` -- {stage_title, clock, frame, mission_state}
    * ``chips``  -- list of (label, value) measured this frame
    """
    w, h = size
    img = Image.new('RGB', (w, h), BG)
    d = ImageDraw.Draw(img)
    pad = 16

    f_title = _font(22, bold=True)
    f_sub = _font(14)
    f_stage = _font(16, bold=True)
    f_step = _font(14)
    f_reason = _font(12)
    f_chip = _font(13)

    # --- Header ---
    y = pad
    d.text((pad, y), 'DECISION / REASON TREE', font=f_title, fill=TEXT)
    y += 30
    hdr = entry.get('header', {})
    sub = f"mission {hdr.get('mission_state', 'running')}  ·  t {hdr.get('clock', 0):.1f}s  ·  frame {hdr.get('frame', 0)}"
    d.text((pad, y), sub, font=f_sub, fill=MUTED)
    y += 22
    stage_line = f"stage: {hdr.get('stage_title', '')}"
    d.text((pad, y), _truncate(d, stage_line, f_sub, w - 2 * pad), font=f_sub, fill=ACCENT)
    y += 24
    d.line([(pad, y), (w - pad, y)], fill=BORDER, width=1)
    y += 10

    # --- Tree ---
    tree = entry.get('tree', {})
    row_h = 24
    dot_r = 5
    for stage_node in tree.get('children', []):
        # Stage header row.
        st = stage_node.get('status', 'idle')
        cx = pad + dot_r + 2
        cy = y + row_h // 2
        d.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=STATUS_RGB.get(st, STATUS_RGB['idle']))
        name = stage_node.get('name', '').replace('stage:', '')
        d.text((cx + dot_r + 8, y + 3), name, font=f_stage, fill=TEXT)
        if stage_node.get('feedback'):
            d.text((cx + dot_r + 8 + 150, y + 4), _truncate(d, stage_node['feedback'], f_reason, w - pad - (cx + dot_r + 8 + 150)),
                   font=f_reason, fill=MUTED)
        y += row_h
        # Step rows (indented).
        for step in stage_node.get('children', []):
            sst = step.get('status', 'idle')
            ix = pad + 22 + dot_r
            iy = y + row_h // 2
            # connector
            d.line([(pad + 12, y), (pad + 12, iy)], fill=BORDER, width=1)
            d.line([(pad + 12, iy), (ix - dot_r - 3, iy)], fill=BORDER, width=1)
            d.ellipse([ix - dot_r, iy - dot_r, ix + dot_r, iy + dot_r],
                      fill=STATUS_RGB.get(sst, STATUS_RGB['idle']))
            label = step.get('name', '').split(':', 1)[-1]
            tx = ix + dot_r + 8
            lcol = TEXT if sst in ('running', 'success') else MUTED
            d.text((tx, y + 4), label, font=f_step, fill=lcol)
            reason = step.get('feedback', '')
            if reason:
                lw = d.textlength(label, font=f_step)
                rx = tx + lw + 10
                d.text((rx, y + 5), _truncate(d, reason, f_reason, w - pad - rx),
                       font=f_reason, fill=STATUS_RGB['running'] if sst == 'running' else MUTED)
            y += row_h
        y += 4

    # --- Measured-from-scene footer ---
    chips = entry.get('chips', [])
    if chips:
        fy = h - pad - 18 - (len(chips) // 2 + len(chips) % 2) * 26
        d.line([(pad, fy - 12), (w - pad, fy - 12)], fill=BORDER, width=1)
        d.text((pad, fy - 10), 'MEASURED FROM SCENE', font=f_sub, fill=MUTED)
        fy += 18
        col_w = (w - 2 * pad) // 2
        for i, (label, value) in enumerate(chips):
            col = i % 2
            row = i // 2
            x = pad + col * col_w
            yy = fy + row * 26
            d.text((x, yy), f'{label}', font=f_chip, fill=MUTED)
            d.text((x, yy + 0), '', font=f_chip, fill=TEXT)
            vw = d.textlength(value, font=f_chip)
            d.text((x + col_w - vw - 14, yy), value, font=f_chip, fill=TEXT)
    return img
