#!/usr/bin/env python3
"""Render product labels, shelf-edge price tags and aisle signs with PIL.

Outputs (PNG):
  assets/labels/<sku>.png      product artwork (front label, or a full wrap for round forms)
  assets/labels/tag_<sku>.png  shelf-edge price tag (white, price + name + barcode)
  assets/signs/aisle_<n>.png   hanging aisle sign
  assets/signs/store_sign.png  facade sign
Round forms (can/jar/tub/bottle) get a wrap label whose centre (u=0.5) is the front.
"""
import json, math, os, random, sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catalog import SKUS, CATEGORIES

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAB = os.path.join(ROOT, "assets", "labels"); SIG = os.path.join(ROOT, "assets", "signs")
os.makedirs(LAB, exist_ok=True); os.makedirs(SIG, exist_ok=True)
FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
F_BOLD, F_REG, F_SERIF = FONT_DIR + "DejaVuSans-Bold.ttf", FONT_DIR + "DejaVuSans.ttf", FONT_DIR + "DejaVuSerif-Bold.ttf"
PX_PER_M = 3200
ROUND = {"can", "jar", "tub", "bottle"}

def hexrgb(h): h = h.lstrip("#"); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
def mix(a, b, t): return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))
def font(path, size): return ImageFont.truetype(path, max(8, int(size)))

def fit_text(draw, text, path, max_w, max_h):
    size = max_h
    while size > 8:
        f = font(path, size); box = draw.textbbox((0, 0), text, font=f)
        if box[2] - box[0] <= max_w and box[3] - box[1] <= max_h: return f
        size *= 0.92
    return font(path, 8)

def text_center(draw, cx, cy, text, f, fill):
    box = draw.textbbox((0, 0), text, font=f); w, h = box[2] - box[0], box[3] - box[1]
    draw.text((cx - w / 2 - box[0], cy - h / 2 - box[1]), text, font=f, fill=fill)

def barcode(draw, x, y, w, h, rng, fg=(0, 0, 0), bg=(255, 255, 255)):
    draw.rectangle([x, y, x + w, y + h], fill=bg)
    n = 44; bw = w / n; cx = x + bw
    for i in range(n - 2):
        if rng.random() < 0.55: draw.rectangle([cx, y + h * 0.08, cx + bw * rng.choice([0.6, 1.0, 1.3]), y + h * 0.78], fill=fg)
        cx += bw
    f = font(F_REG, h * 0.2); draw.text((x + w * 0.2, y + h * 0.78), " ".join(str(rng.randint(0, 9)) for _ in range(12)), font=f, fill=fg)

def noise_overlay(img, amount=10):
    rng = random.Random(1); px = img.load(); W, H = img.size
    small = Image.effect_noise((max(1, W // 4), max(1, H // 4)), amount).resize((W, H)).convert("L")
    return Image.composite(img, Image.blend(img, Image.new("RGB", (W, H), (0, 0, 0)), 0.15), small.point(lambda v: 255 - abs(v - 128)))

def hero_shape(draw, cx, cy, r, accent, fg, kind, rng):
    """A simple 'product hero' so labels are not just text."""
    if kind in ("cereal", "snacks", "bakery"):        # sunburst
        for i in range(24):
            a = i * math.pi / 12; a2 = a + math.pi / 24
            draw.polygon([(cx, cy), (cx + r * math.cos(a), cy + r * math.sin(a)), (cx + r * math.cos(a2), cy + r * math.sin(a2))], fill=accent if i % 2 else mix(accent, fg, 0.3))
        draw.ellipse([cx - r * 0.55, cy - r * 0.55, cx + r * 0.55, cy + r * 0.55], fill=fg)
    elif kind in ("soft_drinks", "water_juice", "beer_wine"):   # swoosh / droplet
        draw.pieslice([cx - r, cy - r, cx + r, cy + r], 200, 340, fill=accent)
        draw.ellipse([cx - r * 0.35, cy - r * 0.9, cx + r * 0.35, cy - r * 0.1], fill=fg)
    elif kind in ("canned", "pasta_rice", "condiments", "baking"):   # plate / ingredient blobs
        draw.ellipse([cx - r, cy - r * 0.7, cx + r, cy + r * 0.7], fill=mix(accent, (255, 255, 255), 0.5))
        for i in range(5):
            a = i * 2 * math.pi / 5; rr = r * 0.28
            x, y = cx + r * 0.5 * math.cos(a), cy + r * 0.35 * math.sin(a)
            draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=accent)
    elif kind in ("cleaning", "paper", "personal_care"):   # sparkle
        for a in (0, math.pi / 2):
            draw.polygon([(cx + r * math.cos(a), cy + r * math.sin(a)), (cx + r * 0.15 * math.cos(a + math.pi / 2), cy + r * 0.15 * math.sin(a + math.pi / 2)),
                          (cx - r * math.cos(a), cy - r * math.sin(a)), (cx - r * 0.15 * math.cos(a + math.pi / 2), cy - r * 0.15 * math.sin(a + math.pi / 2))], fill=accent)
        draw.ellipse([cx - r * 0.25, cy - r * 0.25, cx + r * 0.25, cy + r * 0.25], fill=fg)
    elif kind == "dairy":
        draw.rounded_rectangle([int(cx - r * 0.9), int(cy - r * 0.5), int(cx + r * 0.9), int(cy + r * 0.6)], radius=max(1, int(r * 0.3)), fill=mix(accent, (255, 255, 255), 0.4))
        draw.ellipse([cx - r * 0.5, cy - r * 0.9, cx + r * 0.5, cy - r * 0.2], fill=accent)
    else:                                              # confectionery / pet / coffee: rings
        for k, t in enumerate((1.0, 0.7, 0.4)):
            draw.ellipse([cx - r * t, cy - r * t, cx + r * t, cy + r * t], fill=accent if k % 2 == 0 else fg)

def draw_panel(img, x0, y0, x1, y1, sku, rng, main=True):
    """Draw one label panel in box (x0,y0,x1,y1)."""
    d = ImageDraw.Draw(img)
    bg, fg, ac = hexrgb(sku["bg"]), hexrgb(sku["fg"]), hexrgb(sku["accent"])
    W, H = x1 - x0, y1 - y0
    # brand band
    band_h = H * 0.13
    d.rectangle([x0, y0, x1, y0 + band_h], fill=ac)
    f = fit_text(d, sku["brand"].upper(), F_BOLD, W * 0.85, band_h * 0.6)
    text_center(d, x0 + W / 2, y0 + band_h / 2, sku["brand"].upper(), f, fill=mix(bg, (0, 0, 0), 0.5) if sum(ac) > 380 else (255, 255, 255))
    # hero
    hero_r = min(W, H) * 0.24
    hero_shape(d, x0 + W / 2, y0 + H * 0.46, hero_r, ac, fg, sku["cat"], rng)
    # product name (below hero) with a translucent backing for contrast
    name_h = H * 0.15
    ny = y0 + H * 0.68
    d.rectangle([x0, ny - name_h * 0.55, x1, ny + name_h * 0.55], fill=mix(bg, (0, 0, 0), 0.18))
    f = fit_text(d, sku["name"], F_SERIF if sku["cat"] in ("beer_wine", "confectionery") else F_BOLD, W * 0.9, name_h * 0.85)
    text_center(d, x0 + W / 2, ny, sku["name"], f, fill=fg)
    f = fit_text(d, sku["sub"], F_REG, W * 0.85, H * 0.06)
    text_center(d, x0 + W / 2, ny + name_h * 0.75, sku["sub"], f, fill=mix(fg, bg, 0.15))
    # bottom: weight pill + small barcode
    pill_h = max(6, H * 0.08)
    d.rounded_rectangle([int(x0 + W * 0.06), int(y1 - H * 0.04 - pill_h), int(x0 + W * 0.42), int(y1 - H * 0.04)], radius=max(1, int(pill_h / 2)), fill=fg)
    f = fit_text(d, sku["weight"], F_BOLD, W * 0.3, pill_h * 0.6)
    text_center(d, x0 + W * 0.24, y1 - H * 0.04 - pill_h / 2, sku["weight"], f, fill=bg)
    if main:
        barcode(d, x0 + W * 0.55, y1 - H * 0.04 - pill_h * 1.3, W * 0.38, pill_h * 1.3, rng)

def draw_info_panel(img, x0, y0, x1, y1, sku, rng):
    """Back-of-pack: nutrition table lines + barcode (used on the wrap seam)."""
    d = ImageDraw.Draw(img); bg, fg = hexrgb(sku["bg"]), hexrgb(sku["fg"])
    W, H = x1 - x0, y1 - y0
    d.rectangle([x0 + W * 0.08, y0 + H * 0.12, x1 - W * 0.08, y0 + H * 0.62], fill=(255, 255, 255))
    f = font(F_BOLD, H * 0.05); d.text((x0 + W * 0.12, y0 + H * 0.14), "NUTRITION / 100g", font=f, fill=(0, 0, 0))
    f2 = font(F_REG, H * 0.04)
    for i, row in enumerate(("Energy", "Fat", "Carbohydrate", "Sugars", "Protein", "Salt")):
        y = y0 + H * (0.22 + i * 0.062)
        d.text((x0 + W * 0.12, y), row, font=f2, fill=(0, 0, 0))
        d.text((x1 - W * 0.36, y), f"{rng.randint(0, 60)}.{rng.randint(0, 9)} g", font=f2, fill=(0, 0, 0))
        d.line([x0 + W * 0.12, y + H * 0.055, x1 - W * 0.12, y + H * 0.055], fill=(120, 120, 120), width=1)
    barcode(d, x0 + W * 0.2, y0 + H * 0.68, W * 0.6, H * 0.2, rng)
    f = font(F_REG, H * 0.035); text_center(d, x0 + W / 2, y0 + H * 0.93, f"Made in EU • Store in a cool dry place • {sku['brand']}", f, fill=fg)

def label_image(sku, rng):
    form = sku["form"]
    bg = hexrgb(sku["bg"])
    if form in ROUND:
        circ = math.pi * sku["w"]
        lab_h = sku["h"] * (0.42 if form == "bottle" else 0.8 if form == "tub" else 0.92)
        W, H = int(circ * PX_PER_M), int(lab_h * PX_PER_M)
        W = max(384, min(2048, W)); H = max(160, min(1024, H))
        img = Image.new("RGB", (W, H), bg)
        # main panel centred at u=0.5, half-width 0.3 of circumference; info panel wraps the seam
        draw_panel(img, int(W * 0.2), 0, int(W * 0.8), H, sku, rng, main=False)
        info = Image.new("RGB", (int(W * 0.4), H), bg); draw_info_panel(info, 0, 0, info.size[0], H, sku, rng)
        img.paste(info.crop((int(W * 0.2), 0, int(W * 0.4), H)), (0, 0)); img.paste(info.crop((0, 0, int(W * 0.2), H)), (int(W * 0.8), 0))
    else:
        W, H = int(sku["w"] * PX_PER_M), int(sku["h"] * PX_PER_M)
        if form == "flat":   # flat packs are seen from the top: label is w × d
            H = int(sku["d"] * PX_PER_M)
        W = max(192, min(1536, W)); H = max(96, min(1536, H))
        img = Image.new("RGB", (W, H), bg)
        draw_panel(img, 0, 0, W, H, sku, rng, main=True)
    # subtle vertical gradient for print feel
    grad = Image.linear_gradient("L").resize(img.size).point(lambda v: int(255 - v * 0.18))
    img = Image.composite(img, Image.new("RGB", img.size, (0, 0, 0)), grad)
    return img

def price_tag(sku, rng):
    W, H = 640, 320
    img = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 34], fill=(230, 40, 40) if sku["price"] < 1.0 else (30, 60, 140))
    f = font(F_BOLD, 22); d.text((12, 6), CATEGORIES[sku["cat"]].upper(), font=f, fill=(255, 255, 255))
    f = fit_text(d, f"{sku['brand']} {sku['name']}", F_BOLD, W * 0.55, 46); d.text((14, 48), f"{sku['brand']} {sku['name']}", font=f, fill=(0, 0, 0))
    f = font(F_REG, 26); d.text((14, 100), sku["weight"], font=f, fill=(60, 60, 60))
    f = font(F_BOLD, 118); text_center(d, W * 0.72, 150, f"€{sku['price']:.2f}", f, fill=(0, 0, 0))
    per = sku["price"] / max(0.1, rng.uniform(0.3, 2.0))
    f = font(F_REG, 22); d.text((W * 0.5, 232), f"€{per:.2f} / kg", font=f, fill=(80, 80, 80))
    barcode(d, 14, 220, 240, 84, rng)
    return img

def aisle_sign(a):
    W, H = 1600, 640
    img = Image.new("RGB", (W, H), (20, 40, 90)); d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 380, H], fill=(245, 180, 0))
    f = font(F_BOLD, 420); text_center(d, 190, H / 2, str(a["no"]), f, fill=(20, 40, 90))
    lines = a["left"] + a["right"]
    fs = min(96, int(H * 0.8 / max(1, len(lines)) * 0.7))
    f = font(F_BOLD, fs)
    for i, t in enumerate(lines):
        d.text((430, 60 + i * (H - 120) / max(1, len(lines))), t, font=f, fill=(255, 255, 255))
    return img

def store_sign():
    W, H = 2400, 520
    img = Image.new("RGB", (W, H), (200, 30, 40)); d = ImageDraw.Draw(img)
    f = font(F_BOLD, 330); text_center(d, W / 2, H / 2 - 20, "FRESHMART", f, fill=(255, 255, 255))
    f = font(F_REG, 70); text_center(d, W / 2, H - 70, "railway station supermarket • open 06–23", f, fill=(255, 230, 200))
    return img

def text_panel(text, W, H, bg, fg, path=F_BOLD, sub=None):
    img = Image.new("RGB", (W, H), bg); d = ImageDraw.Draw(img)
    f = fit_text(d, text, path, W * 0.9, H * (0.55 if sub else 0.7)); text_center(d, W / 2, H * (0.4 if sub else 0.5), text, f, fill=fg)
    if sub:
        f = fit_text(d, sub, F_REG, W * 0.9, H * 0.22); text_center(d, W / 2, H * 0.78, sub, f, fill=fg)
    return img

def main():
    rng = random.Random(3)
    for c, label in CATEGORIES.items():
        text_panel(label, 1400, 380, (20, 40, 90), (255, 255, 255)).save(os.path.join(SIG, f"cat_{c}.png"), optimize=True)
    for n in range(1, 5):
        text_panel(str(n), 400, 500, (20, 40, 90), (245, 180, 0), sub="CHECKOUT").save(os.path.join(SIG, f"lane_{n}.png"), optimize=True)
    text_panel("EXIT", 1200, 400, (30, 120, 50), (255, 255, 255)).save(os.path.join(SIG, "exit.png"), optimize=True)
    text_panel("ENTRANCE", 1200, 400, (30, 120, 50), (255, 255, 255)).save(os.path.join(SIG, "entrance.png"), optimize=True)
    text_panel("WELCOME TO FRESHMART", 2400, 500, (200, 30, 40), (255, 255, 255), sub="fresh every day • bakery • dairy • produce").save(os.path.join(SIG, "welcome.png"), optimize=True)
    text_panel("PRODUCE", 1400, 380, (40, 120, 50), (255, 255, 255), sub="fresh fruit & vegetables").save(os.path.join(SIG, "produce.png"), optimize=True)
    text_panel("BAKERY", 1400, 380, (140, 80, 30), (255, 240, 210), sub="baked in store daily").save(os.path.join(SIG, "bakery.png"), optimize=True)
    text_panel("DAIRY", 1400, 380, (20, 90, 160), (255, 255, 255), sub="milk • yogurt • cheese").save(os.path.join(SIG, "dairy.png"), optimize=True)
    text_panel("DRINKS", 1400, 380, (20, 90, 160), (255, 255, 255), sub="chilled").save(os.path.join(SIG, "drinks.png"), optimize=True)
    for s in SKUS:
        label_image(s, rng).save(os.path.join(LAB, f"{s['id']}.png"), optimize=True)
        price_tag(s, rng).save(os.path.join(LAB, f"tag_{s['id']}.png"), optimize=True)
    plan = json.load(open(os.path.join(HERE, "planogram.json")))
    for a in plan["aisles"]:
        aisle_sign(a).save(os.path.join(SIG, f"aisle_{a['no']}.png"), optimize=True)
    store_sign().save(os.path.join(SIG, "store_sign.png"), optimize=True)
    print(f"{len(SKUS)} labels + tags -> {LAB}; {len(plan['aisles'])} aisle signs -> {SIG}")

if __name__ == "__main__":
    main()
