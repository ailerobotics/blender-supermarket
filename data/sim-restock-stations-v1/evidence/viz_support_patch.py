import sys, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

def load(npz):
    d = np.load(npz)
    return d['rgb'], d['depth'], d['support_uv'], d['witness_uv']

def draw(npz, out_prefix):
    rgb, depth, support_uv, witness_uv = load(npz)
    rgb = np.ascontiguousarray(rgb[:,:,:3].astype(np.uint8))
    im = Image.fromarray(rgb); dr = ImageDraw.Draw(im)
    def poly(uv, color):
        pts = [tuple(map(float,p)) for p in np.asarray(uv)]
        dr.line(pts+[pts[0]], fill=color, width=6)
    poly(support_uv, (0,255,0))     # green = support
    poly(witness_uv, (255,0,0))     # red = witness
    # downscale for cheap viewing
    im.thumbnail((900,900))
    im.save(out_prefix+'-rgb.png')
    # depth: within witness polygon, print stats
    dd = np.asarray(depth,float)
    finite = dd[np.isfinite(dd)]
    print(f'{out_prefix}: depth shape {dd.shape} min {finite.min():.3f} max {finite.max():.3f}')
    # depth visualization
    dv = np.clip((dd - np.nanpercentile(finite,2))/(np.nanpercentile(finite,98)-np.nanpercentile(finite,2)+1e-9),0,1)
    dvi = Image.fromarray((dv*255).astype(np.uint8))
    dvi.thumbnail((900,900)); dvi.save(out_prefix+'-depth.png')
    # witness & support bbox depth spread (camera-space, crude)
    for name, uv in (('support',support_uv),('witness',witness_uv)):
        u = np.asarray(uv); umin,vmin = u.min(0); umax,vmax = u.max(0)
        sub = dd[int(vmin):int(vmax)+1, int(umin):int(umax)+1]
        s = sub[np.isfinite(sub)]
        if s.size:
            print(f'  {name} bbox px=({int(umin)},{int(vmin)})-({int(umax)},{int(vmax)}) '
                  f'depth[min={s.min():.3f} max={s.max():.3f} ptp={s.max()-s.min():.3f} p50={np.median(s):.3f}]')

for npz, pfx in [(sys.argv[1], sys.argv[2])]:
    draw(npz, pfx)
