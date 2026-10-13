#!/usr/bin/env python3
"""Create a portable static evidence report; never render a fake simulation video."""
import argparse
import base64
import html
import json
from pathlib import Path


def map_svg(manifest):
    x0,x1,y0,y1=manifest['map_bounds']
    width,height=x1-x0,y1-y0
    parts=[f'<svg role="img" aria-label="Static environment collision map" viewBox="{x0} {-y1} {width} {height}" xmlns="http://www.w3.org/2000/svg">',
           f'<rect x="{x0}" y="{-y1}" width="{width}" height="{height}" fill="#e8eef3"/>']
    for obstacle in manifest.get('static_obstacles',[]):
        lo,hi=obstacle['min'],obstacle['max']
        parts.append(f'<rect x="{lo[0]}" y="{-hi[1]}" width="{hi[0]-lo[0]}" height="{hi[1]-lo[1]}" fill="#758696"/>')
    for name,pose in manifest.get('poses',{}).items():
        x,y,_=pose
        parts.append(f'<circle cx="{x}" cy="{-y}" r=".16" fill="#a32920"/><text x="{x+.25}" y="{-y}" font-size=".48">{html.escape(name)}</text>')
    # Semantic route hints intentionally omitted: they have not passed loaded path validation.
    parts.append('</svg>')
    return ''.join(parts)


def build(manifest, narrative, evidence, pictures=(), videos=()):
    cards=''.join(f'<article><b>{html.escape(str(k))}</b><p>{html.escape(str(v))}</p></article>'
                  for k,v in evidence.items())
    gallery=''.join('<figure><img alt="'+html.escape(label,quote=True)+
        '" src="data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()+
        '"><figcaption>'+html.escape(label)+'</figcaption></figure>' for label,path in pictures)
    recordings=''.join('<figure><video controls playsinline style="width:100%" src="data:video/mp4;base64,'+
        base64.b64encode(Path(path).read_bytes()).decode()+'"></video><figcaption>'+
        html.escape(label)+'</figcaption></figure>' for label,path in videos)
    return '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Mobile restocking evidence</title>
<style>body{margin:0;background:#f6f4ee;color:#1b202b;font:16px/1.55 system-ui}main{max-width:1100px;margin:auto;padding:36px}h1{font-size:clamp(38px,7vw,64px);line-height:1.05;color:#14285a}.status{padding:14px;background:#fff0cc;border-left:5px solid #f5b400}.cards{display:flex;flex-wrap:wrap;gap:14px;margin:24px 0}article{padding:16px;background:white;border-radius:4px;border-top:4px solid #14285a;flex:1;min-width:180px}figure{margin:20px 0;background:white;padding:20px;border-radius:4px}svg{width:100%;max-height:600px}img{width:100%;height:auto}pre{white-space:pre-wrap;font:15px/1.6 system-ui;background:white;padding:24px}figcaption{color:#4b5262}details{margin-top:24px}</style>
<main><h1>Mobile restocking simulation</h1>
<p class="status"><strong>INCOMPLETE — autonomous restocking cycle not yet demonstrated.</strong><br>Native sensor/render checks, CPU tests and authored layouts are reported separately below.</p>
<div class="cards">'''+cards+'''</div>'''+recordings+gallery+'''<figure>'''+map_svg(manifest)+'''
<figcaption>Static, authored environment collision map. Markers are scenario setup poses, not sensor detections or a driven trajectory. No loaded route is asserted.</figcaption></figure>
<details open><summary>Validation, limitations and remaining delivery gates</summary><pre>'''+html.escape(narrative)+'''</pre></details></main></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--narrative',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--image',action='append',default=[],help='caption=PNG path; embedded in output')
    p.add_argument('--video',action='append',default=[],help='honest caption=recorded MP4 path; embedded in output')
    a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(build(json.loads(a.manifest.read_text()),a.narrative.read_text(),
                              json.loads(a.evidence.read_text()),[v.split('=',1) for v in a.image],
                              [v.split('=',1) for v in a.video]))


if __name__=='__main__':
    main()
