"""Live, phone-friendly visualisation of the mission tree and event handling.

Writes three files into a run directory (``firstmate`` links them from the watch page; this lane
never edits the watch page itself):

* ``bt_snapshot.json`` -- the current tree (nodes with name / status / feedback, nested), the
  blackboard (active events, control flags, mission state/stage) and the event log. Overwritten
  every tick (atomic rename), so a poller always reads a consistent frame.
* ``bt_events.jsonl``  -- append-only event/action timeline (one JSON object per line), the durable
  record of what happened and how it was handled.
* ``bt_view.html``     -- a self-contained page (no external assets) that polls ``bt_snapshot.json``
  and renders the tree with node states coloured, the control flags, the active events and the
  recent timeline, auto-refreshing. Written once at setup.

There is no server here: the files sit in the run directory and are served by whatever already
serves the run (the watch page's static file server). A desktop viewer alternative
(``py_trees_ros_viewer`` / ``rqt_py_trees`` over the optional snapshot stream) is documented in
``docs/restock_bt.md``; this HTML is the primary phone view the brief asks for.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import py_trees
from py_trees.common import Status

from .context import (CLOCK, CTRL_ABORT, CTRL_PAUSE, CTRL_REPLAN, CTRL_SLOW,
                      EVENT_ACTIVE, EVENT_LAST, EVENT_LOG)

_STATUS_NAME = {Status.SUCCESS: 'success', Status.RUNNING: 'running',
                Status.FAILURE: 'failure', Status.INVALID: 'invalid'}


def serialise_node(node):
    """Recursively serialise a py_trees behaviour/composite into a plain dict."""
    return {
        'name': node.name,
        'status': _STATUS_NAME.get(node.status, str(node.status)),
        'feedback': node.feedback_message or '',
        'kind': type(node).__name__,
        'children': [serialise_node(c) for c in getattr(node, 'children', [])],
    }


def snapshot_dict(ctx, root, *, tick=None):
    """Build the full snapshot dict (tree + blackboard + log) for one tick."""
    bb = ctx.bb
    return {
        'tick': tick,
        'clock': bb.get(CLOCK),
        'mission': {'state': ctx.mission.state, 'reason': ctx.mission.reason,
                    'degraded': list(ctx.mission.degraded)},
        'control': {'pause': bool(bb.get(CTRL_PAUSE)), 'slow': bool(bb.get(CTRL_SLOW)),
                    'replan': bool(bb.get(CTRL_REPLAN)), 'abort': bb.get(CTRL_ABORT)},
        'active_events': bb.get(EVENT_ACTIVE),
        'last_event': bb.get(EVENT_LAST),
        'log': bb.get(EVENT_LOG)[-40:],
        'tree': serialise_node(root),
    }


class SnapshotWriter:
    """Writes ``bt_snapshot.json`` / ``bt_events.jsonl`` per tick and ``bt_view.html`` once."""
    def __init__(self, out_dir, *, title='restock mission', min_interval_s=None):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.snapshot_path = self.dir / 'bt_snapshot.json'
        self.events_path = self.dir / 'bt_events.jsonl'
        self.html_path = self.dir / 'bt_view.html'
        self._log_written = 0
        # sim_speed bt_snapshot_hz (sim-video-speedup-v1): rewrite at most every ``min_interval_s`` of
        # SIM time (ctx.now); a terminal mission is always written. None (default) => every tick.
        self.min_interval_s = min_interval_s
        self._last_written = None
        self.html_path.write_text(_HTML.replace('__TITLE__', title))

    def write(self, ctx, root, *, tick=None):
        if self.min_interval_s is not None and self._last_written is not None:
            now = float(getattr(ctx, 'now', 0.0) or 0.0)
            terminal = bool(getattr(getattr(ctx, 'mission', None), 'terminal', False))
            if not terminal and now - self._last_written < self.min_interval_s:
                return None
        self._last_written = float(getattr(ctx, 'now', 0.0) or 0.0)
        snap = snapshot_dict(ctx, root, tick=tick)
        tmp = self.snapshot_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(snap, allow_nan=False))
        os.replace(tmp, self.snapshot_path)        # atomic: poller never reads a half file
        log = ctx.bb.get(EVENT_LOG)
        if len(log) > self._log_written:
            with self.events_path.open('a') as fh:
                for entry in log[self._log_written:]:
                    fh.write(json.dumps(entry, allow_nan=False) + '\n')
            self._log_written = len(log)
        return snap


# Self-contained page: polls bt_snapshot.json, renders the tree + flags + events. No external
# assets, so it works behind the watch page's static server and on a phone.
_HTML = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>__TITLE__ -- behaviour tree</title>
<style>
 body{font:14px/1.4 system-ui,sans-serif;margin:0;background:#0d1117;color:#c9d1d9}
 header{padding:10px 14px;background:#161b22;position:sticky;top:0;border-bottom:1px solid #30363d}
 h1{font-size:15px;margin:0 0 4px} .sub{color:#8b949e;font-size:12px}
 .wrap{padding:12px 14px;max-width:760px;margin:0 auto}
 .flags{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
 .chip{padding:2px 8px;border-radius:10px;font-size:12px;background:#21262d;border:1px solid #30363d}
 .chip.on{background:#9e6a03;border-color:#bb8009;color:#fff}
 .chip.abort{background:#b62324;border-color:#da3633;color:#fff}
 .node{margin:2px 0 2px 0;padding-left:14px;border-left:2px solid #30363d}
 .row{display:flex;align-items:baseline;gap:8px;padding:2px 0}
 .dot{width:9px;height:9px;border-radius:50%;flex:0 0 auto;margin-top:4px}
 .s-success{background:#2ea043}.s-running{background:#d29922}.s-failure{background:#da3633}
 .s-invalid{background:#484f58}
 .nm{font-weight:600}.fb{color:#8b949e;font-size:12px}
 .ev{font-family:ui-monospace,monospace;font-size:12px;padding:2px 0;border-bottom:1px solid #21262d}
 .ev .a{color:#58a6ff}.ev .t{color:#8b949e}
 h2{font-size:13px;color:#8b949e;margin:16px 0 4px;text-transform:uppercase;letter-spacing:.04em}
 .mstate{font-weight:600} .err{color:#f85149}
</style></head><body>
<header><h1>__TITLE__ &mdash; behaviour tree</h1>
 <div class=sub id=hdr>loading&hellip;</div></header>
<div class=wrap>
 <div class=flags id=flags></div>
 <h2>Tree</h2><div id=tree></div>
 <h2>Active events</h2><div id=active class=sub>none</div>
 <h2>Timeline</h2><div id=log></div>
</div>
<script>
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function renderNode(n){
 var kids=(n.children||[]).map(renderNode).join('');
 return '<div class=node><div class=row><span class="dot s-'+esc(n.status)+'"></span>'+
  '<span class=nm>'+esc(n.name)+'</span>'+
  (n.feedback?'<span class=fb>'+esc(n.feedback)+'</span>':'')+'</div>'+kids+'</div>';
}
function flag(label,on,cls){return '<span class="chip '+(on?(cls||'on'):'')+'">'+label+(on&&on!==true?': '+esc(on):'')+'</span>';}
async function tick(){
 try{
  var r=await fetch('bt_snapshot.json?_='+Date.now(),{cache:'no-store'});
  var d=await r.json();
  var m=d.mission||{};
  document.getElementById('hdr').innerHTML='mission <span class=mstate>'+esc(m.state)+'</span>'+
   (m.reason?' <span class=err>'+esc(m.reason)+'</span>':'')+
   ' &middot; clock '+esc((d.clock==null?'?':d.clock))+'s &middot; tick '+esc(d.tick==null?'?':d.tick);
  var c=d.control||{};
  document.getElementById('flags').innerHTML=
   flag('pause',c.pause)+flag('slow',c.slow)+flag('replan',c.replan)+
   (c.abort?flag('ABORT',c.abort,'abort'):'')+
   ((m.degraded&&m.degraded.length)?flag('degraded',m.degraded.join(','),'abort'):'');
  document.getElementById('tree').innerHTML=renderNode(d.tree);
  var ae=d.active_events||{};var keys=Object.keys(ae);
  document.getElementById('active').innerHTML=keys.length?
   keys.map(k=>'<div class=ev><span class=a>'+esc(k)+'</span> '+esc(JSON.stringify(ae[k]))+'</div>').join(''):'none';
  var log=(d.log||[]).slice().reverse();
  document.getElementById('log').innerHTML=log.map(e=>
   '<div class=ev><span class=t>'+esc(e.t)+'s</span> <span class=a>'+esc(e.action||'')+'</span> '+
   esc(e.type)+' '+esc(JSON.stringify(e.data||{}))+'</div>').join('');
 }catch(e){document.getElementById('hdr').textContent='waiting for snapshot ('+e+')';}
}
tick();setInterval(tick,700);
</script></body></html>"""
