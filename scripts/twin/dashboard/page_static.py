"""The static run-directory panel page (polls ``./fleet.json``; no server needed).

A sibling of :mod:`twin.dashboard.page` for the zero-process variant: it fetches its
data from ``fleet.json`` *relative to itself* (so it works under any static mount,
e.g. ``python3 -m http.server`` over a run directory) rather than from a live
``/status.json`` endpoint, and it is strictly read-only — no trigger form, since a
static mount cannot accept a POST. Media links are whatever ``snapshot.build_run_snapshot``
wrote: run-relative files when served over the run directory, or absolute watch-page
URLs when a media base was given.
"""

STATIC_PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Fleet panel - mission + store</title>
<style>
:root{--bg:#0b0f14;--panel:#141b24;--line:#223040;--muted:#9fb3c8;--fg:#e6edf3;
--ok:#1f6f3f;--warn:#8a6d1f;--bad:#7a2b2b;--active:#1f5f8a;--pending:#2a3340}
*{box-sizing:border-box}
body{font-family:-apple-system,system-ui,sans-serif;background:var(--bg);color:var(--fg);
margin:0;padding:12px;max-width:760px;margin:0 auto}
h1{font-size:19px;margin:4px 0}h2{font-size:14px;margin:16px 0 8px;color:var(--muted);
text-transform:uppercase;letter-spacing:.04em}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px;margin:8px 0}
.pill{display:inline-block;padding:2px 9px;border-radius:11px;font-size:12px;margin:0 6px 2px 0}
.on{background:var(--ok)}.off{background:var(--bad)}.dim{background:var(--pending);color:var(--muted)}
.warn{background:var(--warn)}
.cap{font-size:12px;color:var(--muted);margin:6px 0}
img{width:100%;border-radius:8px;background:#000;display:block}
a{color:#7cc4ff;text-decoration:none}
.steps{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0}
.step{flex:1 1 auto;min-width:72px;text-align:center;padding:7px 4px;border-radius:8px;
font-size:12px;background:var(--pending);color:var(--muted)}
.step.done{background:var(--ok);color:#fff}.step.active{background:var(--active);color:#fff}
.step.failed{background:var(--bad);color:#fff}.step.pending{background:var(--pending)}
.kv{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:13px;color:var(--muted);margin-top:6px}
.kv b{color:var(--fg)}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:4px 6px;
border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:500}
.sev-warn{color:#e3b341}.sev-critical,.sev-crit{color:#ff7b72}.sev-info{color:var(--muted)}
.list li{margin:5px 0;font-size:13px}
.muted{color:var(--muted)}.err{color:#ff7b72}
.state-oos,.state-empty,.state-out_of_stock{color:#ff7b72}.state-low,.state-limited{color:#e3b341}
.bt{font-size:13px;margin:4px 0}
.btn{padding:3px 6px;border-radius:5px;margin:2px 0;display:flex;align-items:center;gap:7px}
.btn .dot{width:8px;height:8px;border-radius:50%;flex:0 0 auto;background:var(--muted)}
.bt-success .dot{background:#2ea043}.bt-running .dot{background:#58a6ff}
.bt-failure .dot{background:#f85149}.bt-invalid .dot{background:var(--pending)}
.bt-running{color:#fff}.bt-failure{color:#ffb3ad}
</style></head><body>
<h1>Fleet panel <span class=muted style="font-size:13px">(static)</span></h1>
<div class=cap id=gen>loading...</div>

<div class=card>
  <div id=pulse></div>
  <div class=kv id=trace></div>
</div>

<h2>Mission pipeline</h2>
<div class=card>
  <div class=steps id=steps></div>
  <div class=cap id=verdict></div>
</div>

<div class=card>
  <img id=frame alt="newest rendered frame">
  <div class=cap id=mediacap>Newest rendered frame from this run.</div>
</div>

<h2>Behaviour tree</h2>
<div class=card id=btcard><span class=muted>Not published by this run.</span></div>

<h2>Store state</h2>
<div class=card id=storecard><span class=muted>Not configured.</span></div>

<h2>Restock requests</h2>
<div class=card id=trigcard><span class=muted>None.</span></div>

<script>
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function tick(){
  let s;
  try{s=await (await fetch('fleet.json?'+Date.now())).json();}catch(e){
    document.getElementById('gen').innerHTML='<span class=err>fleet.json unreachable</span>';return;}
  const m=s.mission||{},t=m.trace||{},tl=m.timeline||{};
  const age=(m.run_age_s==null)?'':' · '+m.run_age_s+'s old';
  document.getElementById('gen').textContent='snapshot '+(s.generated_at||'')+age;
  document.getElementById('pulse').innerHTML=
    `<span class="pill dim">run ${esc(m.run||'-')}</span>`+
    `<span class="pill dim">${m.frames||0} frames</span>`+
    `<span class="pill ${tl.verdict==='complete'?'on':(tl.verdict==='failed'?'off':'dim')}">${esc(tl.verdict||'-')}</span>`+
    (s.approval_required?`<span class="pill warn">approval required</span>`:'');
  document.getElementById('trace').innerHTML=
    `stage <b>${esc(tl.active||t.action||'-')}</b>`+
    `<span>tick <b>${t.tick??'-'}</b></span><span>sim <b>${t.t??'-'}</b>s</span>`+
    `<span>coupled <b>${t.coupled??'-'}</b></span><span>parked <b>${t.parked??'-'}</b></span>`+
    `<span>x <b>${t.x??'-'}</b> y <b>${t.y??'-'}</b></span>`;
  document.getElementById('steps').innerHTML=(tl.steps||[]).map(st=>
    `<div class="step ${st.status}">${esc(st.stage)}</div>`).join('');
  document.getElementById('verdict').innerHTML= tl.verdict==='failed'
    ? `<span class=err>failed: ${esc(tl.reason||'?')}</span>`
    : (tl.verdict==='complete'
        ? `complete · physics_verified ${esc(tl.physics_verified)}`
        : `in progress · ${tl.event_count||0} events recorded`);
  const med=s.media||{};
  if(med.latest_frame){document.getElementById('frame').src=med.latest_frame+'?'+Date.now();
    document.getElementById('frame').style.display='block';}
  else{document.getElementById('frame').style.display='none';}
  const links=[];
  if(med.walkthrough) links.push(`<a href="${esc(med.walkthrough)}">walkthrough.mp4</a>`);
  if(med.report) links.push(`<a href="${esc(med.report)}">report</a>`);
  document.getElementById('mediacap').innerHTML='Newest rendered frame from this run.'+
    (links.length?' · '+links.join(' · '):'');

  const bt=s.behaviour_tree;
  if(bt){
    const c=bt.control||{}, me=bt.mission||{};
    let h=`<div class=kv>`+
      `<span>tick <b>${bt.tick??'-'}</b></span><span>clock <b>${bt.clock??'-'}</b>s</span>`+
      `<span>mission <b>${esc(me.state||'-')}</b></span></div>`;
    const flags=[['pause',c.pause],['slow',c.slow],['replan',c.replan],['abort',c.abort]]
      .filter(([k,v])=>v).map(([k,v])=>`<span class="pill warn">${esc(k)}${v===true?'':': '+esc(v)}</span>`);
    h+=`<div style="margin:4px 0">`+(flags.length?flags.join(''):`<span class="pill on">nominal</span>`)+`</div>`;
    const le=bt.last_event;
    if(le) h+=`<div class=cap>last event <b>${esc(le.type)}</b> @ ${esc(le.t)}s</div>`;
    h+=`<div class=bt>`+(bt.nodes||[]).map(n=>{
      const st=String(n.status||'').toLowerCase();
      return `<div class="btn bt-${esc(st)}" style="padding-left:${6+(n.depth||0)*16}px">`+
        `<span class=dot></span>${esc(n.name)}`+
        (n.feedback?` <span class=muted>- ${esc(n.feedback)}</span>`:'')+`</div>`;}).join('')+`</div>`;
    const lg=bt.log||[];
    if(lg.length) h+=`<table style="margin-top:8px"><tr><th>t</th><th>event</th><th>action</th></tr>`+
      lg.slice(-8).map(e=>`<tr><td>${esc(e.t)}</td><td>${esc(e.type)}</td><td>${esc(e.action)}</td></tr>`).join('')+`</table>`;
    if(bt.view_url) h+=`<div class=cap><a href="${esc(bt.view_url)}">open full tree view (bt_view.html) &rarr;</a></div>`;
    document.getElementById('btcard').innerHTML=h;
  }

  const st=s.store;
  if(st){
    const su=st.summary||{};
    let h=`<div class=kv>`+
      `<span>availability <b>${st.on_shelf_availability_pct??'-'}%</b></span>`+
      `<span>needs restock <b>${su.needs_restock??0}</b></span>`+
      `<span>out of stock <b>${su.out_of_stock??0}</b></span>`+
      `<span>alerts <b>${su.active_alerts??0}</b></span></div>`;
    const slots=(st.slots||[]).filter(x=>['oos','low','empty','out_of_stock','limited'].includes(String(x.state).toLowerCase()));
    if(slots.length) h+=`<table><tr><th>slot</th><th>sku</th><th>fill</th><th>state</th></tr>`+
      slots.map(x=>`<tr><td>${esc(x.slot)}</td><td>${esc(x.sku)}</td><td>${x.fill_pct??'-'}%</td>`+
      `<td class="state-${esc(String(x.state).toLowerCase())}">${esc(x.state)}</td></tr>`).join('')+`</table>`;
    const al=(st.alerts||[]).filter(a=>(a.count||0)>0);
    if(al.length) h+=`<table style="margin-top:8px"><tr><th>alert</th><th>sev</th><th>count</th><th>where</th></tr>`+
      al.map(a=>`<tr><td>${esc(a.type)}</td><td class="sev-${esc(String(a.severity).toLowerCase())}">${esc(a.severity)}</td>`+
      `<td>${a.count}</td><td>${esc(a.location||'-')}</td></tr>`).join('')+`</table>`;
    h+=`<div class=cap>source ${esc(st.source||'?')} · ${esc(st.generated_at||'')}</div>`;
    document.getElementById('storecard').innerHTML=h;
  }

  const tr=s.triggers||[];
  document.getElementById('trigcard').innerHTML = tr.length
    ? `<table><tr><th>when</th><th>sku</th><th>qty</th><th>status</th></tr>`+
      tr.slice(0,10).map(r=>`<tr><td>${esc((r.created_at||'').slice(11,19))}</td>`+
      `<td>${esc((r.request||{}).sku)}</td><td>${(r.request||{}).quantity}</td><td>${esc(r.status)}</td></tr>`).join('')+`</table>`
    : '<span class=muted>None.</span>';
}
tick();setInterval(tick,3000);
</script></body></html>"""
