"""The phone-first dashboard page (static HTML; all data via ``/status.json``).

Kept as one self-contained document (no build step, no external assets) so it loads
on a phone over the tailnet with zero dependencies. The live frame and run/stage
videos are linked back to the upstream watch page rather than re-served here.
"""

PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Fleet dashboard - mission + store</title>
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
#trigbox{display:none}
button{background:var(--active);color:#fff;border:0;border-radius:8px;padding:9px 14px;font-size:14px}
input{background:#0b0f14;color:var(--fg);border:1px solid var(--line);border-radius:7px;padding:8px;
font-size:14px;width:100%;margin:4px 0}
</style></head><body>
<h1>Fleet dashboard</h1>
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
  <div class=cap>Live frame &amp; videos from the upstream watch page (read-only).</div>
  <ul class=list id=runs></ul>
</div>

<h2>Behaviour tree</h2>
<div class=card id=btcard><span class=muted>Not published by this run.</span></div>

<h2>Store state</h2>
<div class=card id=storecard><span class=muted>Not configured.</span></div>

<h2>Restock requests</h2>
<div class=card>
  <div id=triggers><span class=muted>None.</span></div>
  <div id=trigbox>
    <input id=t_sku placeholder="SKU (e.g. SKU-1234)">
    <input id=t_qty placeholder="quantity" inputmode=numeric value="1">
    <input id=t_shelf placeholder="destination shelf slot (optional)">
    <button onclick="sendTrigger()">Queue restock request</button>
    <div class=cap id=trigmsg></div>
  </div>
</div>

<script>
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
let TRIG_ENABLED=false;
async function tick(){
  let s;
  try{s=await (await fetch('/status.json?'+Date.now())).json();}catch(e){
    document.getElementById('gen').innerHTML='<span class=err>dashboard unreachable</span>';return;}
  const m=s.mission||{},t=m.trace||{},tl=m.timeline||{},up=s.upstream||{};
  document.getElementById('gen').textContent='updated '+(s.generated_at||'')+(up.ok?'':' · upstream pulse unavailable');
  document.getElementById('pulse').innerHTML=
    `<span class="pill ${m.alive?'on':'off'}">${m.alive?'simulator running':'simulator idle'}</span>`+
    `<span class="pill dim">run ${esc(m.run||'-')}</span>`+
    `<span class="pill dim">${m.frames||0} frames</span>`+
    `<span class="pill ${tl.verdict==='complete'?'on':(tl.verdict==='failed'?'off':'dim')}">${esc(tl.verdict||'-')}</span>`;
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
  if(s.media&&s.media.latest_frame) document.getElementById('frame').src=s.media.latest_frame+'?'+Date.now();
  document.getElementById('runs').innerHTML=(m.videos||[]).slice(0,8).map(v=>{
    const base=up.base||''; const name=v.split('/').slice(-2,-1)[0]||v;
    return `<li><a href="${esc(base+'/'+v)}">${esc(name)}</a></li>`;}).join('');

  const bt=s.behaviour_tree;
  document.getElementById('btcard').innerHTML = bt
    ? `<pre class=muted style="white-space:pre-wrap;font-size:12px;margin:0">${esc(JSON.stringify(bt,null,1))}</pre>`
    : '<span class=muted>Not published by this run.</span>';

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
  document.getElementById('triggers').innerHTML = tr.length
    ? `<table><tr><th>when</th><th>sku</th><th>qty</th><th>status</th></tr>`+
      tr.slice(0,10).map(r=>`<tr><td>${esc((r.created_at||'').slice(11,19))}</td>`+
      `<td>${esc((r.request||{}).sku)}</td><td>${(r.request||{}).quantity}</td><td>${esc(r.status)}</td></tr>`).join('')+`</table>`
    : '<span class=muted>None.</span>';
  if(TRIG_ENABLED) document.getElementById('trigbox').style.display='block';
}
async function sendTrigger(){
  const body={sku:document.getElementById('t_sku').value,
    quantity:parseInt(document.getElementById('t_qty').value||'1',10),
    shelf:document.getElementById('t_shelf').value||null, provenance:'dashboard-phone'};
  const msg=document.getElementById('trigmsg');
  try{const r=await fetch('/trigger',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json(); msg.innerHTML = r.ok ? 'queued '+esc(j.id) : '<span class=err>'+esc(j.error||'rejected')+'</span>';
    tick();}catch(e){msg.innerHTML='<span class=err>network error</span>';}
}
async function init(){try{const c=await (await fetch('/config.json')).json();TRIG_ENABLED=!!c.trigger_enabled;}catch(e){}tick();setInterval(tick,3000);}
init();
</script></body></html>"""
