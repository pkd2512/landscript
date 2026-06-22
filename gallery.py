#!/usr/bin/env python3
"""Landscript gallery — browse + curate candidate tiles.

Endpoints:
    GET  /                        → HTML gallery
    GET  /api/candidates          → JSON list
    GET  /api/candidates/<id>/similar?k=24
                                  → list of nearest candidates by descriptor
    POST /api/candidates/<id>/status   body: {"status": "accepted"|"rejected"|"pending"}
    POST /api/candidates/<id>/letter   body: {"letter": "A"|...|null}
    POST /api/candidates/<id>/delete
    GET  /data/...                → static files (the tile PNGs)

Keyboard in detail view:
    A–Z      assign that letter
    Space    toggle accept
    X        reject
    Del      delete
    Escape   close detail
"""

import argparse
import json
import socket
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from landscript.metadata import CandidateStore

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CAND_DIR = DATA_DIR / "candidates"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def build_html(region: str, store: CandidateStore) -> str:
    items = store.all(limit=100000)
    accepted = sum(1 for c in items if c.get("status") == "accepted")
    rejected = sum(1 for c in items if c.get("status") == "rejected")
    pending = len(items) - accepted - rejected
    with_letter = sum(1 for c in items if c.get("letter"))

    # Strip the descriptor vector from the in-page JSON dump to keep the
    # HTML small; similarity lookups go through the API instead.
    light = []
    for c in items:
        d = dict(c)
        d.pop("descriptor", None)
        light.append(d)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Landscript — {region}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0a0a0a; color: #e0e0e0; padding: 20px; padding-bottom: 80px; }}
h1 {{ font-size: 1.4rem; font-weight: 300; margin-bottom: 4px; letter-spacing: 0.08em; }}
h1 span {{ color: #666; }}
.sub {{ color: #666; font-size: 0.8rem; margin-bottom: 14px; }}
.stats {{ display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }}
.stat {{ background: #141414; border-radius: 6px; padding: 8px 14px; }}
.stat .num {{ font-size: 1.15rem; font-weight: 600; color: #fff; }}
.stat .label {{ font-size: 0.65rem; color: #666; text-transform: uppercase;
               letter-spacing: 0.04em; }}

.toolbar {{ position: sticky; top: 0; background: #0a0a0a; padding: 10px 0;
            border-bottom: 1px solid #1a1a1a; margin-bottom: 14px; z-index: 50;
            display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
.toolbar label {{ font-size: 0.72rem; color: #888; display: flex;
                 align-items: center; gap: 4px; }}
.toolbar select, .toolbar input[type=number] {{ background: #1a1a1a;
    color: #e0e0e0; border: 1px solid #2a2a2a; border-radius: 5px;
    padding: 5px 8px; font-size: 0.78rem; }}
.toolbar input[type=number] {{ width: 70px; }}

.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 8px; }}
.card {{ background: #141414; border-radius: 6px; overflow: hidden;
         cursor: pointer; border: 2px solid transparent; transition: all 0.15s;
         position: relative; }}
.card:hover {{ border-color: #444; transform: translateY(-2px); }}
.card.accepted {{ border-color: #22c55e; }}
.card.rejected {{ opacity: 0.35; border-color: #ef4444; }}
.card.selected {{ border-color: #3b82f6; box-shadow: 0 0 0 2px #3b82f6; }}
.card img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block;
            background: #0f0f0f; }}
.card .info {{ display: flex; justify-content: space-between;
              padding: 4px 8px; font-size: 0.72rem; }}
.card .letter-tag {{ position: absolute; top: 4px; left: 4px;
    background: rgba(0,0,0,0.7); color: #fff; padding: 2px 8px;
    border-radius: 4px; font-weight: 700; font-size: 0.85rem;
    letter-spacing: 0.05em; }}
.card .serial {{ position: absolute; bottom: 28px; left: 4px;
    background: rgba(0,0,0,0.65); color: #aaa; padding: 1px 6px;
    border-radius: 3px; font-size: 0.65rem; font-family: 'SF Mono', Menlo, monospace; }}
.card .badge {{ position: absolute; top: 4px; right: 4px;
    font-size: 0.6rem; padding: 2px 6px; border-radius: 4px;
    font-weight: 600; }}
.card.accepted .badge {{ background: #22c55e; color: #000; }}
.card.rejected .badge {{ background: #ef4444; color: #fff; }}
.card.pending .badge {{ display: none; }}
.card .interest {{ color: #888; }}

#detail {{ display: none; position: fixed; top: 0; right: 0; width: 480px;
          height: 100%; background: #111; border-left: 1px solid #222;
          padding: 22px; overflow-y: auto; z-index: 100; }}
#detail.open {{ display: block; }}
#detail img {{ width: 100%; border-radius: 8px; background: #0a0a0a;
              margin-bottom: 14px; }}
#detail h3 {{ font-size: 1.6rem; margin-bottom: 6px; }}
#detail .meta {{ font-size: 0.75rem; color: #888; line-height: 1.6; }}
#detail .meta div {{ word-break: break-word; }}
#detail .meta span {{ color: #ccc; margin-right: 4px; }}
#detail .map-link {{ display: inline-block; margin-top: 6px; padding: 6px 12px;
    background: #1f3a5f; color: #8ec5ff; border: 1px solid #2c5a8f;
    border-radius: 4px; text-decoration: none; font-size: 0.78rem; }}
#detail .map-link:hover {{ background: #2c5a8f; color: #fff; }}
#close {{ position: absolute; top: 10px; right: 14px; background: none;
          border: none; color: #666; font-size: 1.4rem; cursor: pointer; }}
#close:hover {{ color: #fff; }}

.actions {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
           margin-top: 14px; }}
.actions button {{ padding: 8px; border: 1px solid #333; border-radius: 6px;
                  background: #1a1a1a; color: #e0e0e0; cursor: pointer;
                  font-size: 0.8rem; }}
.actions button:hover {{ background: #222; }}
.actions .accept {{ border-color: #22c55e; color: #22c55e; }}
.actions .reject {{ border-color: #ef4444; color: #ef4444; }}
.actions .delete {{ border-color: #f59e0b; color: #f59e0b; }}
.actions .similar {{ border-color: #3b82f6; color: #3b82f6;
    grid-column: span 2; }}

#letter-row {{ display: grid;
    grid-template-columns: repeat(13, 1fr); gap: 4px; margin-top: 12px;
    font-family: 'SF Mono', Menlo, monospace; }}
#letter-row button {{ padding: 6px 0; background: #1a1a1a;
    border: 1px solid #2a2a2a; border-radius: 4px; color: #aaa;
    cursor: pointer; font-size: 0.85rem; font-weight: 700; }}
#letter-row button.on {{ background: #2a4a2a; border-color: #4ade80;
    color: #4ade80; }}
#letter-row button:hover {{ background: #222; }}
#clear-letter {{ grid-column: span 13; margin-top: 2px; padding: 5px 0;
    background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 4px;
    color: #777; cursor: pointer; font-size: 0.75rem; }}

#backdrop {{ display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.5); z-index: 99; }}
#backdrop.open {{ display: block; }}

#status-msg {{ position: fixed; bottom: 16px; left: 50%;
    transform: translateX(-50%); background: #1f1f1f; color: #fff;
    padding: 8px 16px; border-radius: 6px; font-size: 0.8rem; opacity: 0;
    transition: opacity 0.2s; pointer-events: none; border: 1px solid #333; }}
#status-msg.show {{ opacity: 1; }}

@media (max-width: 768px) {{ #detail {{ width: 100%; }} }}
</style>
</head>
<body>

<h1>Landscript <span>— {region}</span></h1>
<div class="sub">{len(items)} candidates · click any to label it manually (A–Z keys)</div>

<div class="stats">
    <div class="stat"><div class="num" id="stat-total">{len(items)}</div><div class="label">Total</div></div>
    <div class="stat"><div class="num" id="stat-accepted">{accepted}</div><div class="label">Accepted</div></div>
    <div class="stat"><div class="num" id="stat-rejected">{rejected}</div><div class="label">Rejected</div></div>
    <div class="stat"><div class="num" id="stat-pending">{pending}</div><div class="label">Pending</div></div>
    <div class="stat"><div class="num" id="stat-letter">{with_letter}</div><div class="label">Labelled</div></div>
</div>

<div class="toolbar">
    <label>Sort
        <select id="f-sort">
            <option value="interest" selected>interest score</option>
            <option value="recent">recent</option>
            <option value="similar">similarity (pick one)</option>
        </select>
    </label>
    <label>Show
        <select id="f-status">
            <option value="all">all</option>
            <option value="pending" selected>pending</option>
            <option value="accepted">accepted</option>
            <option value="rejected">rejected</option>
            <option value="not_rejected">accepted + pending</option>
            <option value="labelled">labelled</option>
        </select>
    </label>
    <label>Limit
        <input type="number" id="f-limit" value="0" min="0" step="100">
        <span style="color:#555">0 = all</span>
    </label>
    <label>Min interest
        <input type="number" id="f-min" value="0.0" min="0" max="1" step="0.05">
    </label>
</div>

<div id="gallery"></div>

<div id="backdrop" onclick="closeDetail()"></div>
<div id="detail">
    <button id="close" onclick="closeDetail()">✕</button>
    <img id="dimg" src="">
    <h3 id="dletter">—</h3>
    <div class="meta" id="dmeta"></div>
    <div id="letter-row"></div>
    <div class="actions">
        <button class="accept" onclick="setStatus('accepted')">✓ Accept</button>
        <button class="reject" onclick="setStatus('rejected')">✗ Reject</button>
        <button onclick="setStatus('pending')">↺ Pending</button>
        <button class="delete" onclick="del()">🗑 Delete</button>
        <button class="similar" onclick="showSimilar()">⌕ Find similar shapes</button>
    </div>
</div>

<div id="status-msg"></div>

<script>
let ITEMS = {json.dumps(light)};
let selected = null;
let similarMode = null;     // when set, gallery shows IDs in this order
let similarOrder = [];

const fSort = document.getElementById('f-sort');
const fStatus = document.getElementById('f-status');
const fLimit = document.getElementById('f-limit');
const fMin = document.getElementById('f-min');
[fSort, fStatus, fLimit, fMin].forEach(el =>
    el.addEventListener('input', () => {{ similarMode = null; render(); }}));

function statusOf(c) {{ return c.status || 'pending'; }}

function flash(msg) {{
    const el = document.getElementById('status-msg');
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.remove('show'), 1300);
}}

function buildLetterRow() {{
    const row = document.getElementById('letter-row');
    row.innerHTML = '';
    for (const L of 'ABCDEFGHIJKLMNOPQRSTUVWXYZ') {{
        const b = document.createElement('button');
        b.textContent = L;
        b.onclick = () => setLetter(L);
        row.appendChild(b);
    }}
    const clr = document.createElement('button');
    clr.id = 'clear-letter';
    clr.textContent = 'clear letter';
    clr.onclick = () => setLetter(null);
    row.appendChild(clr);
}}

function applyFilters() {{
    const status = fStatus.value;
    const rawLimit = parseInt(fLimit.value, 10);
    const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? rawLimit : Infinity;
    const minInt = parseFloat(fMin.value);
    let list = ITEMS.filter(c => {{
        const s = statusOf(c);
        if (status === 'labelled' && !c.letter) return false;
        if (status !== 'all' && status !== 'labelled') {{
            if (status === 'not_rejected' && s === 'rejected') return false;
            else if (status !== 'not_rejected' && s !== status) return false;
        }}
        if ((c.interest ?? 0) < minInt) return false;
        return true;
    }});

    if (similarMode) {{
        const idx = new Map();
        similarOrder.forEach((id, i) => idx.set(id, i));
        list = list.filter(c => idx.has(c.id));
        list.sort((a, b) => idx.get(a.id) - idx.get(b.id));
    }} else if (fSort.value === 'recent') {{
        list.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
    }} else {{
        list.sort((a, b) => (b.interest ?? 0) - (a.interest ?? 0));
    }}
    return limit === Infinity ? list : list.slice(0, limit);
}}

function render() {{
    const list = applyFilters();
    const root = document.getElementById('gallery');
    if (list.length === 0) {{
        root.innerHTML = '<div style="color:#666;padding:40px;text-align:center">No candidates match the current filters.</div>';
        refreshStats();
        return;
    }}
    let html = `<div class="sub" style="margin-bottom:8px">Showing ${{list.length}} of ${{ITEMS.length}} candidates</div><div class="grid">`;
    list.forEach((c, idx) => {{
        const s = statusOf(c);
        const img = `data/glyphs/{region}/${{c.id}}.png`;
        const letterTag = c.letter ? `<span class="letter-tag">${{c.letter}}</span>` : '';
        const serial = `#${{idx + 1}}`;
        html += `
        <div class="card ${{s}}" onclick="select(this, '${{c.id}}')">
            ${{letterTag}}
            <span class="badge">${{s}}</span>
            <span class="serial">${{serial}}</span>
            <img src="${{img}}" loading="lazy" onerror="this.parentElement.style.display='none'">
            <div class="info">
                <span class="interest">int ${{(c.interest ?? 0).toFixed(3)}}</span>
                <span class="region">${{c.region || ''}}</span>
            </div>
        </div>`;
    }});
    html += '</div>';
    root.innerHTML = html;
    refreshStats();
}}

function refreshStats() {{
    document.getElementById('stat-total').textContent = ITEMS.length;
    document.getElementById('stat-accepted').textContent =
        ITEMS.filter(c => statusOf(c) === 'accepted').length;
    document.getElementById('stat-rejected').textContent =
        ITEMS.filter(c => statusOf(c) === 'rejected').length;
    document.getElementById('stat-pending').textContent =
        ITEMS.filter(c => statusOf(c) === 'pending').length;
    document.getElementById('stat-letter').textContent =
        ITEMS.filter(c => c.letter).length;
}}

function select(el, id) {{
    document.querySelectorAll('.card.selected').forEach(c => c.classList.remove('selected'));
    el.classList.add('selected');
    selected = ITEMS.find(c => c.id === id);
    if (!selected) return;
    document.getElementById('dimg').src = `data/glyphs/{region}/${{selected.id}}.png`;
    document.getElementById('dletter').textContent =
        (selected.letter || '—') + '  ·  ' + statusOf(selected);
    let meta = '';
    // Lat/lon → Google Maps link at the top of the meta block.
    if (selected.lat != null && selected.lon != null) {{
        const lat = selected.lat;
        const lon = selected.lon;
        const url = `https://www.google.com/maps/@${{lat}},${{lon}},14z/data=!3m1!1e3`;
        meta += `<div><a class="map-link" href="${{url}}" target="_blank" rel="noopener noreferrer">📍 Open ${{lat}}, ${{lon}} in Google Maps (satellite)</a></div>`;
    }}
    for (const [k, v] of Object.entries(selected)) {{
        if (k === 'id' || k === 'descriptor') continue;
        const val = (typeof v === 'object') ? JSON.stringify(v) : v;
        meta += `<div><span>${{k}}:</span> ${{val}}</div>`;
    }}
    document.getElementById('dmeta').innerHTML = meta;
    document.getElementById('detail').classList.add('open');
    document.getElementById('backdrop').classList.add('open');
    highlightLetterRow();
}}

function highlightLetterRow() {{
    document.querySelectorAll('#letter-row button').forEach(b => b.classList.remove('on'));
    if (!selected || !selected.letter) return;
    [...document.querySelectorAll('#letter-row button')]
        .find(b => b.textContent === selected.letter)
        ?.classList.add('on');
}}

function closeDetail() {{
    document.getElementById('detail').classList.remove('open');
    document.getElementById('backdrop').classList.remove('open');
}}

async function setStatus(status) {{
    if (!selected) return;
    const id = selected.id;
    const resp = await fetch(`/api/candidates/${{id}}/status`, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ status }}),
    }});
    if (!resp.ok) {{ flash('failed'); return; }}
    selected.status = status;
    flash(`${{status}}`);
    render();
}}

async function setLetter(letter) {{
    if (!selected) return;
    const id = selected.id;
    const resp = await fetch(`/api/candidates/${{id}}/letter`, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ letter }}),
    }});
    if (!resp.ok) {{ flash('failed'); return; }}
    selected.letter = letter;
    document.getElementById('dletter').textContent =
        (letter || '—') + '  ·  ' + statusOf(selected);
    flash(letter ? `→ ${{letter}}` : 'cleared');
    highlightLetterRow();
    render();
}}

async function del() {{
    if (!selected) return;
    if (!confirm('Delete this candidate (PNG + metadata)?')) return;
    const id = selected.id;
    const resp = await fetch(`/api/candidates/${{id}}/delete`, {{ method: 'POST' }});
    if (!resp.ok) {{ flash('failed'); return; }}
    ITEMS = ITEMS.filter(c => c.id !== id);
    selected = null;
    closeDetail();
    flash('deleted');
    render();
}}

async function showSimilar() {{
    if (!selected) return;
    const id = selected.id;
    const resp = await fetch(`/api/candidates/${{id}}/similar?k=48`);
    if (!resp.ok) {{ flash('similar lookup failed'); return; }}
    const data = await resp.json();
    similarOrder = [id, ...data.candidates.map(c => c.id)];
    similarMode = id;
    fSort.value = 'similar';
    fStatus.value = 'all';
    flash(`showing ${{data.candidates.length}} similar shapes`);
    closeDetail();
    render();
}}

document.addEventListener('keydown', e => {{
    if (e.key === 'Escape') return closeDetail();
    const open = document.getElementById('detail').classList.contains('open');
    if (!open || !selected) return;
    if (e.key === ' ') {{
        e.preventDefault();
        setStatus(statusOf(selected) === 'accepted' ? 'pending' : 'accepted');
        return;
    }}
    if (e.key === 'x' || e.key === 'X') return setStatus('rejected');
    if (e.key === 'Backspace' || e.key === 'Delete') return del();
    if (e.key.length === 1 && /[a-zA-Z]/.test(e.key)) {{
        setLetter(e.key.toUpperCase());
    }}
}});

buildLetterRow();
render();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class GalleryHandler(SimpleHTTPRequestHandler):
    def _json(self, code: int, payload):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", "0"))
        if n == 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            html = build_html(self.server.region, self.server.store).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if self.path == "/api/candidates":
            return self._json(200, {"candidates": self.server.store.all()})

        # /api/candidates/<id>/similar?k=24
        if self.path.startswith("/api/candidates/") and "/similar" in self.path:
            head, _, query = self.path.partition("?")
            parts = head.strip("/").split("/")
            if len(parts) == 4 and parts[3] == "similar":
                cid = parts[2]
                k = 24
                for kv in (query or "").split("&"):
                    if kv.startswith("k="):
                        try:
                            k = int(kv[2:])
                        except ValueError:
                            pass
                return self._json(
                    200,
                    {"candidates": self.server.store.find_similar(cid, k=k)},
                )

        return super().do_GET()

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "candidates":
            cid, action = parts[2], parts[3]
            store: CandidateStore = self.server.store
            if action == "status":
                body = self._read_json()
                return self._json(200, {"ok": store.set_status(cid, body.get("status"))})
            if action == "letter":
                body = self._read_json()
                return self._json(200, {"ok": store.set_letter(cid, body.get("letter"))})
            if action == "delete":
                c = store.get(cid)
                ok = store.delete(cid)
                if ok and c:
                    img_dir = PROJECT_ROOT / "data" / "glyphs" / self.server.region
                    p = img_dir / f"{cid}.png"
                    if p.exists():
                        try:
                            p.unlink()
                        except OSError:
                            pass
                return self._json(200 if ok else 404, {"ok": ok})
        return self._json(404, {"ok": False, "error": "no such route"})

    def log_message(self, format, *args):
        return


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Landscript gallery viewer")
    parser.add_argument("--region", default="bangalore",
                        help="Region or country id (the basename of the JSON "
                             "file in data/candidates/).")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--open", action="store_true", help="Open browser")
    args = parser.parse_args()

    cand_path = CAND_DIR / f"{args.region}.json"
    if not cand_path.exists():
        print(f"No candidates found at {cand_path}.\n"
              f"Run the pipeline first:\n"
              f"  python run_pipeline.py --region {args.region}")
        return

    store = CandidateStore(cand_path)
    addr = ("", args.port)
    server = HTTPServer(addr, GalleryHandler)
    server.region = args.region
    server.store = store
    try:
        host = socket.gethostbyname(socket.gethostname())
    except (socket.gaierror, OSError):
        host = None
    print(f"Gallery: http://localhost:{args.port}/")
    if host:
        print(f"         http://{host}:{args.port}/")
    print(f"  Region: {args.region}  |  {store.count()} candidates")
    if args.open:
        webbrowser.open(f"http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()