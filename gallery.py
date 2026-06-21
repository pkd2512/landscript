#!/usr/bin/env python3
"""Landscript gallery: browse, curate (accept/reject/delete) and filter glyphs.

Phase B of the roadmap: persistent review decisions.

Routes:
    GET  /                      → HTML gallery
    GET  /api/glyphs            → JSON list of all glyphs
    POST /api/glyphs/<id>/accept
    POST /api/glyphs/<id>/reject
    POST /api/glyphs/<id>/pending
    POST /api/glyphs/<id>/delete
    GET  /data/...              → static files (tiles, glyph PNGs)
"""

import argparse
import json
import socket
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from landscript.metadata import GlyphStore

PROJECT_ROOT = Path(__file__).resolve().parent
GLYPHS_DIR = PROJECT_ROOT / "data" / "glyphs"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def build_gallery_html(region: str, store: GlyphStore) -> str:
    """Render the gallery HTML for a region. All filtering happens client-side."""
    glyphs = store.all(limit=10000)
    by_letter: dict = {}
    for g in glyphs:
        by_letter.setdefault(g.get("letter", "?"), []).append(g)

    accepted = sum(1 for g in glyphs if g.get("status") == "accepted")
    rejected = sum(1 for g in glyphs if g.get("status") == "rejected")
    pending = len(glyphs) - accepted - rejected

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
h1 {{ font-size: 1.5rem; font-weight: 300; margin-bottom: 4px; letter-spacing: 0.1em; }}
h1 span {{ color: #666; }}
.sub {{ color: #666; font-size: 0.85rem; margin-bottom: 16px; }}
.stats {{ display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
.stat {{ background: #141414; border-radius: 8px; padding: 10px 16px; }}
.stat .num {{ font-size: 1.3rem; font-weight: 600; color: #fff; }}
.stat .label {{ font-size: 0.7rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }}

/* Toolbar */
.toolbar {{ position: sticky; top: 0; background: #0a0a0a; padding: 12px 0;
            border-bottom: 1px solid #1a1a1a; margin-bottom: 16px; z-index: 50;
            display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
.toolbar label {{ font-size: 0.75rem; color: #888; display: flex; align-items: center; gap: 4px; }}
.toolbar select, .toolbar input[type=number] {{ background: #1a1a1a; color: #e0e0e0;
    border: 1px solid #2a2a2a; border-radius: 5px; padding: 5px 8px; font-size: 0.8rem; }}
.toolbar input[type=number] {{ width: 70px; }}
.toolbar button.chip {{ background: #1a1a1a; color: #e0e0e0; border: 1px solid #2a2a2a;
    border-radius: 5px; padding: 5px 10px; font-size: 0.75rem; cursor: pointer; }}
.toolbar button.chip.on {{ background: #2a2a2a; border-color: #444; color: #fff; }}

.letter-group {{ margin-bottom: 32px; }}
.letter-group h2 {{ font-size: 1rem; font-weight: 600; margin-bottom: 8px;
                   border-bottom: 1px solid #222; padding-bottom: 4px; }}
.letter-group h2 .count {{ color: #666; font-size: 0.8rem; font-weight: 400; margin-left: 8px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 8px; }}
.card {{ background: #141414; border-radius: 6px; overflow: hidden; cursor: pointer;
         border: 2px solid transparent; transition: all 0.15s; position: relative; }}
.card:hover {{ border-color: #444; transform: translateY(-2px); }}
.card.accepted {{ border-color: #22c55e; }}
.card.rejected {{ opacity: 0.35; border-color: #ef4444; }}
.card.selected {{ border-color: #3b82f6; box-shadow: 0 0 0 2px #3b82f6; }}
.card .badge {{ position: absolute; top: 4px; right: 4px; font-size: 0.65rem;
    padding: 2px 6px; border-radius: 4px; font-weight: 600; }}
.card.accepted .badge {{ background: #22c55e; color: #000; }}
.card.rejected .badge {{ background: #ef4444; color: #fff; }}
.card.pending .badge {{ display: none; }}
.card img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block;
            background: #0f0f0f; }}
.card .info {{ display: flex; justify-content: space-between; padding: 4px 8px; font-size: 0.75rem; }}
.card .letter {{ font-weight: 600; }}
.card .score {{ color: #888; }}

/* Detail panel */
#detail {{ display: none; position: fixed; top: 0; right: 0; width: 420px; height: 100%;
          background: #111; border-left: 1px solid #222; padding: 24px; overflow-y: auto;
          z-index: 100; }}
#detail.open {{ display: block; }}
#detail img {{ width: 100%; border-radius: 8px; background: #0a0a0a; margin-bottom: 16px; }}
#detail h3 {{ font-size: 1.3rem; margin-bottom: 4px; }}
#detail .meta {{ font-size: 0.8rem; color: #888; line-height: 1.7; }}
#detail .meta div {{ word-break: break-word; }}
#detail .meta span {{ color: #ccc; margin-right: 4px; }}
#close {{ position: absolute; top: 12px; right: 16px; background: none; border: none;
          color: #666; font-size: 1.5rem; cursor: pointer; }}
#close:hover {{ color: #fff; }}
.actions {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 16px; }}
.actions button {{ padding: 10px; border: 1px solid #333; border-radius: 6px;
                  background: #1a1a1a; color: #e0e0e0; cursor: pointer; font-size: 0.85rem; }}
.actions button:hover {{ background: #222; }}
.actions .accept {{ border-color: #22c55e; color: #22c55e; }}
.actions .accept:hover {{ background: #22c55e22; }}
.actions .reject {{ border-color: #ef4444; color: #ef4444; }}
.actions .reject:hover {{ background: #ef444422; }}
.actions .delete {{ border-color: #f59e0b; color: #f59e0b; grid-column: span 1; }}
.actions .delete:hover {{ background: #f59e0b22; }}
.actions .pending {{ grid-column: span 1; }}
#backdrop {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 99; }}
#backdrop.open {{ display: block; }}

#status-msg {{ position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
    background: #1f1f1f; color: #fff; padding: 8px 16px; border-radius: 6px;
    font-size: 0.8rem; opacity: 0; transition: opacity 0.2s; pointer-events: none;
    border: 1px solid #333; }}
#status-msg.show {{ opacity: 1; }}

@media (max-width: 768px) {{
    #detail {{ width: 100%; }}
}}
</style>
</head>
<body>

<h1>Landscript <span>— {region}</span></h1>
<div class="sub">{region}.json · {len(glyphs)} glyphs · {len(by_letter)} letters · keys: <b>A</b> accept · <b>R</b> reject · <b>D</b> delete · <b>Esc</b> close</div>

<div class="stats">
    <div class="stat"><div class="num" id="stat-total">{len(glyphs)}</div><div class="label">Total</div></div>
    <div class="stat"><div class="num" id="stat-accepted">{accepted}</div><div class="label">Accepted</div></div>
    <div class="stat"><div class="num" id="stat-rejected">{rejected}</div><div class="label">Rejected</div></div>
    <div class="stat"><div class="num" id="stat-pending">{pending}</div><div class="label">Pending</div></div>
    <div class="stat"><div class="num">{len(by_letter)}</div><div class="label">Letters</div></div>
</div>

<div class="toolbar">
    <label>Show
        <select id="f-status">
            <option value="all">all</option>
            <option value="pending" selected>pending</option>
            <option value="accepted">accepted</option>
            <option value="rejected">rejected</option>
            <option value="not_rejected">accepted + pending</option>
        </select>
    </label>
    <label>Top
        <input type="number" id="f-topn" value="0" min="0" step="5">
        <span style="color:#555">per letter (0 = all)</span>
    </label>
    <label>Letter
        <select id="f-letter">
            <option value="">all</option>
            {''.join(f'<option value="{L}">{L}</option>' for L in sorted(by_letter.keys()))}
        </select>
    </label>
    <label>Max score
        <input type="number" id="f-maxscore" value="1.0" step="0.01" min="0" max="2">
    </label>
</div>

<div id="gallery"></div>

<div id="backdrop" onclick="closeDetail()"></div>
<div id="detail">
    <button id="close" onclick="closeDetail()">✕</button>
    <img id="dimg" src="">
    <h3 id="dletter"></h3>
    <div class="meta" id="dmeta"></div>
    <div class="actions">
        <button class="accept" onclick="setStatus('accepted')">✓ Accept</button>
        <button class="reject" onclick="setStatus('rejected')">✗ Reject</button>
        <button class="pending" onclick="setStatus('pending')">↺ Pending</button>
        <button class="delete" onclick="del()">🗑 Delete</button>
    </div>
</div>

<div id="status-msg"></div>

<script>
let GLYPHS = {json.dumps(glyphs)};
let selected = null;

const fStatus = document.getElementById('f-status');
const fTopN = document.getElementById('f-topn');
const fLetter = document.getElementById('f-letter');
const fMaxScore = document.getElementById('f-maxscore');
[fStatus, fTopN, fLetter, fMaxScore].forEach(el => el.addEventListener('input', render));

function statusOf(g) {{ return g.status || (g.accepted ? 'accepted' : 'pending'); }}

function flash(msg) {{
    const el = document.getElementById('status-msg');
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.remove('show'), 1500);
}}

function applyFilters() {{
    const status = fStatus.value;
    const topN = parseInt(fTopN.value, 10) || 0;
    const letter = fLetter.value;
    const maxScore = parseFloat(fMaxScore.value);

    let list = GLYPHS.filter(g => {{
        if (letter && g.letter !== letter) return false;
        if (!isNaN(maxScore) && g.score > maxScore) return false;
        const s = statusOf(g);
        if (status === 'all') return true;
        if (status === 'not_rejected') return s !== 'rejected';
        return s === status;
    }});

    // Group by letter, sort within each by score asc, then optionally top-N.
    const groups = {{}};
    for (const g of list) (groups[g.letter] = groups[g.letter] || []).push(g);
    for (const k of Object.keys(groups)) {{
        groups[k].sort((a,b) => (a.score ?? 1) - (b.score ?? 1));
        if (topN > 0) groups[k] = groups[k].slice(0, topN);
    }}
    return groups;
}}

function render() {{
    const groups = applyFilters();
    const root = document.getElementById('gallery');
    const letters = Object.keys(groups).sort();
    if (letters.length === 0) {{
        root.innerHTML = '<div style="color:#666;padding:40px;text-align:center">No glyphs match the current filters.</div>';
        return;
    }}
    let html = '';
    for (const L of letters) {{
        const items = groups[L];
        html += `<div class="letter-group"><h2 id="${{L}}">${{L}} <span class="count">${{items.length}}</span></h2><div class="grid">`;
        for (const g of items) {{
            const s = statusOf(g);
            const img = `data/glyphs/${{g.letter}}/${{g.id}}.png`;
            const coords = (g.lat && g.lon) ? `<div class="coords">${{g.lat}}, ${{g.lon}}</div>` : '';
            html += `
            <div class="card ${{s}}" onclick="select(this, '${{g.id}}')">
                <span class="badge">${{s}}</span>
                <img src="${{img}}" loading="lazy" onerror="this.parentElement.style.display='none'">
                <div class="info"><span class="letter">${{g.letter}}</span><span class="score">${{(g.score ?? 0).toFixed(3)}}</span></div>
                ${{coords}}
            </div>`;
        }}
        html += '</div></div>';
    }}
    root.innerHTML = html;
    refreshStats();
}}

function refreshStats() {{
    document.getElementById('stat-total').textContent = GLYPHS.length;
    document.getElementById('stat-accepted').textContent = GLYPHS.filter(g => statusOf(g) === 'accepted').length;
    document.getElementById('stat-rejected').textContent = GLYPHS.filter(g => statusOf(g) === 'rejected').length;
    document.getElementById('stat-pending').textContent = GLYPHS.filter(g => statusOf(g) === 'pending').length;
}}

function select(el, id) {{
    document.querySelectorAll('.card.selected').forEach(c => c.classList.remove('selected'));
    el.classList.add('selected');
    selected = GLYPHS.find(g => g.id === id);
    if (!selected) return;
    document.getElementById('dimg').src = `data/glyphs/${{selected.letter}}/${{selected.id}}.png`;
    document.getElementById('dletter').textContent = selected.letter + '  ·  ' + statusOf(selected);
    let meta = '';
    for (const [k, v] of Object.entries(selected)) {{
        if (k === 'id' || k === 'letter') continue;
        const val = (typeof v === 'object') ? JSON.stringify(v) : v;
        meta += `<div><span>${{k}}:</span> ${{val}}</div>`;
    }}
    document.getElementById('dmeta').innerHTML = meta;
    document.getElementById('detail').classList.add('open');
    document.getElementById('backdrop').classList.add('open');
}}

function closeDetail() {{
    document.getElementById('detail').classList.remove('open');
    document.getElementById('backdrop').classList.remove('open');
}}

async function setStatus(status) {{
    if (!selected) return;
    const id = selected.id;
    const resp = await fetch(`/api/glyphs/${{id}}/${{status}}`, {{method: 'POST'}});
    if (!resp.ok) {{ flash('failed: ' + resp.status); return; }}
    selected.status = status;
    selected.accepted = (status === 'accepted');
    flash(`${{id.substring(6,17)}} → ${{status}}`);
    render();
}}

async function del() {{
    if (!selected) return;
    const id = selected.id;
    if (!confirm('Delete this glyph (PNG + metadata)?')) return;
    const resp = await fetch(`/api/glyphs/${{id}}/delete`, {{method: 'POST'}});
    if (!resp.ok) {{ flash('failed: ' + resp.status); return; }}
    GLYPHS = GLYPHS.filter(g => g.id !== id);
    selected = null;
    flash('deleted');
    closeDetail();
    render();
}}

document.addEventListener('keydown', e => {{
    if (e.key === 'Escape') return closeDetail();
    if (!selected || !document.getElementById('detail').classList.contains('open')) return;
    if (e.key === 'a' || e.key === 'A') setStatus('accepted');
    else if (e.key === 'r' || e.key === 'R') setStatus('rejected');
    else if (e.key === 'p' || e.key === 'P') setStatus('pending');
    else if (e.key === 'd' || e.key === 'D') del();
}});

render();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class GalleryHandler(SimpleHTTPRequestHandler):
    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            store: GlyphStore = self.server.store
            html = build_gallery_html(self.server.region, store).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        if self.path == "/api/glyphs":
            self._json(200, {"glyphs": self.server.store.all(limit=100000)})
            return
        return super().do_GET()

    def do_POST(self):
        # Routes: /api/glyphs/<id>/<action>
        parts = self.path.strip("/").split("/")
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "glyphs":
            glyph_id, action = parts[2], parts[3]
            store: GlyphStore = self.server.store
            if action == "delete":
                g = store.get(glyph_id)
                ok = store.delete(glyph_id)
                if ok and g:
                    png = (PROJECT_ROOT / "data" / "glyphs" / g["letter"]
                           / f"{glyph_id}.png")
                    if png.exists():
                        try:
                            png.unlink()
                        except OSError:
                            pass
                return self._json(200 if ok else 404, {"ok": ok})
            if action in ("accepted", "rejected", "pending"):
                ok = store.set_status(glyph_id, action)
                return self._json(200 if ok else 404, {"ok": ok})
        self._json(404, {"ok": False, "error": "no such route"})

    def log_message(self, format, *args):  # quieter logs
        return


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Landscript gallery viewer")
    parser.add_argument("--region", default="bangalore")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--open", action="store_true", help="Open browser")
    args = parser.parse_args()

    meta_path = METADATA_DIR / f"{args.region}.json"
    if not meta_path.exists():
        print(f"No metadata found. Run the pipeline first:\n"
              f"  python run_pipeline.py --region {args.region}")
        return

    store = GlyphStore(meta_path)
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
    print(f"  Region: {args.region}  |  {store.count()} glyphs")
    if args.open:
        webbrowser.open(f"http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()