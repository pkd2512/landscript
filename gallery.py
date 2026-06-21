#!/usr/bin/env python3
"""Generate a static HTML gallery of all glyphs for inspection."""

import json
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

PROJECT_ROOT = Path(__file__).resolve().parent
GLYPHS_DIR = PROJECT_ROOT / "data" / "glyphs"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


def build_gallery_html(region: str = "bangalore") -> str:
    meta_path = METADATA_DIR / f"{region}.json"
    if not meta_path.exists():
        return f"<h1>No metadata found for {region}</h1>"

    with open(meta_path) as f:
        glyphs = json.load(f)

    by_letter = {}
    for g in glyphs:
        letter = g.get("letter", "?")
        by_letter.setdefault(letter, []).append(g)

    letters_html = ""
    for letter in sorted(by_letter.keys()):
        items = sorted(by_letter[letter], key=lambda x: x.get("score", 1))
        cards = ""
        for g in items:
            img_path = "data/glyphs/" + g["letter"] + "/" + g["id"] + ".png"
            badge = "accepted" if g.get("accepted") else "candidate"
            coords_html = ""
            if g.get("lat") and g.get("lon"):
                coords_html = '<div class="coords">{}, {}</div>'.format(g["lat"], g["lon"])
            cards += """
            <div class="card {}" onclick="select(this, '{}')">
                <img src="{}" loading="lazy" onerror="this.parentElement.style.display='none'">
                <div class="info">
                    <span class="letter">{}</span>
                    <span class="score">{:.3f}</span>
                </div>
                {}
            </div>""".format(badge, g["id"], img_path, g["letter"], g["score"], coords_html)

        letters_html += f"""
        <div class="letter-group">
            <h2 id="{letter}">{letter} <span class="count">{len(items)}</span></h2>
            <div class="grid">{cards}</div>
        </div>"""

    accepted = sum(1 for g in glyphs if g.get("accepted"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Landscript — {region}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0a0a0a; color: #e0e0e0; padding: 20px; }}
h1 {{ font-size: 1.5rem; font-weight: 300; margin-bottom: 4px; letter-spacing: 0.1em; }}
h1 span {{ color: #666; }}
.sub {{ color: #666; font-size: 0.85rem; margin-bottom: 24px; }}
.stats {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
.stat {{ background: #141414; border-radius: 8px; padding: 12px 20px; }}
.stat .num {{ font-size: 1.5rem; font-weight: 600; color: #fff; }}
.stat .label {{ font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }}
.letter-group {{ margin-bottom: 32px; }}
.letter-group h2 {{ font-size: 1rem; font-weight: 600; margin-bottom: 8px;
                   border-bottom: 1px solid #222; padding-bottom: 4px; }}
.letter-group h2 .count {{ color: #666; font-size: 0.8rem; font-weight: 400; margin-left: 8px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 8px; }}
.card {{ background: #141414; border-radius: 6px; overflow: hidden; cursor: pointer;
         border: 2px solid transparent; transition: all 0.15s; }}
.card:hover {{ border-color: #444; transform: translateY(-2px); }}
.card.accepted {{ border-color: #22c55e; }}
.card.selected {{ border-color: #3b82f6; box-shadow: 0 0 0 2px #3b82f6; }}
.card img {{ width: 100%; aspect-ratio: 1 / 2; object-fit: contain; display: block;
            background: #0f0f0f; }}
.card .info {{ display: flex; justify-content: space-between; padding: 4px 8px; font-size: 0.75rem; }}
.card .letter {{ font-weight: 600; }}
.card .score {{ color: #888; }}

/* Detail panel */
#detail {{ display: none; position: fixed; top: 0; right: 0; width: 360px; height: 100%;
          background: #111; border-left: 1px solid #222; padding: 24px; overflow-y: auto;
          z-index: 100; }}
#detail.open {{ display: block; }}
#detail img {{ width: 100%; border-radius: 8px; background: #0a0a0a; margin-bottom: 16px; }}
#detail h3 {{ font-size: 1.1rem; margin-bottom: 4px; }}
#detail .meta {{ font-size: 0.8rem; color: #888; line-height: 1.6; }}
#detail .meta span {{ color: #ccc; }}
#close {{ position: absolute; top: 12px; right: 16px; background: none; border: none;
          color: #666; font-size: 1.5rem; cursor: pointer; }}
#close:hover {{ color: #fff; }}
.actions {{ display: flex; gap: 8px; margin-top: 16px; }}
.actions button {{ flex: 1; padding: 8px; border: 1px solid #333; border-radius: 6px;
                  background: #1a1a1a; color: #e0e0e0; cursor: pointer; font-size: 0.8rem; }}
.actions button:hover {{ background: #222; }}
.actions .accept {{ border-color: #22c55e; color: #22c55e; }}
.actions .accept:hover {{ background: #22c55e22; }}
#backdrop {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 99; }}
#backdrop.open {{ display: block; }}
@media (max-width: 768px) {{
    #detail {{ width: 100%; }}
}}
</style>
</head>
<body>

<h1>Landscript <span>— {region}</span></h1>
<div class="sub">{meta_path.name} · {len(glyphs)} glyphs · {len(by_letter)} letters</div>

<div class="stats">
    <div class="stat"><div class="num">{len(glyphs)}</div><div class="label">Total</div></div>
    <div class="stat"><div class="num">{accepted}</div><div class="label">Accepted</div></div>
    <div class="stat"><div class="num">{len(glyphs) - accepted}</div><div class="label">Candidates</div></div>
    <div class="stat"><div class="num">{len(by_letter)}</div><div class="label">Letters</div></div>
</div>

{letters_html}

<div id="backdrop" onclick="closeDetail()"></div>
<div id="detail">
    <button id="close" onclick="closeDetail()">✕</button>
    <img id="dimg" src="">
    <h3 id="dletter"></h3>
    <div class="meta" id="dmeta"></div>
    <div class="actions">
        <button class="accept" onclick="toggleAccept()">✓ Accept</button>
        <button onclick="closeDetail()">Close</button>
    </div>
</div>

<script>
const GLYPHS = {json.dumps(glyphs, indent=2)};
let selected = null;

function select(el, id) {{
    document.querySelectorAll('.card.selected').forEach(c => c.classList.remove('selected'));
    el.classList.add('selected');
    selected = GLYPHS.find(g => g.id === id);
    if (!selected) return;
    document.getElementById('dimg').src = `data/glyphs/${{selected.letter}}/${{selected.id}}.png`;
    document.getElementById('dletter').textContent = selected.letter;
    let meta = '';
    for (const [k, v] of Object.entries(selected)) {{
        if (k === 'id' || k === 'letter') continue;
        if (typeof v === 'object') {{ meta += `<div><span>${{k}}:</span> ${{JSON.stringify(v)}}</div>`; continue; }}
        meta += `<div><span>${{k}}:</span> ${{v}}</div>`;
    }}
    document.getElementById('dmeta').innerHTML = meta;
    document.getElementById('detail').classList.add('open');
    document.getElementById('backdrop').classList.add('open');
}}

function closeDetail() {{
    document.getElementById('detail').classList.remove('open');
    document.getElementById('backdrop').classList.remove('open');
}}

function toggleAccept() {{
    if (!selected) return;
    selected.accepted = !selected.accepted;
    document.querySelectorAll('.card').forEach(c => {{
        const img = c.querySelector('img');
        if (img && img.src.includes(selected.id)) {{
            c.classList.toggle('accepted');
        }}
    }});
}}

document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeDetail(); }});
</script>
</body>
</html>"""


class GalleryHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            region = self.server.region
            html = build_gallery_html(region)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            super().do_GET()


def main():
    import argparse, socket
    parser = argparse.ArgumentParser(description="Landscript gallery viewer")
    parser.add_argument("--region", default="bangalore")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--open", action="store_true", help="Open browser")
    args = parser.parse_args()

    meta_path = METADATA_DIR / f"{args.region}.json"
    if not meta_path.exists():
        print(f"No metadata found. Run the pipeline first:\n  python run_pipeline.py --region {args.region}")
        return

    addr = ("", args.port)
    server = HTTPServer(addr, GalleryHandler)
    server.region = args.region
    host = socket.gethostbyname(socket.gethostname())
    print(f"Gallery: http://localhost:{args.port}/")
    print(f"         http://{host}:{args.port}/")
    if args.open:
        webbrowser.open(f"http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
