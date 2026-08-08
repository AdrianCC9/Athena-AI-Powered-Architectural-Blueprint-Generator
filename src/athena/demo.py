from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse
import re
import socket
import webbrowser


DEFAULT_PROMPT = "two bedroom apartment with open kitchen, living room, and one bathroom"
ROOM_PALETTE = {
    "Living": "#d8eef2",
    "Kitchen": "#f7e0b5",
    "Dining": "#e9d7f7",
    "Bedroom": "#dcebc2",
    "Bathroom": "#cbdcf7",
    "Office": "#f4cfce",
    "Laundry": "#d9d2c3",
    "Garage": "#d7dadd",
    "Balcony": "#cfe8d5",
}
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


def export_demo_html(prompt=DEFAULT_PROMPT, output_path="outputs/athena_demo.html"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_demo_page(prompt), encoding="utf-8")
    return output_path


def build_demo_page(prompt=DEFAULT_PROMPT):
    prompt = prompt.strip() or DEFAULT_PROMPT
    spec = parse_floorplan_prompt(prompt)
    svg = build_floorplan_svg(prompt)
    escaped_prompt = escape(prompt, quote=True)
    encoded_prompt = quote(prompt)
    summary = _summary_list_html(spec)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Athena Floorplan Demo</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #5c6870;
      --line: #d7dee2;
      --panel: #ffffff;
      --bg: #f5f2ea;
      --accent: #1f7a8c;
      --accent-dark: #135e6e;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    main {{
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 36px;
    }}

    header {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 18px;
    }}

    h1 {{
      margin: 0;
      font-size: clamp(2rem, 6vw, 4.5rem);
      line-height: 0.95;
      letter-spacing: 0;
    }}

    .tagline {{
      margin: 10px 0 0;
      max-width: 640px;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.5;
    }}

    .badge {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--muted);
      padding: 9px 12px;
      border-radius: 999px;
      white-space: nowrap;
      font-size: 0.9rem;
    }}

    form {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      margin: 18px 0;
    }}

    label {{
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }}

    input {{
      width: 100%;
      min-height: 48px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 14px;
      font: inherit;
      background: var(--panel);
      color: var(--ink);
    }}

    button,
    .download {{
      min-height: 48px;
      border: 0;
      border-radius: 8px;
      padding: 0 18px;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }}

    button:hover,
    .download:hover {{ background: var(--accent-dark); }}

    .workspace {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 18px;
      align-items: start;
    }}

    .canvas {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 18px 45px rgba(37, 46, 52, 0.12);
    }}

    .canvas svg {{
      display: block;
      width: 100%;
      height: auto;
    }}

    aside {{
      display: grid;
      gap: 12px;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}

    h2 {{
      margin: 0 0 10px;
      font-size: 0.96rem;
      letter-spacing: 0;
    }}

    p,
    li {{
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.45;
    }}

    ul {{
      margin: 0;
      padding-left: 18px;
    }}

    code {{
      background: #eef3f4;
      border-radius: 5px;
      padding: 2px 5px;
      color: var(--ink);
    }}

    @media (max-width: 760px) {{
      main {{ width: min(100vw - 20px, 680px); padding-top: 18px; }}
      header {{ display: block; }}
      .badge {{ display: inline-flex; margin-top: 12px; }}
      form {{ grid-template-columns: 1fr; }}
      .workspace {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Athena</h1>
        <p class="tagline">Text-guided concept plans for fast floorplan exploration.</p>
      </div>
      <div class="badge">Local demo mode</div>
    </header>

    <form method="get" action="/">
      <label for="prompt">Floorplan prompt</label>
      <input id="prompt" name="prompt" value="{escaped_prompt}" placeholder="Try: three bedroom house with office, garage, and balcony">
      <button type="submit">Generate</button>
    </form>

    <section class="workspace" aria-label="Athena demo workspace">
      <div class="canvas">{svg}</div>
      <aside>
        <div class="panel">
          <h2>Prompt</h2>
          <p>{escaped_prompt}</p>
        </div>
        <div class="panel">
          <h2>Plan Summary</h2>
          {summary}
        </div>
        <a class="download" href="/floorplan.svg?prompt={encoded_prompt}" download="athena-floorplan.svg">Download SVG</a>
      </aside>
    </section>
  </main>
</body>
</html>
"""


def build_floorplan_svg(prompt=DEFAULT_PROMPT, width=920, height=640):
    spec = parse_floorplan_prompt(prompt)
    rooms = layout_rooms(spec, width=width, height=height)
    room_markup = "\n".join(_room_svg(room) for room in rooms)
    door_markup = "\n".join(_door_svg(room) for room in rooms if room["name"] != "Balcony")
    window_markup = "\n".join(_window_svg(room) for room in rooms)
    x0, y0, plan_width, plan_height = _plan_bounds(width, height)

    return f"""<svg role="img" aria-label="Generated floorplan" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">
      <path d="M 24 0 L 0 0 0 24" fill="none" stroke="#e8eef0" stroke-width="1"/>
    </pattern>
  </defs>
  <rect width="{width}" height="{height}" fill="#fbfaf6"/>
  <rect x="0" y="0" width="{width}" height="{height}" fill="url(#grid)" opacity="0.8"/>
  {room_markup}
  <rect x="{x0}" y="{y0}" width="{plan_width}" height="{plan_height}" fill="none" stroke="#172026" stroke-width="8"/>
  {door_markup}
  {window_markup}
</svg>"""


def parse_floorplan_prompt(prompt):
    prompt = prompt.strip() or DEFAULT_PROMPT
    lower = prompt.lower()
    bedrooms = _count_for(lower, "bedroom", default=2)
    bathrooms = _count_for(lower, "bathroom", default=1)
    return {
        "prompt": prompt,
        "bedrooms": min(max(bedrooms, 1), 5),
        "bathrooms": min(max(bathrooms, 1), 3),
        "has_dining": "dining" in lower,
        "has_office": any(term in lower for term in ("office", "study", "den")),
        "has_laundry": "laundry" in lower,
        "has_garage": "garage" in lower,
        "has_balcony": any(term in lower for term in ("balcony", "patio", "terrace")),
    }


def layout_rooms(spec, width=920, height=640):
    x0, y0, plan_width, plan_height = _plan_bounds(width, height)
    public_width = int(plan_width * (0.58 if spec["bedrooms"] <= 3 else 0.52))
    private_width = plan_width - public_width
    rooms = []

    living_height = int(plan_height * (0.56 if spec["has_office"] else 0.62))
    if spec["has_office"]:
        office_width = int(public_width * 0.36)
        rooms.append(_room("Living", x0, y0, public_width - office_width, living_height))
        rooms.append(_room("Office", x0 + public_width - office_width, y0, office_width, living_height))
    else:
        rooms.append(_room("Living", x0, y0, public_width, living_height))

    lower_y = y0 + living_height
    lower_height = plan_height - living_height
    if spec["has_garage"]:
        garage_height = int(lower_height * 0.46)
        lower_height -= garage_height
        rooms.append(_room("Garage", x0, y0 + plan_height - garage_height, public_width, garage_height))

    if spec["has_dining"]:
        kitchen_width = int(public_width * 0.56)
        rooms.append(_room("Kitchen", x0, lower_y, kitchen_width, lower_height))
        rooms.append(_room("Dining", x0 + kitchen_width, lower_y, public_width - kitchen_width, lower_height))
    else:
        rooms.append(_room("Kitchen", x0, lower_y, public_width, lower_height))

    right_x = x0 + public_width
    private_rooms = []
    private_rooms.extend(f"Bedroom {i + 1}" for i in range(spec["bedrooms"]))
    private_rooms.extend(f"Bathroom {i + 1}" for i in range(spec["bathrooms"]))
    if spec["has_laundry"]:
        private_rooms.append("Laundry")

    private_room_names = private_rooms or ["Bedroom 1"]
    row_height = plan_height / len(private_room_names)
    for index, name in enumerate(private_room_names):
        y = y0 + int(index * row_height)
        next_y = y0 + int((index + 1) * row_height)
        rooms.append(_room(name, right_x, y, private_width, next_y - y))

    if spec["has_balcony"]:
        rooms.append(_room("Balcony", x0 + 20, y0 - 38, int(public_width * 0.7), 38))

    return rooms


def _plan_bounds(width, height):
    return 42, 64, width - 84, height - 106


def serve_demo(host="127.0.0.1", port=8000, prompt=DEFAULT_PROMPT, open_browser=True):
    server_port = _available_port(host, port)
    handler = _handler_for(prompt)
    server = ThreadingHTTPServer((host, server_port), handler)
    url = f"http://{host}:{server_port}"
    print(f"[INFO] Athena demo running at {url}")
    print("[INFO] Press Ctrl+C to stop the server.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Stopping Athena demo server.")
    finally:
        server.server_close()


def _handler_for(default_prompt):
    class DemoHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            prompt = unquote(query.get("prompt", [default_prompt])[0])
            if parsed.path in ("", "/"):
                self._send_text(build_demo_page(prompt), "text/html; charset=utf-8")
                return
            if parsed.path == "/floorplan.svg":
                self._send_text(build_floorplan_svg(prompt), "image/svg+xml; charset=utf-8")
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format, *args):
            return

        def _send_text(self, text, content_type):
            body = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DemoHandler


def _room(name, x, y, width, height):
    family = "Bedroom" if name.startswith("Bedroom") else "Bathroom" if name.startswith("Bathroom") else name
    return {
        "name": name,
        "family": family,
        "x": int(x),
        "y": int(y),
        "width": int(width),
        "height": int(height),
    }


def _room_svg(room):
    color = ROOM_PALETTE.get(room["family"], "#eef1f2")
    label = escape(room["name"])
    area = int(room["width"] * room["height"] / 120)
    cx = room["x"] + room["width"] / 2
    cy = room["y"] + room["height"] / 2
    label_size = 18 if room["width"] > 140 and room["height"] > 72 else 13
    return f"""<g>
    <rect x="{room['x']}" y="{room['y']}" width="{room['width']}" height="{room['height']}" fill="{color}" stroke="#172026" stroke-width="4"/>
    <text x="{cx:.1f}" y="{cy - 4:.1f}" text-anchor="middle" font-size="{label_size}" font-weight="800" fill="#172026">{label}</text>
    <text x="{cx:.1f}" y="{cy + 17:.1f}" text-anchor="middle" font-size="12" fill="#52616b">{area} sq ft</text>
  </g>"""


def _door_svg(room):
    x = room["x"] + min(42, max(20, room["width"] // 3))
    y = room["y"] + room["height"]
    return f"""<path d="M {x} {y} h 36" stroke="#fbfaf6" stroke-width="7"/>
  <path d="M {x} {y} q 20 -22 36 -36" fill="none" stroke="#866c3b" stroke-width="3"/>"""


def _window_svg(room):
    if room["height"] < 58 or room["width"] < 90:
        return ""
    x = room["x"] + room["width"] - min(72, max(42, room["width"] // 4))
    y = room["y"]
    return f"""<path d="M {x} {y} h 46" stroke="#2d8fb3" stroke-width="6"/>
  <path d="M {x} {y + 7} h 46" stroke="#fbfaf6" stroke-width="2"/>"""


def _count_for(prompt, noun, default):
    plural = f"{noun}s"
    patterns = [
        rf"\b(\d+)\s+{noun}s?\b",
        rf"\b({'|'.join(NUMBER_WORDS)})\s+{noun}s?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt)
        if match:
            value = match.group(1)
            return int(value) if value.isdigit() else NUMBER_WORDS[value]
    if plural in prompt or noun in prompt:
        return default
    return default


def _summary_list_html(spec):
    items = [
        f"{spec['bedrooms']} bedroom{'s' if spec['bedrooms'] != 1 else ''}",
        f"{spec['bathrooms']} bathroom{'s' if spec['bathrooms'] != 1 else ''}",
        "living room",
        "kitchen",
    ]
    for key, label in (
        ("has_dining", "dining area"),
        ("has_office", "office"),
        ("has_laundry", "laundry"),
        ("has_garage", "garage"),
        ("has_balcony", "balcony"),
    ):
        if spec[key]:
            items.append(label)
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


def _available_port(host, requested_port):
    for candidate in range(requested_port, requested_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, candidate))
            except OSError:
                continue
            return candidate
    raise OSError(f"No available port found from {requested_port} to {requested_port + 19}")
