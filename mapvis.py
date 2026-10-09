"""
mapviz.py

Visualization helpers for the disaster-geomapper project.

- create_map(): builds an interactive Folium/Leaflet map from a list of
  geolocated points and opens it in the browser. Each marker's popup
  includes a link into Mapillary centered on that spot.
- open_mapillary_at(): opens the Mapillary web viewer directly at a
  given lat/lon so you can browse the closest street-level imagery.

Setup:
    pip install folium

Usage:
    from mapviz import create_map, open_mapillary_at

    points = [
        {"lat": 40.7580, "lon": -73.9855, "label": "Times Square photo"},
        {"lat": 41.8781, "lon": -87.6298, "label": "Chicago photo"},
    ]
    create_map(points)
    open_mapillary_at(40.7580, -73.9855)
"""

import base64
import functools
import http.server
import io
import json
import os
import socketserver
import threading
import time
import webbrowser
import folium
from folium.plugins import MarkerCluster
from PIL import Image

from geolocate import POINTS_OUTPUT_PATH


def load_points(path: str = POINTS_OUTPUT_PATH) -> list[dict]:
    """Load geolocated points saved by geolocate.py."""
    with open(path) as f:
        return json.load(f)


def _open_via_local_server(output_path: str, port: int = 8765, keep_alive: float = 3.0) -> None:
    """
    Serve output_path's directory over http://localhost and open it there.

    Opening a Folium/Leaflet map straight from a file:// URL makes Chrome
    treat the page as a unique security origin (breaking same-origin frame
    checks) and sends tile requests with no proper Referer, which is exactly
    what OpenStreetMap's tile usage policy blocks. A brief local HTTP server
    avoids both — only the HTML document needs it, since the map's JS/CSS and
    tiles are all loaded from absolute https:// CDN/tile URLs afterward.
    """
    directory = os.path.dirname(os.path.abspath(output_path)) or "."
    filename = os.path.basename(output_path)

    class ReusableServer(socketserver.TCPServer):
        allow_reuse_address = True

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    httpd = ReusableServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    webbrowser.open(f"http://127.0.0.1:{port}/{filename}")
    time.sleep(keep_alive)  # give the browser time to fetch the page before the server goes away
    httpd.shutdown()
    httpd.server_close()


def _mapillary_url(lat: float, lon: float, zoom: int = 17) -> str:
    """Build a Mapillary web-app URL centered on a coordinate."""
    return f"https://www.mapillary.com/app/?lat={lat}&lng={lon}&z={zoom}"


def _image_to_data_uri(image_path: str, max_width: int = 320) -> str | None:
    """
    Downscale an image and embed it as a base64 data: URI, so the popup can
    show the actual source photo without the saved HTML depending on that
    image file still being around at the same path later (it's self-contained
    either served locally or opened as a standalone file).

    Returns None if the image can't be read, so callers can fall back to a
    text-only popup instead of breaking the whole map.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_width, max_width))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=80)
    except Exception:
        return None

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _marker_color(p: dict) -> str:
    """
    Color-code a marker by how much it should be trusted:
    green = confirmed via EXIF GPS, purple = an OCR-confirmed brand/POI match
    (see geolocate.py's find_brand_location_candidates), red = GeoCLIP's
    best/ensembled guess, blue = an alternate (lower-ranked) candidate,
    gray = no rank info (legacy points).
    """
    if p.get("source") == "exif":
        return "green"
    if p.get("origin") == "ocr_brand":
        return "purple"
    kind = p.get("kind")
    if kind == "best":
        return "red"
    if kind == "alt":
        return "blue"
    return "gray"


def create_map(points: list[dict], output_path: str = "disaster_map.html", open_browser: bool = True) -> folium.Map:
    """
    Build an interactive map from geolocated points and save it as HTML.

    Args:
        points: list of dicts with "lat", "lon", and optional "label", "kind"
            ("best"/"alt"), "source" ("exif"/"geoclip"), "spread_km",
            "num_clusters", and "image_path" (as produced by geolocate.py's
            analyze_image()). When "image_path" is set and readable, the
            source photo is embedded as a thumbnail in the marker's popup.
            Best/EXIF-sourced points are drawn as standalone markers so they
            stay visible when zoomed out; alternates are grouped into a
            cluster. A best point with spread_km > 0 gets a shaded circle
            showing GeoCLIP's uncertainty radius around it.
        output_path: where to save the generated HTML map.
        open_browser: if True, opens the saved map in your default browser.

    Returns:
        The folium.Map object (in case you want to keep customizing it).
    """
    if not points:
        raise ValueError("points list is empty - nothing to map")

    # Center the map on the average of all points
    avg_lat = sum(p["lat"] for p in points) / len(points)
    avg_lon = sum(p["lon"] for p in points) / len(points)

    fmap = folium.Map(location=[avg_lat, avg_lon], zoom_start=6, tiles="OpenStreetMap")
    cluster = MarkerCluster().add_to(fmap)

    for p in points:
        lat, lon = p["lat"], p["lon"]
        label = p.get("label", f"{lat:.5f}, {lon:.5f}")
        mapillary_link = _mapillary_url(lat, lon)
        kind = p.get("kind", "best")

        popup_lines = [f"<b>{label}</b><br>"]

        image_path = p.get("image_path")
        if image_path:
            data_uri = _image_to_data_uri(image_path)
            if data_uri:
                popup_lines.append(f'<img src="{data_uri}" style="max-width:100%; display:block; margin-bottom:4px;"><br>')

        popup_lines.append(f"{lat:.5f}, {lon:.5f}<br>")
        if p.get("source"):
            popup_lines.append(f"Source: {p['source']}<br>")
        if p.get("origin") == "ocr_brand":
            popup_lines.append(
                f"OCR brand match: '{p.get('matched_text')}' found ~{p.get('radius_km', 0):.0f}km "
                f"from GeoCLIP's guess<br>"
            )
        spread_km = p.get("spread_km")
        if spread_km:
            popup_lines.append(
                f"GeoCLIP uncertainty: ~{spread_km:.1f}km spread across "
                f"{p.get('num_clusters', 1)} region(s)<br>"
            )
        popup_lines.append(
            f'<a href="{mapillary_link}" target="_blank">Open nearest street-level imagery (Mapillary)</a>'
        )
        popup_html = "".join(popup_lines)

        marker = folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=340),
            tooltip=label,
            icon=folium.Icon(color=_marker_color(p)),
        )
        # Keep the best/only guess visible even when zoomed out; bury lower-ranked
        # alternates in the cluster so they don't clutter the default view.
        marker.add_to(fmap if kind == "best" else cluster)

        if kind == "best" and spread_km:
            folium.Circle(
                location=[lat, lon],
                radius=spread_km * 1000,
                color="orange",
                fill=True,
                fill_opacity=0.08,
                weight=1,
                tooltip=f"GeoCLIP uncertainty radius: ~{spread_km:.1f}km",
            ).add_to(fmap)

    fmap.save(output_path)

    if open_browser:
        _open_via_local_server(output_path)

    return fmap


def open_mapillary_at(lat: float, lon: float, zoom: int = 17) -> None:
    """Open the Mapillary web viewer directly in the browser, centered on a coordinate."""
    webbrowser.open(_mapillary_url(lat, lon, zoom))


if __name__ == "__main__":
    if os.path.exists(POINTS_OUTPUT_PATH):
        points = load_points()
    else:
        print(f"No {POINTS_OUTPUT_PATH} found - run geolocate.py first. Using sample points instead.")
        points = [
            {"lat": 40.7580, "lon": -73.9855, "label": "Sample point 1"},
            {"lat": 41.8781, "lon": -87.6298, "label": "Sample point 2"},
        ]
    create_map(points)