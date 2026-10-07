"""
ocr_match.py

Refines GeoCLIP's candidate GPS guesses using scene text.

Pipeline:
    1. extract_text_candidates() — run EasyOCR on the photo and keep the
       strings that look like proper nouns (brand names, street names).
    2. query_overpass_pois() — for each candidate string, ask OpenStreetMap's
       Overpass API for named POIs within a bounding box around one of
       GeoCLIP's top-k guesses.
    3. find_proximity_matches() — for every pair of distinct brand names,
       find POI pairs within `threshold_m` of each other using a KD-tree.
       A close pair (e.g. a Starbucks 40m from a McDonald's) is a much
       stronger location signature than either brand alone, since that
       specific spatial arrangement is rare.
    4. find_brand_location_candidates() — for a single distinctive name (no
       pair needed), search outward from an anchor point in expanding rings
       using query_overpass_pois_fuzzy() until a match turns up. Unlike (2)
       and (3), this can propose a location far outside GeoCLIP's own
       guesses, which matters when GeoCLIP is confidently wrong about the
       city/region.

Setup:
    pip install easyocr scipy requests

Usage:
    from ocr_match import extract_text_candidates, query_overpass_pois, find_proximity_matches

    texts = extract_text_candidates("photo.jpg")
    pois_by_name = {name: query_overpass_pois(name, bbox=(lat_min, lon_min, lat_max, lon_max))
                    for name in texts}
    matches = find_proximity_matches(pois_by_name, threshold_m=100)
"""

import re
from itertools import combinations

import requests
from scipy.spatial import cKDTree

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
EARTH_RADIUS_M = 6_371_000
# Overpass rejects requests carrying requests' default "python-requests/x.x"
# User-Agent with a 406, so every call needs an explicit one.
_OVERPASS_HEADERS = {"User-Agent": "disaster-geomapper/1.0 (scene-text location refinement)"}

_ocr_reader = None  # lazily built; loading EasyOCR's models is slow


def _get_reader(languages: list[str] | None = None):
    """Lazily construct and cache the EasyOCR reader (loading it is slow)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr  # deferred import: heavy, and pulls in torch
        # verbose=False: EasyOCR's download progress bar prints U+2588 block
        # characters, which crash on Windows consoles using cp1252 (charmap).
        _ocr_reader = easyocr.Reader(languages or ["en"], gpu=False, verbose=False)
    return _ocr_reader


def extract_text_candidates(image_path: str, min_confidence: float = 0.5, languages: list[str] | None = None) -> list[str]:
    """
    Run OCR on an image and return the detected strings that plausibly name
    a place (brand, business, street) rather than OCR noise.

    This is a heuristic filter, not NLP: it keeps strings that start with an
    uppercase letter and contain only letters/digits/spaces/basic punctuation,
    are at least 3 characters, and clears `min_confidence`. Anything that
    isn't actually a real POI name will simply return no Overpass results
    later, so false positives here are cheap.
    """
    reader = _get_reader(languages)
    results = reader.readtext(image_path)

    candidates = []
    seen = set()
    for _bbox, text, confidence in results:
        text = text.strip()
        if confidence < min_confidence:
            continue
        if len(text) < 3 or not text[0].isupper():
            continue
        if not re.fullmatch(r"[A-Za-z0-9&'.\- ]+", text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(text)

    return candidates


def _bbox_from_point(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """Build an Overpass-style (south, west, north, east) bbox around a point."""
    import math
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 1e-6))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


def query_overpass_pois(
    name: str,
    bbox: tuple[float, float, float, float] | None = None,
    center: tuple[float, float] | None = None,
    radius_km: float = 5.0,
    timeout: int = 25,
) -> list[dict]:
    """
    Query Overpass for named nodes/ways matching `name` within a bounding box.

    Pass either `bbox` directly as (south, west, north, east), or `center` +
    `radius_km` to build one around a GeoCLIP candidate point. Keep the box
    small (a few km) — Overpass rate-limits and is slow over large areas, so
    shrink the search region using GeoCLIP's top-k guesses first rather than
    searching a whole country.

    Returns a list of {"lat", "lon", "name"} dicts.
    """
    if bbox is None:
        if center is None:
            raise ValueError("query_overpass_pois requires either bbox or center")
        bbox = _bbox_from_point(center[0], center[1], radius_km)

    south, west, north, east = bbox
    escaped_name = name.replace('"', '\\"')
    query = f"""
    [out:json][timeout:{timeout}];
    nwr["name"="{escaped_name}"]({south},{west},{north},{east});
    out center {timeout};
    """

    response = requests.post(OVERPASS_URL, data={"data": query}, timeout=timeout, headers=_OVERPASS_HEADERS)
    response.raise_for_status()
    elements = response.json().get("elements", [])

    pois = []
    for el in elements:
        if "lat" in el and "lon" in el:
            lat, lon = el["lat"], el["lon"]
        elif "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue
        pois.append({"lat": lat, "lon": lon, "name": name})

    return pois


def query_overpass_pois_fuzzy(name: str, center: tuple[float, float], radius_km: float, timeout: int = 25) -> list[dict]:
    """
    Like query_overpass_pois(), but matches `name` as a case-insensitive
    substring against both the `name` and `brand` tags within `radius_km` of
    `center`, instead of requiring an exact `name=` match.

    Exact matching misses most franchise POIs: OSM usually tags each
    location with its full, location-specific name (e.g. "TownePlace Suites
    Champaign/Urbana"), not the bare brand text OCR extracts off a sign.
    """
    lat, lon = center
    radius_m = radius_km * 1000
    pattern = re.escape(name)

    # nwr (node/way/relation combined) halves the request cost vs separate
    # node[]/way[] clauses — regex tag matches can't use an index, so cost
    # scales with clause count and this is an expensive query either way.
    query = f"""
    [out:json][timeout:{timeout}];
    (
      nwr["name"~"{pattern}",i](around:{radius_m},{lat},{lon});
      nwr["brand"~"{pattern}",i](around:{radius_m},{lat},{lon});
    );
    out center {timeout};
    """

    response = requests.post(OVERPASS_URL, data={"data": query}, timeout=timeout, headers=_OVERPASS_HEADERS)
    response.raise_for_status()
    elements = response.json().get("elements", [])

    pois = []
    seen = set()
    for el in elements:
        if "lat" in el and "lon" in el:
            poi_lat, poi_lon = el["lat"], el["lon"]
        elif "center" in el:
            poi_lat, poi_lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue
        key = (round(poi_lat, 5), round(poi_lon, 5))
        if key in seen:
            continue
        seen.add(key)
        tags = el.get("tags", {})
        pois.append({"lat": poi_lat, "lon": poi_lon, "name": tags.get("name") or tags.get("brand") or name})

    return pois


def find_brand_location_candidates(
    texts: list[str],
    anchor: tuple[float, float],
    radii_km: tuple[float, ...] = (10.0, 50.0, 200.0),
    max_results_per_name: int = 5,
    timeout: int = 25,
    verbose: bool = True,
) -> list[dict]:
    """
    Search outward from `anchor` in expanding rings for each OCR'd name,
    stopping at the first radius that turns up a fuzzy name/brand match.

    This lets a single distinctive brand sighting propose a location even
    when it's nowhere near `anchor` — unlike find_proximity_matches(), which
    can only re-rank points that are already candidates. If GeoCLIP's own
    guess is in the wrong city entirely, a query boxed tightly around that
    guess can never see a hotel chain photographed somewhere else; expanding
    the search radius until something turns up fixes that.

    Regex tag matches can't use an index, so this is a genuinely expensive
    query for Overpass's public instance, and larger radii cost more. If it
    returns a 429, that instance's fair-use quota is spent for now — this
    stops immediately rather than burning through the remaining radii/names
    (which would only make the throttling worse); a self-hosted or paid
    Overpass endpoint would not have this ceiling.

    Returns a list of {"lat", "lon", "name", "matched_text", "radius_km"}
    dicts, one entry per name that found something (using the smallest
    radius that found it — closer confirmation is stronger evidence).
    """
    found = []
    rate_limited = False

    for text in texts:
        if rate_limited:
            break

        text_found = False
        for radius_km in radii_km:
            try:
                pois = query_overpass_pois_fuzzy(text, center=anchor, radius_km=radius_km, timeout=timeout)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    if verbose:
                        print("  (Overpass rate-limited this session; skipping remaining brand searches)")
                    rate_limited = True
                    break
                if verbose:
                    print(f"  (brand search for '{text}' failed at {radius_km:.0f}km: {e})")
                continue
            except Exception as e:
                if verbose:
                    print(f"  (brand search for '{text}' failed at {radius_km:.0f}km: {e})")
                continue

            if pois:
                if verbose:
                    print(f"  Brand search: '{text}' found {len(pois)} match(es) within {radius_km:.0f}km")
                for poi in pois[:max_results_per_name]:
                    found.append({**poi, "matched_text": text, "radius_km": radius_km})
                text_found = True
                break

        if not text_found and not rate_limited and verbose:
            print(f"  Brand search: '{text}' not found within {radii_km[-1]:.0f}km")

    return found


def _to_local_xy(lat: float, lon: float, ref_lat: float) -> tuple[float, float]:
    """Cheap equirectangular projection to meters, accurate enough over a few km."""
    import math
    x = math.radians(lon) * EARTH_RADIUS_M * math.cos(math.radians(ref_lat))
    y = math.radians(lat) * EARTH_RADIUS_M
    return x, y


def find_proximity_matches(pois_by_name: dict[str, list[dict]], threshold_m: float = 100.0) -> list[dict]:
    """
    Find POI pairs from two *different* names that fall within `threshold_m`
    of each other, using a KD-tree per name for fast nearest-neighbor lookup
    instead of checking every pair by brute force.

    A close pair (e.g. "Starbucks" 40m from "McDonald's") is a much stronger
    location signature than either brand alone, since that spatial
    arrangement is comparatively rare.

    Returns a list of {"name_a", "poi_a", "name_b", "poi_b", "distance_m"}
    dicts, sorted by distance ascending.
    """
    names = [n for n, pois in pois_by_name.items() if pois]
    if len(names) < 2:
        return []

    ref_lat = next(iter(pois_by_name.values()))[0]["lat"]
    trees = {}
    for name in names:
        pois = pois_by_name[name]
        coords = [_to_local_xy(p["lat"], p["lon"], ref_lat) for p in pois]
        trees[name] = (cKDTree(coords), pois)

    matches = []
    for name_a, name_b in combinations(names, 2):
        tree_a, pois_a = trees[name_a]
        tree_b, pois_b = trees[name_b]
        pair_indices = tree_a.query_ball_tree(tree_b, r=threshold_m)
        for i, js in enumerate(pair_indices):
            for j in js:
                poi_a, poi_b = pois_a[i], pois_b[j]
                xa, ya = _to_local_xy(poi_a["lat"], poi_a["lon"], ref_lat)
                xb, yb = _to_local_xy(poi_b["lat"], poi_b["lon"], ref_lat)
                distance_m = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
                matches.append({
                    "name_a": name_a,
                    "poi_a": poi_a,
                    "name_b": name_b,
                    "poi_b": poi_b,
                    "distance_m": distance_m,
                })

    matches.sort(key=lambda m: m["distance_m"])
    return matches
