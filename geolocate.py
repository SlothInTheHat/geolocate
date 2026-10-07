"""
geolocate.py

Minimal sample script for the disaster-geomapper project. For each image:

  1. Check EXIF GPS metadata first (see exif_gps.py) — if the photo still
     carries it, that's ground truth and everything below is skipped.
  2. Otherwise run GeoCLIP over a wide candidate pool and cluster the results
     (see cluster_candidates()) into distinct spatial regions. This is a
     coarse-to-fine pass: the cluster holding the most probability mass is
     GeoCLIP's real "coarse" guess, and how scattered the clusters are is a
     direct, surfaced measure of how uncertain that guess is — printed to the
     console and saved into geolocated_points.json as spread_km/num_clusters
     for mapvis.py to visualize.
  3. Ensemble the top members of that dominant cluster with scene text (see
     ocr_match.py): brand/place names OCR'd from the photo are cross-
     referenced against OpenStreetMap, and candidates where two different
     names turn out to sit near each other in real life get boosted, since
     that spatial pairing is a much rarer signature than either name alone.

All resulting candidates per image are saved to a JSON file so mapviz.py can
plot the full spread, not just a single point.

Setup:
    pip install geoclip easyocr scipy requests pillow

Usage:
    python geolocate.py path/to/image.jpg [more/images.jpg ...]
    python geolocate.py --no-refine path/to/image.jpg   # skip OCR/Overpass stage
"""

import json
import math
import os
import sys
from geoclip import GeoCLIP

from exif_gps import get_exif_gps

POINTS_OUTPUT_PATH = "geolocated_points.json"


def geolocate_image(image_path: str, top_k: int = 5, model: "GeoCLIP | None" = None):
    """Run GeoCLIP on an image and return top-k (lat, lon) guesses with probabilities."""
    model = model or GeoCLIP()
    top_pred_gps, top_pred_prob = model.predict(image_path, top_k=top_k)
    return top_pred_gps, top_pred_prob


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    earth_radius_km = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def cluster_candidates(coords: list[tuple[float, float]], probs: list[float], radius_km: float = 50.0) -> list[dict]:
    """
    Greedily group GeoCLIP's candidate GPS points into spatial clusters.

    This is the coarse-to-fine step: rather than trusting one point in
    isolation, points landing within `radius_km` of each other are treated as
    votes for the same rough region. The cluster with the most combined
    probability mass is GeoCLIP's real "coarse" guess. How many clusters
    exist, and how far apart they are, is a direct measure of how uncertain
    the model actually is — a single tight cluster means GeoCLIP is
    confident about the area; several far-apart clusters mean it isn't.

    Returns clusters sorted by total probability mass descending, each as
    {"members": [((lat, lon), prob), ...], "centroid": (lat, lon), "mass": float}.
    """
    items = sorted(zip(coords, probs), key=lambda item: item[1], reverse=True)
    clusters: list[dict] = []

    for (lat, lon), prob in items:
        for cluster in clusters:
            c_lat, c_lon = cluster["centroid"]
            if _haversine_km(lat, lon, c_lat, c_lon) <= radius_km:
                cluster["members"].append(((lat, lon), prob))
                total = cluster["mass"] + prob
                cluster["centroid"] = (
                    (c_lat * cluster["mass"] + lat * prob) / total,
                    (c_lon * cluster["mass"] + lon * prob) / total,
                )
                cluster["mass"] = total
                break
        else:
            clusters.append({"members": [((lat, lon), prob)], "centroid": (lat, lon), "mass": prob})

    clusters.sort(key=lambda c: c["mass"], reverse=True)
    return clusters


def _cluster_spread_km(cluster: dict) -> float:
    """Max distance from a cluster's centroid to any of its member points."""
    c_lat, c_lon = cluster["centroid"]
    return max((_haversine_km(c_lat, c_lon, lat, lon) for (lat, lon), _ in cluster["members"]), default=0.0)


def geolocate_images(image_paths: list[str], top_k: int = 5) -> list[dict]:
    """
    Run GeoCLIP on each image and return all top-k guesses for each image,
    in the format mapviz.create_map() expects: {"lat", "lon", "label"}.
    """
    model = GeoCLIP()
    points = []
    for image_path in image_paths:
        coords, probs = geolocate_image(image_path, top_k=top_k, model=model)
        name = os.path.basename(image_path)
        for rank, ((lat, lon), prob) in enumerate(zip(coords, probs), start=1):
            points.append({
                "lat": float(lat),
                "lon": float(lon),
                "label": f"{name} #{rank} (confidence={prob:.4f})",
            })
    return points


def save_points_json(points: list[dict], output_path: str = POINTS_OUTPUT_PATH) -> None:
    """Save geolocated points to a JSON file for mapviz.py to consume."""
    with open(output_path, "w") as f:
        json.dump(points, f, indent=2)


def refine_with_scene_text(
    image_path: str,
    coords,
    probs,
    texts: list[str] | None = None,
    radius_km: float = 5.0,
    threshold_m: float = 100.0,
    verbose: bool = True,
) -> list[tuple[tuple[float, float], float]]:
    """
    Re-rank GeoCLIP's top-k GPS guesses using scene text spotted in the photo.

    OCRs the image for brand/place names (unless `texts` is already supplied,
    to avoid re-running OCR when the caller needs the same text elsewhere too),
    then for each candidate location looks up matching OSM POIs within
    `radius_km` and checks whether two different detected names sit within
    `threshold_m` of each other there (e.g. a photographed "Starbucks"
    genuinely next to a "Chase"). That kind of spatial pairing is a much
    rarer signature than either name alone, so matching candidates get their
    score boosted.

    Falls back to the unmodified GeoCLIP ranking if OCR finds fewer than two
    usable names, or if Overpass can't be reached (offline, rate-limited, etc).

    Returns a list of ((lat, lon), score) sorted by score descending.
    """
    from ocr_match import query_overpass_pois, find_proximity_matches

    ranked = list(zip(coords, probs))

    if texts is None:
        from ocr_match import extract_text_candidates
        try:
            texts = extract_text_candidates(image_path)
        except Exception as e:
            if verbose:
                print(f"  (scene-text OCR skipped: {e})")
            return ranked

    if len(texts) < 2:
        if verbose:
            print("  (fewer than 2 named-place candidates spotted in image; skipping proximity refinement)")
        return ranked

    if verbose:
        print(f"  Scene text candidates: {texts}")

    scored = []
    for (lat, lon), prob in ranked:
        try:
            pois_by_name = {
                name: query_overpass_pois(name, center=(lat, lon), radius_km=radius_km)
                for name in texts
            }
            matches = find_proximity_matches(pois_by_name, threshold_m=threshold_m)
        except Exception as e:
            if verbose:
                print(f"  (Overpass lookup failed near {lat:.5f},{lon:.5f}: {e})")
            matches = []

        if matches:
            closest = matches[0]
            boost = 1.0 + max(0.0, 1.0 - closest["distance_m"] / threshold_m)
            if verbose:
                print(
                    f"  Match near {lat:.5f},{lon:.5f}: {closest['name_a']} <-> {closest['name_b']} "
                    f"({closest['distance_m']:.0f}m) -> boost x{boost:.2f}"
                )
        else:
            boost = 1.0

        scored.append(((lat, lon), prob * boost))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def analyze_image(
    image_path: str,
    model: "GeoCLIP",
    coarse_k: int = 25,
    top_k: int = 5,
    cluster_radius_km: float = 50.0,
    do_refine: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Full per-image geolocation pipeline: EXIF check, then coarse-to-fine
    GeoCLIP clustering, then OCR/Overpass ensemble refinement — including an
    expanding-radius brand search that can propose a location far outside
    GeoCLIP's own guesses (see ocr_match.find_brand_location_candidates()).

    Returns:
        {
            "source": "exif" | "geoclip",
            "candidates": [{"coord": (lat, lon), "score": float, "origin": str,
                             "matched_text"?: str, "radius_km"?: float}, ...]
                sorted best-first. `origin` is "exif", "geoclip", or
                "ocr_brand" (a candidate proposed by the brand search rather
                than by GeoCLIP itself).
            "spread_km": float,   # 0.0 for EXIF; dominant cluster's radius otherwise
            "num_clusters": int,  # 1 for EXIF; how many distinct regions GeoCLIP proposed
        }
    """
    exif_coords = get_exif_gps(image_path)
    if exif_coords is not None:
        if verbose:
            print(f"  Found EXIF GPS: {exif_coords[0]:.5f}, {exif_coords[1]:.5f} (skipping GeoCLIP)")
        candidates = [{"coord": exif_coords, "score": 1.0, "origin": "exif"}]
        return {"source": "exif", "candidates": candidates, "spread_km": 0.0, "num_clusters": 1}

    coords, probs = geolocate_image(image_path, top_k=coarse_k, model=model)
    coords = [(float(lat), float(lon)) for lat, lon in coords]
    probs = [float(p) for p in probs]

    clusters = cluster_candidates(coords, probs, radius_km=cluster_radius_km)
    dominant = clusters[0]
    spread_km = _cluster_spread_km(dominant)

    if verbose:
        print(
            f"  GeoCLIP: {len(clusters)} distinct region(s) found; dominant region holds "
            f"{dominant['mass']:.1%} of probability mass, spread ~{spread_km:.1f}km"
        )
        if len(clusters) > 1:
            print("  (multiple far-apart regions proposed - treat this guess with caution)")

    fine = sorted(dominant["members"], key=lambda m: m[1], reverse=True)[:top_k]
    fine_coords = [pt for pt, _ in fine]
    fine_probs = [p for _, p in fine]

    if do_refine:
        from ocr_match import extract_text_candidates, find_brand_location_candidates

        try:
            texts = extract_text_candidates(image_path)
        except Exception as e:
            if verbose:
                print(f"  (scene-text OCR skipped: {e})")
            texts = []

        ranked = refine_with_scene_text(image_path, fine_coords, fine_probs, texts=texts, verbose=verbose)
        candidates = [{"coord": coord, "score": score, "origin": "geoclip"} for coord, score in ranked]

        if texts:
            brand_hits = find_brand_location_candidates(texts, anchor=dominant["centroid"], verbose=verbose)
            for hit in brand_hits:
                # A confirmed brand+location match is direct real-world evidence,
                # not just a visual-embedding proxy like GeoCLIP's score — so it's
                # scored on its own scale (well above GeoCLIP's typically tiny
                # top-k probabilities) and can outrank the model's own top guess.
                # Matches found closer in (smaller radius) score higher.
                radius_km = hit["radius_km"]
                score = 0.5 * (1.0 - min(radius_km, 1000.0) / 1000.0 * 0.4)
                candidates.append({
                    "coord": (hit["lat"], hit["lon"]),
                    "score": score,
                    "origin": "ocr_brand",
                    "matched_text": hit["matched_text"],
                    "radius_km": radius_km,
                })
            candidates.sort(key=lambda c: c["score"], reverse=True)
            candidates = candidates[:top_k]
    else:
        candidates = [{"coord": coord, "score": prob, "origin": "geoclip"} for coord, prob in zip(fine_coords, fine_probs)]

    return {"source": "geoclip", "candidates": candidates, "spread_km": spread_km, "num_clusters": len(clusters)}


def main():
    args = [a for a in sys.argv[1:] if a != "--no-refine"]
    do_refine = "--no-refine" not in sys.argv

    if not args:
        print("Usage: python geolocate.py [--no-refine] <image_path> [more_images...]")
        sys.exit(1)

    image_paths = args
    model = GeoCLIP()
    points = []

    for image_path in image_paths:
        print(f"\n{image_path}")
        result = analyze_image(image_path, model=model, do_refine=do_refine)

        name = os.path.basename(image_path)
        for rank, cand in enumerate(result["candidates"], start=1):
            lat, lon = cand["coord"]
            score, origin = cand["score"], cand["origin"]
            tag = "best" if rank == 1 else f"#{rank}"
            extra = f", matched '{cand['matched_text']}' ~{cand['radius_km']:.0f}km out" if origin == "ocr_brand" else ""
            print(f"  {tag} [{origin}]: lat={lat:.5f}, lon={lon:.5f}  (score={score:.4f}{extra})")

            point = {
                "lat": float(lat),
                "lon": float(lon),
                "label": f"{name} ({tag}, {origin}, score={score:.4f})",
                "kind": "best" if rank == 1 else "alt",
                "source": result["source"],
                "origin": origin,
                "spread_km": result["spread_km"],
                "num_clusters": result["num_clusters"],
            }
            if origin == "ocr_brand":
                point["matched_text"] = cand["matched_text"]
                point["radius_km"] = cand["radius_km"]
            points.append(point)

    save_points_json(points)
    print(f"\nSaved {len(points)} point(s) to {POINTS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()