import cv2
import numpy as np
import math


def cluster_angles(angles: list[float], weights: list[float], tolerance_degrees: float = 10.0) -> list[dict]:
    """
    Group line angles into clusters, the same way cluster_candidates() in
    geolocate.py groups GPS points — angles within `tolerance_degrees` of a
    cluster's running weighted-average angle join that cluster, instead of
    every line being blindly averaged together.

    A real camera roll shows up as one tight, high-weight cluster of
    genuinely level lines; perspective/pitch artifacts (a rooflines that's
    level in 3D but whose 2D angle is skewed by the camera looking up at it)
    land in their own small, separate clusters instead of dragging the
    average off toward nonsense.

    Returns clusters sorted by total weight descending, each a dict:
    {"angles": [...], "weights": [...], "angle": weighted_mean, "weight": total}.
    """
    items = sorted(zip(angles, weights), key=lambda item: item[1], reverse=True)
    clusters: list[dict] = []

    for angle, weight in items:
        for cluster in clusters:
            if abs(angle - cluster["angle"]) <= tolerance_degrees:
                cluster["angles"].append(angle)
                cluster["weights"].append(weight)
                total = cluster["weight"] + weight
                cluster["angle"] = (cluster["angle"] * cluster["weight"] + angle * weight) / total
                cluster["weight"] = total
                break
        else:
            clusters.append({"angles": [angle], "weights": [weight], "angle": angle, "weight": weight})

    clusters.sort(key=lambda c: c["weight"], reverse=True)
    return clusters

img = cv2.imread("bankier_032071323.webp")          # loads as a numpy array, shape (height, width, 3)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # shape (height, width) — one brightness value per pixel
edges = cv2.Canny(gray, threshold1=50, threshold2=150)

cv2.imwrite("edges_debug.jpg", edges)    # save it so you can actually look at the result



lines = cv2.HoughLinesP(
    edges,
    rho=1,              # distance resolution, in pixels
    theta=np.pi / 180,  # angle resolution, in radians (1 degree here)
    threshold=50,       # minimum votes for a line to count
    minLineLength=100,   # discard segments shorter than this (pixels)
    maxLineGap=5,       # max gap to still treat two segments as one line
)
print(f"found {len(lines)} line segment(s)")

vis = img.copy()
angles = []
weights = []

for line in lines:
    x1, y1, x2, y2 = line.flatten()   # each line is wrapped in its own array, shape (1, 4)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180

    if abs(angle) < 45:        # discard near-vertical lines — building corners,
        length = math.hypot(x2 - x1, y2 - y1)   # poles, railings aren't horizon candidates
        angles.append(angle)
        weights.append(length)
        color = (0, 255, 0)    # green = horizon candidate (BGR order!)
    else:
        color = (0, 0, 255)    # red = rejected (near-vertical)

    cv2.line(vis, (x1, y1), (x2, y2), color, 2)

cv2.imwrite("lines_debug.jpg", vis)

print(list(zip(angles, [round(w, 1) for w in weights])))

clusters = cluster_angles(angles, weights)
for i, c in enumerate(clusters, start=1):
    print(f"cluster {i}: angle={c['angle']:.2f} deg, weight={c['weight']:.1f}, n={len(c['angles'])}")

roll_angle = clusters[0]["angle"]
print(f"\nestimated roll angle: {roll_angle:.2f} degrees")

# Draw the estimated roll line through the image center, full width, so we can
# visually judge whether it actually tracks something level in the photo.
h, w = img.shape[:2]
cx, cy = w / 2, h / 2
dy = w * math.tan(math.radians(roll_angle))
pt1 = (0, int(cy - dy / 2))
pt2 = (w, int(cy + dy / 2))

overlay = img.copy()
cv2.line(overlay, pt1, pt2, (0, 255, 255), 3)  # yellow, BGR order
cv2.imwrite("roll_estimate_debug.jpg", overlay)
