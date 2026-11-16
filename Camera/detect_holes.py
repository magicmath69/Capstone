"""Detect racquet openings from the live Arducam, a photo, or a folder."""

import argparse
from datetime import datetime
from itertools import combinations
import json
from pathlib import Path
import time

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent


def hole_candidate(contour, gray, min_size, max_size):
    """Check shape and dark-interior/bright-surround contrast of an opening."""
    area = cv2.contourArea(contour)
    if len(contour) < 5 or area < min_size**2 * 0.35:
        return None
    ellipse = cv2.fitEllipse(contour)
    (x, y), (a, b), angle = ellipse
    small, large = sorted((a, b))
    diameter = np.sqrt(a * b)
    if not (min_size <= diameter <= max_size and small / large >= 0.38):
        return None
    if small < min_size * 0.55 or large > max_size * 1.4:
        return None
    if not (large / 2 < x < gray.shape[1] - large / 2
            and large / 2 < y < gray.shape[0] - large / 2):
        return None  # Do not estimate centers from openings clipped by the image.
    fill = area / (np.pi * a * b / 4)
    if not 0.65 < fill < 1.2:
        return None

    theta = np.deg2rad(angle)
    cosine, sine = np.cos(theta), np.sin(theta)
    points = contour[:, 0, :].astype(float) - (x, y)
    u = points[:, 0] * cosine + points[:, 1] * sine
    v = -points[:, 0] * sine + points[:, 1] * cosine
    error = float(np.mean(np.abs(np.sqrt((u / (a / 2))**2 + (v / (b / 2))**2) - 1)))
    if error > 0.24:
        return None

    radius = int(np.ceil(large))
    x0, x1 = max(0, int(x) - radius), min(gray.shape[1], int(x) + radius + 1)
    y0, y1 = max(0, int(y) - radius), min(gray.shape[0], int(y) + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - x, yy - y
    u, v = dx * cosine + dy * sine, -dx * sine + dy * cosine
    distance = (u / (a / 2))**2 + (v / (b / 2))**2
    pixels = gray[y0:y1, x0:x1]
    inside = pixels[distance < 0.55**2]
    surround = pixels[(distance > 1.2**2) & (distance < 1.65**2)]
    if not inside.size or not surround.size:
        return None
    contrast = float(np.median(surround) - np.mean(inside))
    if contrast < 8:
        return None
    return {"ellipse": ellipse, "diameter": float(diameter),
            "contrast": contrast, "ellipse_error": error,
            "score": contrast / (1 + 4 * error)}


def find_candidates(gray, min_size, max_size):
    """Use local thresholds at two scales; merge repeated detections."""
    raw = cv2.GaussianBlur(gray, (5, 5), 0)
    low, high = np.percentile(raw, (1, 99))
    if high - low >= 30:
        raw = np.clip((raw.astype(np.float32) - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    enhanced = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
    candidates = []
    for source, (image, block_size, offset) in enumerate(
            ((raw, 91, 3), (enhanced, 51, 3), (enhanced, 91, 7))):
        mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block_size, offset)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            candidate = hole_candidate(contour, gray, min_size, max_size)
            if candidate is not None:
                candidate["sources"] = {source}
                candidates.append(candidate)

    kept = []
    for candidate in sorted(candidates, key=lambda item: -item["score"]):
        center = np.array(candidate["ellipse"][0])
        match = next((other for other in kept if np.linalg.norm(center - other["ellipse"][0])
                      < 0.7 * min(candidate["diameter"], other["diameter"])), None)
        if match is None:
            kept.append(candidate)
        else:
            match["sources"].update(candidate["sources"])
    # Weak, irregular shapes need support from both the raw image and the
    # smaller enhanced neighbourhood, rather than two large-scale shadows.
    return [item for item in kept if len(item["sources"]) >= 2
            and ({0, 1}.issubset(item["sources"]) or item["contrast"] >= 25
                 or item["ellipse_error"] <= 0.08)][:100]


def fit_hole_row(candidates, shape):
    """RANSAC: find at least four openings following one smooth local arc."""
    if len(candidates) < 4:
        return None
    centers = np.array([item["ellipse"][0] for item in candidates])
    diameters = np.array([item["diameter"] for item in candidates])
    scores = np.array([item["score"] for item in candidates])
    length = max(shape)
    rng = np.random.default_rng(0)
    if len(candidates) <= 22:
        samples = list(combinations(range(len(candidates)), 3))
    else:
        samples = [rng.choice(len(candidates), 3, replace=False) for _ in range(1500)]
    best = None
    for axis in (0, 1):
        uv = centers if axis == 0 else centers[:, ::-1]
        normalized = uv / length
        for sample in samples:
            sample = np.array(sample)
            u, v = normalized[sample].T
            if np.min(np.diff(np.sort(u))) < np.median(diameters) / length:
                continue
            design = np.column_stack((u**2, u, np.ones(3)))
            coefficients = np.linalg.solve(design, v)
            if abs(coefficients[0]) > 0.8:
                continue
            slope = 2 * coefficients[0] * normalized[:, 0] + coefficients[1]
            if np.max(np.abs(slope)) > 1.5:
                continue
            residual = np.abs(normalized[:, 1] - np.polyval(coefficients, normalized[:, 0]))
            residual = residual * length / np.sqrt(1 + slope**2)
            inliers = residual < np.clip(diameters * 0.42, 10, 22)

            indices = np.flatnonzero(inliers)
            # Suppress adjacent detections of the same opening or its shadow.
            selected = []
            for i in indices[np.argsort(-scores[indices])]:
                if all(np.linalg.norm(centers[i] - centers[j])
                       > 1.45 * min(diameters[i], diameters[j]) for j in selected):
                    selected.append(int(i))
            if len(selected) < 4:
                continue
            span = np.ptp(uv[selected, 0])
            if span < length * 0.3:
                continue
            quality = float(np.sum(np.clip(scores[selected] / 25, 0.25, 1.5))
                            + len(selected) * 0.5 + span / length)
            if best is None or quality > best["quality"]:
                best = {"indices": selected, "axis": axis, "quality": quality}

    if best is None:
        return None
    axis = best["axis"]
    uv = centers if axis == 0 else centers[:, ::-1]
    indices = sorted(best["indices"], key=lambda i: uv[i, 0])
    # Opening sizes vary gradually along this frame. Large blobs between smaller
    # openings are usually merged shadows, even when they pass an ellipse check.
    typical_size = np.median(diameters[indices])
    indices = [i for i in indices if diameters[i] <= typical_size * 1.65]
    for _ in range(2):
        if len(indices) < 4 or np.ptp(uv[indices, 0]) < length * 0.3:
            return None
        coefficients = np.polyfit(uv[indices, 0] / length, uv[indices, 1] / length, 2,
                                  w=np.sqrt(scores[indices]))
        slope = 2 * coefficients[0] * uv[indices, 0] / length + coefficients[1]
        residual = np.abs(uv[indices, 1] - np.polyval(coefficients, uv[indices, 0] / length) * length)
        residual /= np.sqrt(1 + slope**2)
        tolerance = np.clip(np.median(diameters[indices]) * 0.42, 10, 22)
        refined = [i for i, error in zip(indices, residual) if error <= tolerance]
        if refined == indices:
            break
        indices = refined
    if len(indices) < 4 or np.ptp(uv[indices, 0]) < length * 0.3:
        return None
    coefficients = np.polyfit(uv[indices, 0] / length, uv[indices, 1] / length, 2,
                              w=np.sqrt(scores[indices]))
    return {"indices": indices, "axis": axis, "coefficients": coefficients}


def frame_outline(gray, candidates, row):
    """Estimate the visible frame band from the row and supporting image edges."""
    axis, coefficients = row["axis"], row["coefficients"]
    length = max(gray.shape)
    centers = np.array([item["ellipse"][0] for item in candidates])
    uv = centers if axis == 0 else centers[:, ::-1]
    diameter = float(np.median([item["diameter"] for item in candidates]))
    u_limit = gray.shape[1] if axis == 0 else gray.shape[0]
    u = np.linspace(max(0, uv[:, 0].min() - diameter * 2),
                    min(u_limit - 1, uv[:, 0].max() + diameter * 2), 120)
    v = np.polyval(coefficients, u / length) * length
    slope = 2 * coefficients[0] * u / length + coefficients[1]
    normal_u, normal_v = -slope / np.hypot(1, slope), 1 / np.hypot(1, slope)
    offsets = np.arange(-int(diameter * 3.5), int(diameter * 3.5) + 1)
    map_u = (u[None, :] + offsets[:, None] * normal_u).astype(np.float32)
    map_v = (v[None, :] + offsets[:, None] * normal_v).astype(np.float32)
    map_x, map_y = (map_u, map_v) if axis == 0 else (map_v, map_u)
    smoothed = cv2.GaussianBlur(gray, (7, 7), 0).astype(np.float32)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    sampled_x = cv2.remap(gx, map_x, map_y, cv2.INTER_LINEAR)
    sampled_y = cv2.remap(gy, map_x, map_y, cv2.INTER_LINEAR)
    nx, ny = (normal_u, normal_v) if axis == 0 else (normal_v, normal_u)
    strength = np.abs(sampled_x * nx + sampled_y * ny)
    # Measure frame edges in the gaps so the repeated grommet rims do not win.
    gaps = np.ones(len(u), dtype=bool)
    for center, candidate in zip(uv, candidates):
        gaps &= np.abs(u - center[0]) > max(candidate["ellipse"][1]) * 0.9
    if np.count_nonzero(gaps) < 8:
        return None
    profile = np.quantile(strength[:, gaps], 0.6, axis=1).astype(np.float32)
    profile = cv2.GaussianBlur(profile[:, None], (1, 9), 0).ravel()
    peaks = np.flatnonzero((profile[1:-1] > profile[:-2])
                           & (profile[1:-1] > profile[2:])) + 1
    upper = [i for i in peaks if offsets[i] < -diameter * 0.9 and profile[i] > 3]
    lower = [i for i in peaks if offsets[i] > diameter * 0.9 and profile[i] > 3]
    if not upper or not lower:
        return None
    pairs = [(top, bottom) for top in upper for bottom in lower
             if 3 < (offsets[bottom] - offsets[top]) / diameter < 5]
    if not pairs:
        return None
    # On this racquet, the band is approximately 3.8 opening diameters thick.
    # Pairing both edges avoids tracing an internal reflection as the bottom edge.
    top_max, bottom_max = max(profile[upper]), max(profile[lower])
    def boundary_score(pair):
        top, bottom = pair
        thickness = (offsets[bottom] - offsets[top]) / diameter
        return profile[top] / top_max + profile[bottom] / bottom_max - 2 * abs(thickness - 3.8)
    pair = max(pairs, key=boundary_score)
    boundaries = []
    for peak in pair:
        # Follow the edge near the aggregate boundary, smoothing local reflections.
        local = np.abs(offsets - offsets[peak]) <= diameter * 0.4
        local_offsets = offsets[local][np.argmax(strength[local], axis=0)].astype(np.float32)
        local_offsets = cv2.GaussianBlur(local_offsets[None, :], (21, 1), 0).ravel()
        edge = np.column_stack((u + local_offsets * normal_u, v + local_offsets * normal_v))
        boundaries.append(edge if axis == 0 else edge[:, ::-1])
    polygon = np.vstack((boundaries[0], boundaries[1][::-1]))
    polygon[:, 0] = np.clip(polygon[:, 0], 0, gray.shape[1] - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, gray.shape[0] - 1)
    return polygon


def detect(image, min_size=18, max_size=70):
    """Return measurements in original image pixels, without drawing on the input."""
    height, width = image.shape[:2]
    scale = 1280 / max(height, width)
    working = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    candidates = find_candidates(gray, min_size, max_size)
    row = fit_hole_row(candidates, gray.shape)
    result = {"image_size": [width, height], "image_center": [width / 2, height / 2],
              "status": "no_reliable_hole_row", "frame_outline": None,
              "holes": [], "nearest_hole_id": None}
    if row is None:
        return result
    holes = [candidates[i] for i in row["indices"]]
    polygon = frame_outline(gray, holes, row)
    if polygon is not None:
        result["frame_outline"] = np.round(polygon / scale, 2).tolist()
    for i, candidate in enumerate(holes, 1):
        (x, y), axes, angle = candidate["ellipse"]
        x, y = x / scale, y / scale
        dx, dy = x - width / 2, y - height / 2
        result["holes"].append({"id": i, "center": [x, y],
                                "axes": [value / scale for value in axes],
                                "angle": angle, "offset": [dx, dy],
                                "distance": float(np.hypot(dx, dy)),
                                "contrast": candidate["contrast"],
                                "ellipse_error": candidate["ellipse_error"]})
    nearest = min(result["holes"], key=lambda hole: hole["distance"])
    result["nearest_hole_id"] = nearest["id"]
    result["status"] = "detected" if polygon is not None else "holes_found_frame_uncertain"
    return result


def annotate(image, result):
    output = image.copy()
    polygon = result["frame_outline"]
    if polygon is not None:
        cv2.polylines(output, [np.rint(polygon).astype(np.int32)], True, (0, 220, 0), 2)
    center = tuple(round(value) for value in result["image_center"])
    cv2.drawMarker(output, center, (255, 255, 255), cv2.MARKER_CROSS, 22, 2)
    nearest = None
    for hole in result["holes"]:
        selected = hole["id"] == result["nearest_hole_id"]
        color = (0, 0, 255) if selected else (0, 255, 255)
        ellipse = (tuple(hole["center"]), tuple(hole["axes"]), hole["angle"])
        point = tuple(round(value) for value in hole["center"])
        cv2.ellipse(output, ellipse, color, 2)
        cv2.circle(output, point, 3, color, -1)
        cv2.putText(output, str(hole["id"]), (point[0] + 8, point[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if selected:
            nearest = hole
            cv2.line(output, center, point, color, 1)
    cv2.rectangle(output, (0, 0), (output.shape[1], 85), (25, 25, 25), -1)
    text = f"Holes: {len(result['holes'])} | {result['status']}"
    cv2.putText(output, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if nearest:
        dx, dy = nearest["offset"]
        text = f"Nearest: #{nearest['id']} | dx={dx:+.1f}, dy={dy:+.1f} px | distance={nearest['distance']:.1f} px"
    else:
        text = "No reliable target detected"
    cv2.putText(output, text, (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return output


def save_snapshot(frame, overlay, result, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("camera_%Y-%m-%d_%H-%M-%S_%f")
    raw_path = output_dir / f"{name}.png"
    overlay_path = output_dir / f"{name}_detected.png"
    if not cv2.imwrite(str(raw_path), frame) or not cv2.imwrite(str(overlay_path), overlay):
        print(f"Failed to save snapshot in {output_dir}")
        return
    saved = dict(result, source=str(raw_path.resolve()), overlay=str(overlay_path.resolve()))
    (output_dir / f"{name}.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print(f"Saved photo, overlay and measurements: {raw_path.resolve()}")


def run_camera(args):
    output_dir = args.output_dir or HERE / "live_results"
    window = "Arducam racquet holes - s: save | q / Esc: quit"
    cap = cv2.VideoCapture(args.index, cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera {args.index}. Close other camera apps or try -i 1.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if args.show_window:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1280, 800)
            print("Live detection: click the window; s saves a snapshot, q / Esc quits.", flush=True)
        else:
            print("Live detection without a window. Press Ctrl+C to quit.", flush=True)

        last_report = 0
        first_frame = True
        while True:
            started = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("Cannot read a frame from the camera. Check its connection.")
            if first_frame:
                print(f"Camera {args.index}: {frame.shape[1]} x {frame.shape[0]} (requested 30 FPS).", flush=True)
                first_frame = False
            result = detect(frame, args.min_hole_size, args.max_hole_size)
            result["camera_index"] = args.index
            fps = 1 / max(time.perf_counter() - started, 1e-6)
            if time.monotonic() - last_report >= 1:
                nearest = next((hole for hole in result["holes"]
                                if hole["id"] == result["nearest_hole_id"]), None)
                target = f"nearest #{nearest['id']}, offset {np.round(nearest['offset'], 1).tolist()} px" if nearest else "no target"
                print(f"{len(result['holes'])} holes, {target} | {fps:.1f} FPS", flush=True)
                last_report = time.monotonic()

            if args.show_window:
                overlay = annotate(frame, result)
                cv2.putText(overlay, f"Processing: {fps:.1f} FPS", (max(15, frame.shape[1] - 260), 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imshow(window, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("s"):
                    save_snapshot(frame, overlay, result, output_dir)
    except KeyboardInterrupt:
        print("\nCamera stopped.")
    finally:
        cap.release()
        if args.show_window:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="Image or folder; omit to use the live camera")
    parser.add_argument("--camera", action="store_true", help="Use the live camera (default when no input is given)")
    parser.add_argument("-i", "--index", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--output-dir", type=Path, help="Output folder (default: Camera/live_results or Camera/results)")
    parser.add_argument("--show-window", action="store_true", help="Display the live feed or annotated photos")
    parser.add_argument("--min-hole-size", type=float, default=18, help="Minimum opening diameter at 1280-pixel image length (default: 18)")
    parser.add_argument("--max-hole-size", type=float, default=70, help="Maximum opening diameter at 1280-pixel image length")
    args = parser.parse_args()
    if not 0 < args.min_hole_size < args.max_hole_size:
        parser.error("Hole sizes must satisfy 0 < min < max")
    if args.camera and args.input is not None:
        parser.error("Choose either --camera or an image/folder")
    if args.camera or args.input is None:
        run_camera(args)
        return
    args.output_dir = args.output_dir or HERE / "results"
    if args.input.is_dir():
        files = sorted(path for path in args.input.iterdir()
                       if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    elif args.input.is_file():
        files = [args.input]
    else:
        parser.error(f"Input does not exist: {args.input}")
    if not files:
        parser.error("No images found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, previews = [], []
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"Cannot read image: {path}")
        result = detect(image, args.min_hole_size, args.max_hole_size)
        result["source"] = str(path.resolve())
        overlay = annotate(image, result)
        cv2.putText(overlay, path.name, (15, overlay.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(overlay, path.name, (15, overlay.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        output_path = args.output_dir / f"{path.stem}_detected.png"
        if not cv2.imwrite(str(output_path), overlay):
            raise SystemExit(f"Cannot save overlay: {output_path}")
        result["overlay"] = str(output_path.resolve())
        results.append(result)
        previews.append(cv2.resize(overlay, (640, 400)))
        nearest = next((hole for hole in result["holes"] if hole["id"] == result["nearest_hole_id"]), None)
        target = f"nearest #{nearest['id']}, offset {np.round(nearest['offset'], 1).tolist()} px" if nearest else "no target"
        print(f"{path.name}: {len(result['holes'])} holes, {target}")

    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if len(previews) % 2:
        previews.append(np.zeros_like(previews[0]))
    overview = np.vstack([np.hstack(previews[i:i + 2]) for i in range(0, len(previews), 2)])
    if not cv2.imwrite(str(args.output_dir / "overview.jpg"), overview):
        raise SystemExit("Cannot save overview")
    print(f"Saved overlays and results.json in {args.output_dir.resolve()}")

    if args.show_window:
        window = "Racquet holes - any key: next | q / Esc: quit"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 800)
        try:
            for result in results:
                cv2.imshow(window, cv2.imread(result["overlay"]))
                while True:
                    key = cv2.waitKey(50)
                    if key != -1 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                if key & 0xFF in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
