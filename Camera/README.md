# Camera tools

Install the dependencies in your Python environment:

```powershell
python -m pip install opencv-python numpy
```

## Take photos

From the Capstone root:

```powershell
python .\Camera\take_photo.py
```

The camera and preview use 1280 x 800. Click the preview and press `s` to save
a timestamped PNG in `Camera/photos`. Press `q` or Esc to quit. Use `-i 1` to
select camera 1.

## Detect racquet holes

Run detection on the live Arducam feed:

```powershell
python .\Camera\detect_holes.py --show-window
```

With no image/folder argument, the detector now opens camera 0 through DirectShow
at 1280 x 800, requesting 30 capture FPS. The preview is 1280 x 800; the displayed
processing FPS depends on detection speed. Results are recomputed for every frame.

- `q`, Esc, or closing the preview stops detection and releases the camera.
- `s` saves the current raw photo, annotated photo and JSON measurements to
  `Camera/live_results` (or `--output-dir`). The live feed is not saved continuously.
- Use `-i 1` to select camera 1. Close other apps that are using the same camera.
- Omit `--show-window` to run with a terminal update about once per second;
  Ctrl+C stops it. There are no snapshot hotkeys in this mode.

The explicit equivalent is:

```powershell
python .\Camera\detect_holes.py --camera -i 0 --show-window
```

Process the saved photos instead:

```powershell
python .\Camera\detect_holes.py .\Camera\photos --show-window
```

For saved photos, any key advances to the next result; `q`, Esc, or closing the
window exits the viewer. All photos are processed and saved before that viewer
opens. Omit `--show-window` to process the photos without opening a window.

Process a particular photo or another folder:

```powershell
python .\Camera\detect_holes.py ".\Camera\photos\photo_2026-11-10_11-41-52_020243.png" --show-window
python .\Camera\detect_holes.py "path\to\photos" --output-dir ".\Camera\other_results"
```

### Outputs

For saved-photo processing, `Camera/results` contains an annotated PNG for each input, an `overview.jpg`
contact sheet, and `results.json` with measurements in the original image pixels.
Re-running with the same filenames replaces those result files; source photos
are unchanged.

- Green: estimated outline of the visible frame segment.
- Yellow: detected openings and their centers.
- Red: detected opening closest to the image center.
- White cross: image center.

Each JSON result includes the holes' centers, ellipse axes, distances and offsets,
plus `nearest_hole_id`. Positive `dx` is right, and positive `dy` is down. Hole IDs
are assigned separately in each image or live frame; they are not physical hole identities.
`nearest_hole_id` is null if no reliable row is found.

### Processing

1. Normalize image size for processing; keep output coordinates at source size.
2. Apply light Gaussian smoothing and CLAHE local contrast enhancement. Normalize
   the raw threshold pass using its 1st/99th brightness percentiles when enough
   contrast is present; keep contrast checks on the source grayscale pixels.
3. Generate dark-opening candidates with local adaptive thresholds at two scales,
   including a threshold on the contrast-normalized grayscale image.
4. Fit ellipses and check size, shape, and interior-to-surrounding contrast. Merge
   duplicates and require agreement from at least two threshold passes. Weak,
   irregular candidates need support from the raw and smaller-neighbourhood
   passes to reject shadows between grommets.
5. Use RANSAC to find a smooth local curve through at least four candidates,
   rejecting outliers and nearby duplicate detections. Check size consistency
   along the row and refine the curve using weighted fitting.
6. Estimate frame boundaries from gradients in the gaps between grommets. A
   broad frame-thickness prior helps separate the outer edge from reflections.
7. Choose the accepted opening with the smallest Euclidean distance to image center.

### Scope of this first version

This detector handles a visible frame segment with at least four
openings. It was tuned on the 19 supplied photos of this racquet. The default
opening-diameter range is now 18-70 pixels at a normalized image length of 1280,
covering the smaller openings in the more distant views. The frame
outline is an estimate supported by the hole row and image edges; it is not a
trained racquet classifier. For this racquet, the boundary search expects a frame
band roughly 3-5 opening diameters thick.

All 19 supplied photos produced a visually plausible nearest target, with 4-10
openings detected per photo. The more distant `15-23-36` photo previously produced
no target and now produces eight openings. The changes also reject two shadows
that appeared when only the minimum size was reduced.

`validation_labels.json` records approximate, manually reviewed nearest-hole
centers and the two known non-hole locations. The 20-pixel test tolerance checks
which opening was selected; it is not a calibrated precision measurement.
Development checks cover all 19 images, 90-degree rotation, brightness multipliers
of 0.65 and 1.3, uniform images, background crops, and the known shadows:

```powershell
python .\Camera\test_detect_holes.py
```

Place the original capture files named in `validation_labels.json` in
`Camera/photos` before running the checks. Captures and generated overlays are
stored locally and excluded from version control.

All these photos have been used for tuning. These checks do not measure accuracy
on unseen conditions or guarantee robustness to glare.
Some edge, obscured, or low-contrast openings are omitted, so the selected target
is the nearest **detected** opening. The center estimates have not been calibrated
for threading precision. Strong glare can obscure an opening entirely.

For a substantially different camera distance, adjust the candidate diameter
range (measured after normalizing the image's longest side to 1280 pixels):

```powershell
python .\Camera\detect_holes.py --min-hole-size 20 --max-hole-size 85 --show-window
```

If holes are found but the boundary evidence is weak, the JSON status is
`holes_found_frame_uncertain`; the openings and nearest target are still reported.

After changing the detector, restart the live script to load the new settings.
