# Project: 3D Box Verification — Proof of Concept

## Why this project exists

We are building a sales app for an Apple phone distributor. When a salesperson
audits stock or a shop records a sale, the user records a short video of the
iPhone box. We need to prove the video shows a REAL physical 3D box — not a
printed photo, a picture on a phone/laptop screen, or any other flat image.

Before adding this to the main product, we want a SEPARATE, standalone test
project to prove that a "3D geometry check" works reliably on real videos.
If it works here, we will move the backend logic and capture approach into
the main app.

This project is ONLY about the 3D check. No login, no IMEI/ERP logic,
no business features.

## The core idea (what the 3D check does)

The user moves the phone slowly around a box placed on a counter
(~8 seconds, covering front → side → top).

- A REAL box has several surfaces at different angles. As the camera moves,
  points on different surfaces move differently across frames.
- A FLAT fake (printout, screen, photo) is a single plane. However the camera
  moves, all its points move together in a way that one flat-surface
  transformation (a homography) explains.

So the backend:
1. Extracts frames from the video.
2. Finds and tracks feature points across frames (e.g. ORB/SIFT/optical flow
   with OpenCV).
3. Compares how well the movement is explained by:
   - a single plane (homography) → suggests FLAT / fake
   - a general 3D scene (fundamental/essential matrix, or multiple planes)
     → suggests REAL 3D object
4. Also checks there was enough camera movement and enough trackable points
   (otherwise the result is "insufficient — re-record", not pass/fail).
5. Returns a verdict with a score and a human-readable reason.

Use well-known computer vision methods; choose and justify the exact approach.
No custom-trained ML models in this project.

## What to build

### 1. Backend — Python (FastAPI + OpenCV)
- Endpoint to upload a video and get a result.
- Result shape (roughly):
  - verdict: "3d" | "flat" | "insufficient" | "review"
  - score / confidence (0–1)
  - reason in plain words (e.g. "Only one flat surface detected",
    "Not enough movement — move around the box", "Too dark / too blurry")
  - metrics used (points tracked, frames used, plane-fit ratio, movement, etc.)
  - processing time
- Debug output for tuning: a few annotated frames showing tracked points,
  which points fit the single plane vs not, and a short summary.
- All thresholds in one config file so they can be tuned without code changes.
- Store uploaded videos + results locally (folder + simple JSON/SQLite) so we
  can review and re-run them.
- A batch evaluation script: run the check over a folder of labelled videos
  and print accuracy, false rejects (real boxes marked flat) and false
  accepts (fakes marked 3D), so we can tune thresholds. Report per category
  (real, flat printout, folded printout, screen, bad recording) and per
  phone model, using separate tuning and test sets.

### 2. Frontend — mobile capture app (React Native + Expo)
The capture must match the main product so results transfer:
- Expo dev build with `react-native-vision-camera` (not Expo Go).
- Live camera, Record button, ~8 s auto-stop, no audio, ~720p.
- On-screen guidance: "Place the box alone on the counter",
  "Move slowly around the box: front → side → top", progress bar.
- After recording: upload to the backend, show the verdict, score, reason
  and metrics on screen.
- A label picker before/after recording (Real box / Flat printout /
  Folded printout / Screen / Bad recording / Other object) so every test
  video is saved with its true label for the evaluation script.
- Send the phone model with the upload so results can be grouped by device.
- A simple history list of past tests and their results.
- Backend URL configurable (phone and laptop on the same Wi-Fi is fine).

### 3. Optional — simple web page
- Upload an existing video file from a laptop, see the verdict and the
  debug frames. Useful for re-testing saved videos quickly.

## What success looks like (end result)

We can:
1. Record videos on real phones (including a cheap Android) of:
   - real iPhone boxes (sealed and opened, different lighting)
   - printed photos of boxes (flat and slightly curved)
   - a box shown on another phone/laptop screen
   - other real objects (a random carton) — these SHOULD pass as 3D;
     that's expected, the main product handles them with other checks
2. See the verdict on the phone within a few seconds.
3. Run the evaluation script on all labelled videos and see:
   - real boxes → "3d" in the vast majority of cases
   - printouts and screens → "flat" in the vast majority of cases
   - bad recordings → "insufficient" with a useful reason
4. Tune thresholds in the config file and re-run until results are good.
5. Know clearly what works, what doesn't, and the limits (e.g. very
   low-texture white boxes, glare, too little movement).

## Constraints and notes
- Keep it simple and well-structured; this is a proof of concept, but the
  backend check should be clean enough to move into production later
  (separate the core check into its own module with no FastAPI dependency).
- iPhone boxes are mostly white, glossy and low-texture — the approach must
  cope with fewer feature points (use the printed image, logo and label
  areas; consider recommending how the user should film).
- Processing should take a few seconds per video on a normal server/laptop.
- Write a README: how to run backend, how to run the dev build on a phone,
  how to record test sets, how to run the evaluation, how to tune.

## The 3D check — required method (use this unless you have a clearly better reason)

Quality gates (run first; if they fail → "insufficient" with the reason):
- Brightness and blur per frame (e.g. Laplacian variance); drop bad frames.
- Enough texture: minimum number of feature points on the object.
- Enough movement: median point movement across the video above a minimum.

Frame selection:
- Sample frames (e.g. 3–5 per second). Build PAIRS of frames that are far
  enough apart (wide baseline) — tiny camera movement can't reveal 3D.

Per frame pair:
1. Detect and match features (ORB or AKAZE; SIFT if available), with a ratio
   test to remove bad matches. Focus on the central object region if possible.
2. Fit a HOMOGRAPHY with RANSAC (one-plane model) → inlier ratio H.
3. Fit a FUNDAMENTAL MATRIX with RANSAC (general 3D model) → inlier ratio F.
4. Flatness signal: if H explains almost all matches as well as F does, the
   scene is likely flat. Use a model-selection measure (e.g. GRIC) or the
   H/F inlier ratio with a tuned threshold.
5. Multi-plane check: after fitting the first plane, remove its inliers and
   try fitting a SECOND plane on the remaining points. A real box should show
   at least two planes with clearly different orientations; a flat fake
   should not have a meaningful second plane.
6. Parallax check: measure how far points deviate from the homography
   prediction (residual parallax). Real 3D → consistent non-zero parallax;
   flat → near zero.

Aggregate across all pairs:
- Combine the signals (flatness, second plane found, parallax) into one
  score with weights from the config file.
- verdict = "3d" / "flat" / "insufficient" / "review", plus the main reason.

Debug output for each video:
- Annotated frames: matches, plane-1 inliers in one colour, plane-2 in
  another, outliers in a third.
- Per-pair metrics table so thresholds can be tuned.

Explain your final method in plain words in the README.

### Risk mitigations to build in
- Pre-process frames with contrast enhancement (CLAHE) before feature
  detection; analyse at up to 1080p even if the stored video is smaller.
- Use only the central region of interest; require plane 1 and plane 2 to
  meet along an edge inside that region (box-corner check), not anywhere
  in the frame.
- If the homography wins but measured movement/parallax is low, return
  "insufficient" (possible pure rotation), never "flat".
- "insufficient" must never count as a pass; after 2 attempts the result
  is "review".
- Small lookup table of focal length / field of view for common phone
  models; prefer video metadata when present.
- Evaluation script reports results per category (real, flat printout,
  folded printout, screen, bad recording) and per phone model, using
  separate tuning and test sets.

## How I want you to start
1. Read this, then propose: the exact 3D-check method (and why), project
   structure, libraries, API shape, and a short build plan in phases.
2. List risks and open questions.
3. Wait for my approval before writing code.
