# Backend plan

How the backend will be built. The Android app comes after this. Until then, a video file on the laptop is enough to get a verdict.

The backend is four parts, built in this order. Parts 1 and 2 have no URLs. The APIs are Part 3. They call Part 1 and Part 2. Part 4 is a script, not an API.

```
backend/
  config/thresholds.yaml       # every tunable number
  config/phones.yaml           # focal length / field of view by phone model
  boxcheck/                    # Part 1 — the checker, no FastAPI
    pipeline.py                # video in, verdict out
    quality.py                 # brightness, blur, texture, movement
    features.py                # detect and match marks
    geometry.py                # homography, fundamental matrix, GRIC, second plane, parallax
    scoring.py                 # blend pairs into one verdict
    debug_draw.py              # marked-up frames
    types.py
  app/                         # Part 3 — the server
    main.py
    schemas.py
    store.py                   # Part 2 — folders + SQLite
    web/index.html             # laptop upload page
  scripts/evaluate.py          # Part 4 — the scorecard
  tests/test_geometry.py       # fake flat sheet and fake box
  data/
    uploads/                   # original videos
    results/                   # one JSON per test
    debug/                     # marked-up frames
    dataset/
      tuning/                  # used while changing thresholds
      test/                    # used only to check those thresholds
        real/
        flat_printout/
        folded_printout/
        screen/
        bad_recording/
        other/
```

Libraries: Python 3.11+, FastAPI, uvicorn, python-multipart, PyYAML, NumPy, opencv-contrib-python-headless (for SIFT), SQLite from the Python standard library, pytest.

One settings file holds every threshold and weight. Nothing in the checker is hardcoded that we expect to tune.

## Part 1 — The checker

The brain. A function takes a video path plus the label and phone model, and returns the result. FastAPI is not imported here.

1. **Settings.** Load `thresholds.yaml` and `phones.yaml`.
2. **Frames.** Open the video (reject anything over 12 seconds). Take about 4 frames per second. Enlarge the working image so the long side is up to 1080p, even when the saved video is 720p. Boost contrast with CLAHE before looking for marks, so the logo and label survive on a white box.
3. **Quality gates.** Drop a frame that is too dark, too bright, or too blurry (Laplacian variance). If too few frames remain, stop with `insufficient` and the reason: too dark, too bright, or too blurry.
4. **Center only.** Keep matches inside a central region of interest. The box is supposed to fill that area.
5. **Marks.** Detect features with SIFT (AKAZE or ORB stay selectable in the settings file). Lowe ratio test drops ambiguous matches.
6. **Pairs.** Pair frames about 0.6–1.5 seconds apart. Skip a pair with too few matches. If the whole video moved too little, stop with `insufficient`: "Not enough movement — move around the box." If there are too few marks, stop with `insufficient`: "Not enough detail on the box — include the printed side and label."
7. **Three signals on each pair.**
   - **Flatness (GRIC).** Fit a homography (one plane, 8 degrees of freedom) and a fundamental matrix (general 3D, 7 degrees of freedom) with RANSAC. Score each with GRIC, using symmetric transfer error for the homography and Sampson error for the fundamental matrix. The lower GRIC wins. Homography winning means one flat surface explains the motion.
   - **Second plane.** Remove plane-1 inliers and fit a second homography. Decompose both with the camera matrix and compare plane normals. A second plane counts when it has enough points and the normals differ by at least 25°.
   - **Box-corner check.** Plane 1 and plane 2 must meet along an edge inside the central region. Two planes that only show up in the background do not count.
   - **Parallax.** Median homography transfer error on points that fit the fundamental matrix and miss the homography. A flat surface sits near the noise floor (about 1–2 px). A box leaves a leftover of several pixels.
8. **Lens.** Prefer focal length from the video metadata. If it is missing, use `phones.yaml` for that phone model. If the model is unknown, use the default field of view (about 64°).
9. **Pure rotation rule.** If the homography wins but measured movement or parallax is low, return `insufficient`, never `flat`.
10. **Score.** For each valid pair, blend the three signals with weights from the settings file (starting point 0.40 GRIC, 0.35 second plane, 0.25 parallax). The video score is the median of the pair scores. Confidence is how tightly the pairs agree.
    - High score and the pairs agree → `3d`
    - Low score, no real second plane, and enough movement → `flat`
    - Otherwise → `insufficient`, with a reason to re-record
11. **Debug.** Save a few annotated frames: plane-1 inliers in one colour, plane-2 inliers in another, outliers in a third. Save the per-pair metrics table.
12. **Synthetic tests.** A projected flat plane must come back flat. Two faces of a box must come back 3D. Too few points must come back insufficient. These tests do not need a real video.

`insufficient` is decided here. `review` is not. That rule lives in Part 3.

## Part 2 — Save and repeat

1. Store the original video under `data/uploads/{id}`.
2. Store the full result as `data/results/{id}.json` (verdict, score, confidence, reason, metrics, per-pair table, processing time, label, phone model).
3. Store debug frames under `data/debug/{id}/`.
4. Index each test in SQLite: id, time, label, phone model, attempt number, verdict, score, and file paths.
5. Re-run reads the saved video, runs Part 1 again with the current settings, and writes a new result. The original video stays.

## Part 3 — The server (the APIs)

The detailed, step-by-step implementation plan for this part is in
[`PART3_API_IMPLEMENTATION_PLAN.md`](PART3_API_IMPLEMENTATION_PLAN.md).

FastAPI on the laptop, listening on the local network so the phone can reach it over the same Wi-Fi. The upload request stays open and the verdict comes back in that same response. No job id and no polling.

The phone sends an attempt number. The checker never returns `review`. The server does: if the checker says `insufficient` and this is attempt 2 or later, the stored verdict is `review`. `insufficient` is never treated as a pass.

| API | What it does | Uses |
|---|---|---|
| `POST /api/v1/analyze` | Multipart upload: video, label, phone model, attempt number. Returns the result in a few seconds. | Part 1, Part 2 |
| `GET /api/v1/analyses` | List past tests. | Part 2 |
| `GET /api/v1/analyses/{id}` | One result: verdict, score, confidence, reason, metrics, pairs. | Part 2 |
| `GET /api/v1/analyses/{id}/debug/{n}.jpg` | One marked-up frame. | Part 2 |
| `POST /api/v1/analyses/{id}/rerun` | Same video, current settings file. | Part 1, Part 2 |
| `GET /` | Laptop page: upload a video, see the verdict and the debug frames. Calls the same upload API. | Part 3 |

Labels: `real`, `flat_printout`, `folded_printout`, `screen`, `bad_recording`, `other`. A missing label is allowed on the laptop page and stored as unlabelled. The scorecard skips those.

Result shape:

```json
{
  "id": "uuid",
  "verdict": "3d",
  "score": 0.81,
  "confidence": 0.74,
  "reason": "Two surfaces at different angles, with parallax between them.",
  "metrics": {
    "frames_sampled": 32,
    "frames_used": 27,
    "pairs_used": 14,
    "median_matches": 86,
    "median_movement_px": 41,
    "median_h_inlier_ratio": 0.62,
    "median_f_inlier_ratio": 0.79,
    "second_plane_pair_fraction": 0.64,
    "median_parallax_px": 6.1,
    "median_brightness": 124,
    "median_blur": 91
  },
  "pairs": [],
  "processing_ms": 2800,
  "label": "real",
  "phone_model": "Pixel 7a",
  "attempt": 1,
  "debug_frames": ["/api/v1/analyses/{id}/debug/0.jpg"]
}
```

`verdict` is one of `3d`, `flat`, `insufficient`, `review`.

## Part 4 — The scorecard

`scripts/evaluate.py` calls the checker directly. It does not go through the APIs.

1. Read videos from `data/dataset/tuning/` or `data/dataset/test/`. Each category is a folder: `real`, `flat_printout`, `folded_printout`, `screen`, `bad_recording`, `other`. Phone model comes from the result record or the filename.
2. Print, per category and per phone model: counts of `3d`, `flat`, `insufficient`, and `review`.
3. False rejects: real boxes marked `flat`. False accepts: flat printouts, folded printouts, and screens marked `3d`. `other` marked `3d` counts as correct. `insufficient` is reported on its own and is not a pass.
4. Tune thresholds on the tuning set only. Run the test set after a settings change to see if the result still holds.

## Done when

- A video uploaded on the laptop page returns a verdict, a reason, metrics, and debug frames within a few seconds.
- A dark, blurry, or barely moving video returns `insufficient` with a useful reason.
- A second insufficient attempt is stored as `review`.
- The same video can be re-run after editing `thresholds.yaml`.
- The scorecard prints per category and per phone model, on a tuning set and a test set kept apart.
- The checker module imports no FastAPI code.
