# 3D Box Checker Backend

This backend decides whether a short video shows a physical 3D box or a flat
representation. It includes the standalone computer-vision checker, local
file/SQLite persistence, a FastAPI server, and a laptop upload page.

The architecture remains separated:

- `boxcheck/` analyzes videos and does not import FastAPI or storage code.
- `app/store.py` saves videos, result JSON, debug images, and SQLite rows.
- `app/` exposes those capabilities through HTTP and the laptop page.

## Requirements

- Python 3.11 or newer
- Windows PowerShell for the commands below
- A trusted private Wi-Fi network for phone testing
- An `.mp4` or `.mov` recording no longer than 12 seconds

Commands below assume the terminal is in the project root:
`C:\Users\Admin\Desktop\3d-geometry`.

## Install for development

Install the package and development dependencies in editable mode:

```powershell
python -m pip install -e ".\backend[dev]"
```

This installs OpenCV, FastAPI, Uvicorn, multipart upload support, and the test
dependencies. Editable mode keeps imports connected to the local source files.

## Run the tests

Run tests from the backend directory so both `app` and `boxcheck` are on the
Python import path:

```powershell
Push-Location .\backend
python -m pytest
Pop-Location
```

## Start the server

From the project root, run:

```powershell
python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

Keep that terminal open. Uvicorn initializes `backend/data/` and its SQLite
database on startup.

The `0.0.0.0` host makes the server reachable through both the laptop and its
private-network IP. It does not mean you browse to `0.0.0.0`; use one of the
addresses below.

## Verify on the laptop

Open these URLs:

- Upload page: <http://127.0.0.1:8000/>
- Interactive API documentation: <http://127.0.0.1:8000/docs>
- OpenAPI JSON: <http://127.0.0.1:8000/openapi.json>

The upload page accepts a video, optional known label and phone model, and an
attempt number. It displays the verdict, reason, score, confidence, metrics,
and annotated debug frames.

## Real-video smoke test

The browser page is the simplest test. Select a real `.mp4` or `.mov`, leave
attempt as `1`, and press **Analyze video**.

The same upload can be tested from PowerShell with Windows `curl.exe`:

```powershell
curl.exe --silent --show-error --fail-with-body `
  -F "video=@C:\path\to\box.mp4;type=video/mp4" `
  -F "label=real" `
  -F "phone_model=Pixel 7a" `
  -F "attempt=1" `
  http://127.0.0.1:8000/api/v1/analyze
```

Replace the video path and phone model. A successful request returns HTTP
`201` and JSON containing an `id`, verdict, reason, metrics, pair data, and
debug-frame URLs.

Check the result in all three places:

```powershell
curl.exe http://127.0.0.1:8000/api/v1/analyses
curl.exe http://127.0.0.1:8000/api/v1/analyses/ANALYSIS_ID
curl.exe -X POST http://127.0.0.1:8000/api/v1/analyses/ANALYSIS_ID/rerun
```

Replace `ANALYSIS_ID` with the returned UUID. The debug URLs in the detail
response can be opened directly in a browser.

Expected behavior:

- A usable recording returns `3d`, `flat`, or `insufficient`.
- A second-or-later attempt that the checker finds insufficient is stored and
  returned as `review`.
- Rerun creates a new ID and preserves the original result.
- Public responses never expose laptop filesystem paths.

## Verify from a phone on Wi-Fi

1. Connect the phone and laptop to the same trusted Wi-Fi network.
2. Start Uvicorn with `--host 0.0.0.0` as shown above.
3. Find the laptop address:

   ```powershell
   ipconfig
   ```

4. Under `Wireless LAN adapter Wi-Fi`, find `IPv4 Address`.
5. On the phone, open `http://LAPTOP_IP:8000/`.
6. Upload a short recording and confirm the verdict and debug images load.

For example, if the laptop address is `192.168.1.6`, use:

```text
http://192.168.1.6:8000/
```

The address can change when the laptop reconnects, so check `ipconfig` again
if a previously working URL stops responding.

### Windows Firewall

Windows may ask whether Python or Uvicorn can accept incoming connections.
Allow access on **Private networks** only. Do not enable Public-network access
for this development server.

If the phone still cannot connect:

- Confirm the laptop page works at `http://127.0.0.1:8000/` first.
- Confirm Uvicorn says it is running on `0.0.0.0:8000`.
- Confirm both devices use the same Wi-Fi and neither is using a VPN.
- Check that the Wi-Fi does not use client/AP isolation, which prevents devices
  on the same network from communicating.
- Recheck the laptop IPv4 address with `ipconfig`.

## API summary

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/analyze` | Upload, analyze, and save a video |
| `GET` | `/api/v1/analyses` | List saved analyses, newest first |
| `GET` | `/api/v1/analyses/{id}` | Retrieve one complete result |
| `GET` | `/api/v1/analyses/{id}/debug/{n}.jpg` | Retrieve one annotated frame |
| `POST` | `/api/v1/analyses/{id}/rerun` | Reanalyze the saved source with current settings |
| `GET` | `/` | Open the laptop upload page |

The upload request remains open until analysis finishes. There is no job ID or
polling API.

## Response and error behavior

- `201 Created` — upload or rerun succeeded.
- `200 OK` — a read request succeeded.
- `400 Bad Request` — unsupported extension or unreadable video.
- `404 Not Found` — analysis or debug frame does not exist.
- `422 Unprocessable Entity` — invalid form, pagination, UUID, or frame index.
- `500 Internal Server Error` — unexpected checker, storage, or server failure.

Errors use this JSON shape:

```json
{
  "detail": "Human-readable error message."
}
```

## Stored data

The server creates:

```text
backend/data/
  uploads/             # Permanent copy of every uploaded source video
  results/             # Complete JSON result for every analysis
  debug/               # Annotated frames grouped by analysis ID
  analyses.sqlite3     # Searchable history index
```

Temporary HTTP uploads are deleted after success or failure. Reruns create new
permanent records and never overwrite the original.

## Troubleshooting

### `No module named app`

Start Uvicorn from the project root with `--app-dir backend`, exactly as shown
above. Run tests from inside `backend`.

### Upload returns `422`

Confirm the multipart field is named `video`, attempt is an integer of at
least `1`, and the optional label is one of the values shown in `/docs`.

### Upload returns `400`

Use an `.mp4` or `.mov` file that a normal video player can open. Renaming an
unrelated file to `.mp4` does not make it a valid video.

### Result is `insufficient`

Read the returned reason. Common fixes are better lighting, sharper focus,
showing the printed side and label, and moving around the box rather than only
rotating the phone in place.

### SQLite reports that the database is locked

Run one development server process against `backend/data/`. Stop duplicate
Uvicorn processes and retry.

## Security scope

This is a local development server. It has no authentication, authorization,
rate limiting, or HTTPS. Run it only on a trusted private network, do not expose
port `8000` through the router, and do not use it as an internet-facing service.

## Command-line checker

The standalone checker remains available without FastAPI:

```powershell
python -m boxcheck ".\sample.mp4" --phone-model "Pixel 7a" --debug-dir ".\debug-output"
```

It prints the raw checker result. The server-only `review` rule is not applied
by the command-line checker.

## Current structure

```text
backend/
  boxcheck/                # Part 1 standalone checker
  app/
    main.py                # FastAPI application factory and startup
    routes.py              # Versioned upload/read/rerun APIs
    schemas.py             # Public request/response contracts
    policies.py            # Server-only review rule
    store.py               # Part 2 files and SQLite persistence
    web.py                 # Laptop-page route
    web/index.html         # No-build laptop interface
  config/                  # Tunable checker and phone settings
  tests/                   # Unit, API, integration, and contract tests
  data/                    # Created automatically at server startup
  pyproject.toml
```

## Current status

Parts 1, 2, and 3 are implemented. The checker, persistence layer, APIs,
laptop page, rerun workflow, and integration contracts are covered by the
backend test suite. Part 4 is the evaluation scorecard.
