# TrackNet: Cluster-Controlled Video Summarization for Intelligent Vehicle Monitoring

A multi-camera traffic monitoring system. Nearby cameras are grouped into **clusters**; vehicles are detected and
tracked in every camera, stored as searchable metadata, **traced across cameras** with a graph-aware matcher, and
condensed into short **summary clips**.

> Based on my B.E. final-year project at Sathyabama Institute of Science and Technology (April 2026), developed with
> Madhaeswaran M under the guidance of Ms. E. Vinothini. This repository is a clean, tested rebuild of that system.

## What it does

| Stage | What happens | Code |
| --- | --- | --- |
| Detection | YOLOv8n finds cars, buses, trucks and motorcycles in each frame | `tracknet/detector.py` |
| Tracking | DeepSORT-style tracker (Kalman filter + IoU + appearance, Hungarian assignment) keeps one ID per vehicle | `tracknet/tracker.py` |
| Metadata | Per-track class, colour, direction, appearance embedding, trajectory, timestamps and best crop are stored in SQLite | `tracknet/pipeline.py`, `tracknet/db.py` |
| Search | Text queries such as "white truck" or "red car on c021" (rule-based parser over the metadata) | `tracknet/query.py` |
| Cross-camera tracing | Bidirectional BFS over the camera graph, candidates scored on appearance, colour, travel time, direction and graph distance; the operator confirms or skips each match | `tracknet/graph.py`, `matcher.py`, `session.py` |
| Events | Congestion periods and stationary vehicles per camera | `tracknet/events.py` |
| Summarization | Cuts only the relevant segments (one vehicle's path, or a cluster's events) into one H.264 clip | `tracknet/summarizer.py` |
| Dashboard | React app: overview, vehicle search, path tracing, cluster summaries, manual labelling | `frontend/` |

```
videos --> detector --> DeepSORT-style tracker --> metadata (SQLite)
                                                        |
        React dashboard <---- Flask REST API <----------+---- query parser
                                   |                    +---- camera graph + BFS matcher
                                   +---- summarizer (OpenCV + ffmpeg)
```

## Quick start (no dataset needed)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m tracknet demo        # renders synthetic multi-camera videos, runs the real pipeline, prints an evaluation
python -m tracknet serve       # API on http://127.0.0.1:5000
```

For the dashboard:

```bash
cd frontend && npm install
npm run dev                    # http://localhost:5173, proxies /api to the Flask server
# or: npm run build, and Flask will serve frontend/dist at http://127.0.0.1:5000
```

Try it: **Find a vehicle** -> search "red car" -> **Trace across cameras** -> confirm matches (or Auto-trace) ->
**Generate summary clip**.

## Using your own traffic videos

```bash
pip install -r requirements-yolo.txt                      # adds ultralytics (PyTorch)
python -m tracknet process --video cam17.mp4 --camera c017
python -m tracknet process --video cam18.mp4 --camera c018
python -m tracknet serve
```

Edit `tracknet/config.py` to describe your cameras, clusters and the road connections between them (with the
min/max travel time in seconds). `--stride 2` processes every second frame for speed. Videos from different cameras
should share a time reference; use `time_offset` in the config when they do not start together.
`process_video(..., embedder=...)` accepts any function that maps a BGR crop to a feature vector, so a ReID network
can replace the colour-histogram embedding.

## How matching works

For a selected track, BFS expands from its camera **forward** (where it may go) and **backward** (where it may have
come from) over the camera graph. Each track seen there is scored 0-1:

`0.40 appearance + 0.20 colour + 0.20 travel-time fit + 0.10 direction + 0.10 graph proximity`

and multiplied by 0.3 when the vehicle class differs. Candidates whose arrival time is physically implausible for
the route are dropped. `shortest_path` uses true bidirectional BFS, and the sum of edge travel windows along the path
defines the expected travel time. Weights live in `tracknet/config.py`.

## Measured results (synthetic demo, seed 7)

`python -m tracknet demo` renders 30 vehicles across 5 cameras (103 camera appearances, known ground truth), runs
background-subtraction detection and the tracker on the videos, then traces every vehicle from its first sighting:

| Metric | Result |
| --- | --- |
| Appearances detected and tracked | 103 / 103 (no fragmented tracks) |
| Cross-camera link precision / recall / F1 (auto-trace, min score 0.55) | 0.77 / 0.67 / 0.72 |
| Pipeline speed (CPU, 640x360) | about 200 frames/s |
| Candidate search latency | 4.5 ms at 100 tracks, 27 ms at 1,000, 71 ms at 2,000 |

These numbers come from simple rendered rectangles, not real CCTV. They show the pipeline works end to end and give
a repeatable test bench, but they are **not** a claim about accuracy on real traffic footage. Latency figures depend
on your machine (`python -m tracknet evaluate` reproduces them).

## REST API

| Method and path | Purpose |
| --- | --- |
| `GET /api/cameras`, `/api/clusters`, `/api/analytics`, `/api/events` | overview data |
| `GET /api/tracks`, `/api/tracks/<id>`, `/api/tracks/<id>/image` | browse stored tracks |
| `POST /api/search` `{query, camera?}` | descriptive vehicle search |
| `POST /api/sessions` `{track_id}` | start a tracing session |
| `GET /api/sessions/<id>/candidates?direction=forward\|backward` | ranked candidates with score breakdown |
| `POST /api/sessions/<id>/confirm`, `/skip`, `/auto` | grow the vehicle chain |
| `POST /api/sessions/<id>/summary`, `/api/clusters/<c>/summary` | build summary clips |
| `POST /api/labels`, `GET /api/labels` | manual labelling of confirmed chains |

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q            # 64 tests: tracker, graph, matcher, sessions, search, events, pipeline on real video, summarizer, API
```

## Project layout

```
tracknet/        detector, tracker, pipeline, graph, matcher, session, query, events, summarizer, api, demo, cli
tests/           pytest suite
frontend/        React + TypeScript + Tailwind dashboard
data/            generated at runtime (database, crops, demo videos, summaries), git-ignored
```

## Limitations

- The text search is a rule-based parser (colour, type, direction, camera), not a trained language-vision model.
- The default appearance embedding is an HSV colour histogram. It works for colour and rough look, but a trained
  ReID network will separate similar-looking vehicles much better (see the `embedder` hook).
- The YOLOv8n detector path needs `requirements-yolo.txt`. The demo uses a background-subtraction detector so that it
  runs anywhere; its class labels are a size heuristic and only meant for the demo.
- The camera graph and travel windows are configured by hand rather than learned from data.
- Sessions are kept in memory, and the system runs on prerecorded footage (simulation mode), not live streams.
- No authentication: do not expose the API publicly. Surveillance data needs access control and privacy safeguards
  before any real deployment.
