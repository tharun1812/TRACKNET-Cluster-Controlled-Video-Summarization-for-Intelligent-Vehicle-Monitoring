import numpy as np

from tracknet.tracker import DeepSortTracker, Detection, iou_matrix


def det(x, y, w=60, h=40, emb=None):
    return Detection(bbox=[x, y, x + w, y + h], embedding=emb)


def run(tracker, frames):
    ids_per_frame = []
    for i, dets in enumerate(frames):
        active = tracker.update(i, dets)
        ids_per_frame.append(sorted(t.track_id for t in active))
    return ids_per_frame


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    iou = iou_matrix(a, b)[0]
    assert iou[0] == 1.0
    assert abs(iou[1] - 1 / 3) < 1e-6
    assert iou[2] == 0.0


def test_single_object_keeps_one_id():
    tr = DeepSortTracker()
    ids = run(tr, [[det(10 + 8 * i, 100)] for i in range(30)])
    assert {i for frame in ids[2:] for i in frame} == {1}
    assert len(tr.flush()) == 1


def test_two_objects_in_separate_lanes_keep_distinct_ids():
    tr = DeepSortTracker()
    frames = [[det(10 + 8 * i, 50), det(500 - 8 * i, 200)] for i in range(40)]
    ids = run(tr, frames)
    assert set(ids[-1]) == {1, 2}
    assert len(tr.flush()) == 2


def test_track_survives_short_occlusion():
    tr = DeepSortTracker(max_age=10)
    frames = []
    for i in range(30):
        frames.append([] if 12 <= i < 15 else [det(10 + 8 * i, 100)])
    ids = run(tr, frames)
    assert {i for frame in ids for i in frame} == {1}


def test_track_is_dropped_after_max_age_and_new_object_gets_new_id():
    tr = DeepSortTracker(max_age=5)
    frames = [[det(10 + 5 * i, 100)] for i in range(10)] + [[] for _ in range(10)] + \
             [[det(300 + 5 * i, 200)] for i in range(10)]
    ids = run(tr, frames)
    assert ids[9] == [1]
    assert ids[-1] == [2]
    assert len(tr.flush()) == 2


def test_short_flicker_is_never_confirmed():
    tr = DeepSortTracker(n_init=3)
    ids = run(tr, [[det(50, 50)], [], [], [], [], []])
    assert all(frame == [] for frame in ids)
    assert tr.flush() == []


def test_appearance_helps_choose_between_overlapping_candidates():
    red = np.array([1.0, 0.0, 0.0])
    blue = np.array([0.0, 1.0, 0.0])
    tr = DeepSortTracker()
    for i in range(5):
        tr.update(i, [det(100 + 4 * i, 100, emb=red)])
    # two overlapping boxes appear; the one that looks like the track must keep its id
    tracks = tr.update(5, [det(118, 100, emb=blue), det(121, 100, emb=red)])
    assert len(tracks) >= 1
    owner = [t for t in tracks if t.track_id == 1][0]
    assert np.allclose(owner.last_detection.embedding, red)
