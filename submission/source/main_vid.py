from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


project_path = Path(__file__).parent.parent.parent
train_path = project_path / "evs_mot-train"
test_path = project_path / "evs_mot-test"
output_path = project_path / "submission" / "data"

CONF_THRESHOLD = 0.95   # high-confidence threshold (stage 1 + new track spawning)
# IOU_THRESHOLD = 0.17    # minimum IoU to match a detection to a track

IOU_THRESHOLD = 0.17        # stage 1 (high-conf dets)
IOU_THRESHOLD_LOW = 0.05    # stage 2 (low-conf dets)

MAX_AGE = 3            # frames a track survives without any match
MIN_HITS = 1           # matches needed before a track is reported


# Utilities

def parse_seqinfo(seq_path: Path) -> dict:
    seqinfo: dict = {}
    with open(seq_path / "seqinfo.ini") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                if key in ("frameRate", "seqLength", "imWidth", "imHeight"):
                    seqinfo[key] = int(value)
                else:
                    seqinfo[key] = value
    return seqinfo


def parse_detections(det_path: Path, conf_threshold: float) -> dict[int, list]:
    det: dict[int, list] = defaultdict(list)
    with open(det_path) as f:
        for line in f:
            parts = line.strip().split(",")
            frame = int(parts[0])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6])
            if conf >= conf_threshold and w > 0 and h > 0:
                det[frame].append([x, y, w, h, conf])
    return det


def iou_matrix(boxes_a: list, boxes_b: list) -> np.ndarray:
    a = np.array(boxes_a, dtype=np.float64)  # (n, 4)
    b = np.array(boxes_b, dtype=np.float64)  # (m, 4)

    ax1, ay1 = a[:, 0][:, None], a[:, 1][:, None]  # (n, 1)
    ax2, ay2 = ax1 + a[:, 2][:, None], ay1 + a[:, 3][:, None]
    bx1, by1 = b[:, 0][None, :], b[:, 1][None, :]  # (1, m)
    bx2, by2 = bx1 + b[:, 2][None, :], by1 + b[:, 3][None, :]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = a[:, 2] * a[:, 3]  # (n,)
    area_b = b[:, 2] * b[:, 3]  # (m,)
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.maximum(union, 1e-6)


def greedy_match(iou_mat: np.ndarray, threshold: float):
    """Greedy bipartite matching by descending IoU."""
    mat = iou_mat.copy()
    matches = []
    while mat.size > 0 and mat.max() >= threshold:
        r, c = np.unravel_index(mat.argmax(), mat.shape)
        matches.append((int(r), int(c)))
        mat[r, :] = -1.0
        mat[:, c] = -1.0

    matched_r = {r for r, _ in matches}
    matched_c = {c for _, c in matches}
    unmatched_r = [i for i in range(iou_mat.shape[0]) if i not in matched_r]
    unmatched_c = [j for j in range(iou_mat.shape[1]) if j not in matched_c]
    return matches, unmatched_r, unmatched_c


def hungarian_match(iou_mat: np.ndarray, threshold: float):
    if iou_mat.size == 0:
        return [], list(range(iou_mat.shape[0])), list(range(iou_mat.shape[1]))

    row_ind, col_ind = linear_sum_assignment(-iou_mat)

    matches = []
    matched_r, matched_c = set(), set()
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if iou_mat[r, c] >= threshold:
            matches.append((r, c))
            matched_r.add(r)
            matched_c.add(c)

    unmatched_r = [i for i in range(iou_mat.shape[0]) if i not in matched_r]
    unmatched_c = [j for j in range(iou_mat.shape[1]) if j not in matched_c]
    return matches, unmatched_r, unmatched_c


# Track

class Track:
    _counter = 0

    @classmethod
    def reset_counter(cls):
        cls._counter = 0

    def __init__(self, bbox: list):
        Track._counter += 1
        self.id = Track._counter
        self.bbox = np.asarray(bbox[:4], dtype=np.float64)  # [x, y, w, h]
        cx, cy = self.bbox[0] + self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2
        self._last_det_center = np.array([cx, cy])
        self.vel = np.zeros(2)
        self.hits = 1
        self.time_since_update = 0

    def predict(self):
        """Move bbox by current velocity and age the track."""
        self.bbox[:2] += self.vel
        self.time_since_update += 1

    def update(self, det_bbox: list):
        """Update track state from a matched detection."""
        new_center = np.array([det_bbox[0] + det_bbox[2] / 2,
                               det_bbox[1] + det_bbox[3] / 2])

        self.vel = (new_center - self._last_det_center) / self.time_since_update
        self._last_det_center = new_center.copy()
        # self.bbox = np.asarray(det_bbox[:4], dtype=np.float64)
        alpha = 0.7
        self.bbox = alpha * np.asarray(det_bbox[:4], dtype=np.float64) + (1 - alpha) * self.bbox
        self.hits += 1
        self.time_since_update = 0


# Tracker

class Tracker:
    # def __init__(self, iou_threshold: float, max_age: int, min_hits: int):
    #     self.tracks: list[Track] = []
    #     self.iou_threshold = iou_threshold
    #     self.max_age = max_age
    #     self.min_hits = min_hits

    def __init__(self, iou_threshold: float, iou_threshold_low: float, max_age: int, min_hits: int):
        self.tracks: list[Track] = []
        self.iou_threshold = iou_threshold
        self.iou_threshold_low = iou_threshold_low
        self.max_age = max_age
        self.min_hits = min_hits

    def step(self, high_dets: list, low_dets: list = ()) -> list[tuple[int, np.ndarray]]:
        """Process one frame using ByteTrack two-stage matching.

        Stage 1: match active tracks vs high-confidence detections.
        Stage 2: match remaining unmatched tracks vs low-confidence detections.
        New tracks are spawned only from unmatched high-confidence detections.
        """
        for t in self.tracks:
            t.predict()

        # Stage 1 — high-confidence detections
        if self.tracks and high_dets:
            pred_boxes = [t.bbox.tolist() for t in self.tracks]
            iou_mat = iou_matrix(pred_boxes, [d[:4] for d in high_dets])
            matches1, unmatched_tracks, unmatched_high = hungarian_match(iou_mat, self.iou_threshold)
        else:
            matches1 = []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_high = list(range(len(high_dets)))

        for ti, di in matches1:
            self.tracks[ti].update(high_dets[di])

        # Stage 2 — low-confidence detections for still-unmatched tracks
        if unmatched_tracks and low_dets:
            remaining = [self.tracks[i] for i in unmatched_tracks]
            iou_mat2 = iou_matrix([t.bbox.tolist() for t in remaining],
                                   [d[:4] for d in low_dets])
            # matches2, _, _ = hungarian_match(iou_mat2, self.iou_threshold)
            matches2, _, _ = hungarian_match(iou_mat2, self.iou_threshold_low)
            for local_ti, di in matches2:
                self.tracks[unmatched_tracks[local_ti]].update(low_dets[di])

        # Spawn new tracks only from unmatched HIGH-confidence detections
        for di in unmatched_high:
            self.tracks.append(Track(high_dets[di]))

        alive = []
        results = []
        for t in self.tracks:
            if t.time_since_update <= self.max_age:
                alive.append(t)
                if t.hits >= self.min_hits:
                    results.append((t.id, t.bbox.copy()))
        self.tracks = alive

        return results


# Visualization

_PALETTE = [
    (230,  25,  75), ( 60, 180,  75), (255, 225,  25), (  0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
]


def _color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def visualize_sequence(
    seq_path: Path,
    conf_threshold: float,
    iou_threshold: float,
    iou_threshold_low: float,
    max_age: int,
    min_hits: int,
    out_video: Path,
) -> None:
    seqinfo = parse_seqinfo(seq_path)
    det = parse_detections(seq_path / "det" / "det.txt", 0.0)
    img_dir = seq_path / seqinfo["imDir"]
    img_ext = seqinfo["imExt"]
    fps = seqinfo["frameRate"]
    w_frame, h_frame = seqinfo["imWidth"], seqinfo["imHeight"]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w_frame, h_frame))

    Track.reset_counter()
    tracker = Tracker(iou_threshold, iou_threshold_low, max_age, min_hits)

    n_frames = seqinfo["seqLength"]
    for frame in range(1, n_frames + 1):
        if frame % 100 == 0:
            print(f"  frame {frame}/{n_frames}")

        img_path = img_dir / f"{frame:06d}{img_ext}"
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((h_frame, w_frame, 3), dtype=np.uint8)

        all_dets = det.get(frame, [])
        high_dets = [d for d in all_dets if d[4] >= conf_threshold]
        low_dets = [d for d in all_dets if d[4] < conf_threshold]
        results = tracker.step(high_dets, low_dets)

        for tid, bbox in results:
            x, y, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            color = _color(tid)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)

            label = str(tid)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(y - 4, th + baseline)
            cv2.rectangle(img, (x, ty - th - baseline), (x + tw, ty), color, -1)
            cv2.putText(img, label, (x, ty - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(img)

    writer.release()
    print(f"  -> video saved to {out_video}")


def run_sequence(
    seq_path: Path,
    conf_threshold: float,
    iou_threshold: float,
    iou_threshold_low: float,
    max_age: int,
    min_hits: int,
) -> list[str]:
    seqinfo = parse_seqinfo(seq_path)
    det = parse_detections(seq_path / "det" / "det.txt", 0.0)

    Track.reset_counter()
    tracker = Tracker(iou_threshold, iou_threshold_low, max_age, min_hits)
    

    lines = []
    for frame in range(1, seqinfo["seqLength"] + 1):
        all_dets = det.get(frame, [])
        high_dets = [d for d in all_dets if d[4] >= conf_threshold]
        low_dets = [d for d in all_dets if d[4] < conf_threshold]
        results = tracker.step(high_dets, low_dets)
        for tid, bbox in results:
            x, y, w, h = bbox
            lines.append(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")

    return lines


# ---------------------------------------------------------------------------


def main():
    output_path.mkdir(parents=True, exist_ok=True)
    viz_path = project_path / "viz"
    viz_path.mkdir(parents=True, exist_ok=True)

    for seq_path in sorted(test_path.iterdir()):
        if not seq_path.is_dir():
            continue
        print(f"Processing {seq_path.name}...")
        lines = run_sequence(
            seq_path,
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            iou_threshold_low=IOU_THRESHOLD_LOW,
            max_age=MAX_AGE,
            min_hits=MIN_HITS,
        )
        out_file = output_path / f"{seq_path.name}.txt"
        with open(out_file, "w") as f:
            f.write("\n".join(lines))
        print(f"  -> {len(lines)} track-frames -> {out_file.name}")

    print("\nGenerating visualization videos...")
    for seq_path in sorted(train_path.iterdir()) + sorted(test_path.iterdir()):
        if not seq_path.is_dir():
            continue
        print(f"Visualizing {seq_path.name}...")
        out_video = viz_path / f"{seq_path.name}.mp4"
        visualize_sequence(
            seq_path,
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            iou_threshold_low=IOU_THRESHOLD_LOW,
            max_age=MAX_AGE,
            min_hits=MIN_HITS,
            out_video=out_video,
        )


if __name__ == "__main__":
    main()