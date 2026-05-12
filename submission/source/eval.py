"""
Evaluate tracker (from main.py) on all training sequences using MOTA / MOTP / IDSW.

Run:
    uv run submission/source/eval.py
    uv run submission/source/eval.py --grid
"""
import sys
from itertools import product
from pathlib import Path

import motmetrics as mm
import numpy as np
from scipy.optimize import linear_sum_assignment

# Allow importing from this same directory
sys.path.insert(0, str(Path(__file__).parent))
from main import (  # noqa: E402
    CONF_THRESHOLD, IOU_THRESHOLD, MAX_AGE, MIN_HITS,
    Track, Tracker, parse_seqinfo, parse_detections,
    iou_matrix as _iou_matrix,
)

project_path = Path(__file__).parent.parent.parent
train_path = project_path / "evs_mot-train"

# IoU threshold used by the MOT benchmark to decide TP vs FP
EVAL_IOU = 0.5


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_gt(gt_file: Path) -> dict[int, list]:
    """Returns {frame: [(id, x, y, w, h), ...]} filtered to eval_flag=1, class=1."""
    gt: dict[int, list] = {}
    with open(gt_file) as f:
        for line in f:
            p = line.strip().split(",")
            frame, obj_id = int(p[0]), int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            eval_flag, cls = int(p[6]), int(p[7])
            if eval_flag == 1 and cls == 1:
                gt.setdefault(frame, []).append((obj_id, x, y, w, h))
    return gt


def run_tracker(seq_path: Path) -> dict[int, list]:
    """Run tracker and return {frame: [(id, x, y, w, h), ...]}."""
    seqinfo = parse_seqinfo(seq_path)
    det = parse_detections(seq_path / "det" / "det.txt", 0.0)

    Track.reset_counter()
    tracker = Tracker(IOU_THRESHOLD, MAX_AGE, MIN_HITS)

    hyp: dict[int, list] = {}
    for frame in range(1, seqinfo["seqLength"] + 1):
        all_dets = det.get(frame, [])
        high_dets = [d for d in all_dets if d[4] >= CONF_THRESHOLD]
        low_dets  = [d for d in all_dets if d[4] < CONF_THRESHOLD]
        results = tracker.step(high_dets, low_dets)
        if results:
            hyp[frame] = [(tid, *bbox.tolist()) for tid, bbox in results]
    return hyp


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence(seq_name: str) -> mm.MOTAccumulator:
    seq_path = train_path / seq_name
    gt = load_gt(seq_path / "gt" / "gt.txt")
    hyp = run_tracker(seq_path)

    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(set(gt) | set(hyp)):
        gt_entries = gt.get(frame, [])   # [(id, x, y, w, h), ...]
        hyp_entries = hyp.get(frame, [])

        gt_ids = [e[0] for e in gt_entries]
        hyp_ids = [e[0] for e in hyp_entries]

        gt_boxes = [list(e[1:]) for e in gt_entries]    # [[x, y, w, h], ...]
        hyp_boxes = [list(e[1:]) for e in hyp_entries]

        if gt_boxes and hyp_boxes:
            # _iou_matrix expects [x, y, w, h] — numpy 2.0 compatible.
            # motmetrics expects distance = 1-IoU, NaN where IoU < threshold.
            iou = _iou_matrix(gt_boxes, hyp_boxes)
            dist = 1.0 - iou
            dist[iou < EVAL_IOU] = np.nan
        else:
            dist = np.full((len(gt_ids), len(hyp_ids)), np.nan)

        acc.update(gt_ids, hyp_ids, dist)
    return acc


# ---------------------------------------------------------------------------
# Fast MOTA (no motmetrics — used by grid search)
# ---------------------------------------------------------------------------

def _fast_mota(gt: dict, hyp: dict, eval_iou: float = 0.5) -> tuple[int, int, int, int]:
    """Returns (fn, fp, idsw, total_gt) — no motmetrics dependency."""
    total_gt = total_fn = total_fp = total_idsw = 0
    gt_to_hyp: dict[int, int] = {}

    for frame in sorted(set(gt) | set(hyp)):
        gt_entries = gt.get(frame, [])
        hyp_entries = hyp.get(frame, [])
        total_gt += len(gt_entries)

        if not gt_entries:
            total_fp += len(hyp_entries)
            continue
        if not hyp_entries:
            total_fn += len(gt_entries)
            continue

        iou = _iou_matrix([list(e[1:]) for e in gt_entries],
                           [list(e[1:]) for e in hyp_entries])
        cost = np.where(iou >= eval_iou, 1.0 - iou, 1.0)
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_gt, matched_hyp = set(), set()
        for r, c in zip(row_ind, col_ind):
            if iou[r, c] >= eval_iou:
                matched_gt.add(r)
                matched_hyp.add(c)
                gt_id, hyp_id = gt_entries[r][0], hyp_entries[c][0]
                if gt_id in gt_to_hyp and gt_to_hyp[gt_id] != hyp_id:
                    total_idsw += 1
                gt_to_hyp[gt_id] = hyp_id

        total_fn += len(gt_entries) - len(matched_gt)
        total_fp += len(hyp_entries) - len(matched_hyp)

    return total_fn, total_fp, total_idsw, total_gt


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

GRID = {
    "conf_threshold": [0.5, 0.7, 0.9, 0.95],
    "iou_threshold":  [0.1, 0.2, 0.3, 0.4],
    "max_age":        [1, 2, 3],
    "min_hits":       [1, 2, 3],
}


def grid_search() -> None:
    sequences = sorted(p.name for p in train_path.iterdir() if p.is_dir())

    print("Pre-loading GT and detections...")
    gt_data  = {s: load_gt(train_path / s / "gt" / "gt.txt") for s in sequences}
    raw_dets = {s: parse_detections(train_path / s / "det" / "det.txt", 0.0) for s in sequences}
    seqinfos = {s: parse_seqinfo(train_path / s) for s in sequences}

    keys   = list(GRID.keys())
    combos = list(product(*GRID.values()))
    print(f"Testing {len(combos)} combinations on {len(sequences)} sequences...\n")

    best_mota = -np.inf
    best_params: dict = {}
    all_results: list[tuple[float, dict]] = []

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        ct, iou_t, age, mh = (params["conf_threshold"], params["iou_threshold"],
                               params["max_age"], params["min_hits"])

        agg_fn = agg_fp = agg_idsw = agg_gt = 0

        for seq in sequences:
            Track.reset_counter()
            tracker = Tracker(iou_t, age, mh)
            hyp: dict[int, list] = {}

            for frame in range(1, seqinfos[seq]["seqLength"] + 1):
                all_d = raw_dets[seq].get(frame, [])
                high  = [d for d in all_d if d[4] >= ct]
                low   = [d for d in all_d if d[4] < ct]
                res   = tracker.step(high, low)
                if res:
                    hyp[frame] = [(tid, *bbox.tolist()) for tid, bbox in res]

            fn, fp, idsw, gt_n = _fast_mota(gt_data[seq], hyp)
            agg_fn += fn; agg_fp += fp; agg_idsw += idsw; agg_gt += gt_n

        mota = 1.0 - (agg_fn + agg_fp + agg_idsw) / max(agg_gt, 1)
        all_results.append((mota, params.copy()))

        if mota > best_mota:
            best_mota = mota
            best_params = params.copy()
            print(f"  [{i+1:>4}/{len(combos)}] NEW BEST  MOTA={mota*100:.2f}%  {params}")

    all_results.sort(key=lambda x: -x[0])
    print(f"\n{'='*65}")
    print(f"  {'MOTA%':>6}  {'CONF':>6}  {'IOU':>5}  {'AGE':>3}  {'HITS':>4}")
    print(f"  {'-'*50}")
    for mota, p in all_results[:10]:
        print(f"  {mota*100:>6.2f}%  {p['conf_threshold']:>6.2f}  "
              f"{p['iou_threshold']:>5.2f}  {p['max_age']:>3}  {p['min_hits']:>4}")
    print(f"\nBest MOTA: {best_mota*100:.2f}%")
    print(f"Best params: {best_params}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Parameters: CONF={CONF_THRESHOLD}  IOU={IOU_THRESHOLD}  "
          f"MAX_AGE={MAX_AGE}  MIN_HITS={MIN_HITS}\n")

    sequences = sorted(p.name for p in train_path.iterdir() if p.is_dir())
    accs, names = [], []

    for seq in sequences:
        print(f"  Evaluating {seq}...", end=" ", flush=True)
        acc = evaluate_sequence(seq)
        accs.append(acc)
        names.append(seq)
        print("done")

    mh = mm.metrics.create()
    metrics = ["mota", "motp", "num_switches", "num_false_positives", "num_misses", "num_detections"]
    summary = mh.compute_many(accs, metrics=metrics, names=names, generate_overall=True)

    # motmetrics stores MOTP as mean distance (1-IoU); convert back to IoU %
    summary["motp"] = (1 - summary["motp"]) * 100
    summary["mota"] *= 100

    print("\n" + summary.rename(columns={
        "mota": "MOTA(%)",
        "motp": "MOTP(IoU%)",
        "num_switches": "IDSW",
        "num_false_positives": "FP",
        "num_misses": "FN",
        "num_detections": "TP",
    }).to_string(float_format="%.2f"))


if __name__ == "__main__":
    if "--grid" in sys.argv:
        grid_search()
    else:
        main()