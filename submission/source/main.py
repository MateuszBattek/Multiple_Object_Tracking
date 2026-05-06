from pathlib import Path

project_path = Path(__file__).parent.parent.parent
MOT_folder = project_path / "evs_mot-train" / "MOT_05"

seqinfo = {}

with open(MOT_folder / "seqinfo.ini") as f:
    for line in f:
        if "=" in line:
            key, value = line.strip().split("=")
            seqinfo[key] = value

det_path = MOT_folder / "det" / "det.txt"

det = {}

with open(det_path) as f:
    for line in f:
        frame, *info = line.strip().split(",")
        info = [float(i) for i in info]
        det[int(frame)] = info

images = []

for frame in range(1, int(seqinfo["frames"] + 1)):
    image_path = MOT_folder / seqinfo["imDir"] / f"{frame:06d}{seqinfo['imExt']}"
    images.append(image_path)
