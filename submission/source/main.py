from collections import defaultdict
from pathlib import Path

import cv2


def draw_detections(image, detection_list):
    for box in detection_list:
        _, x, y, w, h, _ = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)


project_path = Path(__file__).parent.parent.parent
MOT_folder = project_path / "evs_mot-train" / "MOT_05"

seqinfo = {}

with open(MOT_folder / "seqinfo.ini") as f:
    for line in f:
        if "=" in line:
            key, value = line.strip().split("=")
            if key in ["frameRate", "seqLength", "imWidth", "imHeight"]:
                value = int(value)
            seqinfo[key] = value

det_path = MOT_folder / "det" / "det.txt"

det = defaultdict(list)

with open(det_path) as f:
    for line in f:
        frame, *info = line.strip().split(",")
        info = [round(float(i)) for i in info]
        det[int(frame) - 1].append(info)

images = []

print("Loading images...")
for frame in range(1, seqinfo["seqLength"] + 1):
    image_path = MOT_folder / seqinfo["imDir"] / f"{frame:06d}{seqinfo['imExt']}"
    img = cv2.imread(image_path)
    images.append(img)

print("Drawing boxes...")

draw_detections(images[0], det[0])

cv2.imshow("Obraz", images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
