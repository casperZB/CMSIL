import numpy as np
from pathlib import Path


# The official annotations link: https://www.crcv.ucf.edu/projects/real-world/
def get_annotations(file):
    gt = {}
    with open(file, "r") as f:
        contents = [x.strip().split() for x in f.readlines()]
        for content in contents:
            vid = Path(content[0]).stem
            annotations = content[2:]
            annotations = [annotations[i : i + 2] for i in range(0, len(annotations), 2) if int(annotations[i]) != -1]
            gt[vid] = []
            for annotation in annotations:
                gt[vid].append({"start": int(annotation[0]), "end": int(annotation[1])})
    return gt


TEXT_LIST = "./UCF_Test.list"
GT_TXT = "./Temporal_Anomaly_Annotation.txt"
ZIP_FEATS = "./ucfcrime_i3d_roc_ng_w16_s16.zip"
WINDOW = 16

zip_feats = np.load(ZIP_FEATS)
video_ids = dict.fromkeys([x.strip()[:-3] for x in open(TEXT_LIST)]).keys()
annotations = get_annotations(GT_TXT)
frameLabel, videoLabel = {}, {}
for vid in video_ids:
    # get video label
    vid_label = [0] if "Normal" in vid else [1]
    videoLabel[vid] = vid_label

    # get frame label
    num_frame = zip_feats[f"{vid}__0"].shape[0] * WINDOW
    vid_frame_label = np.zeros(num_frame, dtype=np.int8)
    vid_gt = annotations[vid]
    for item in vid_gt:
        startFrame = int(item["start"]) - 1
        endFrame = int(item["end"])
        if endFrame > num_frame:
            endFrame = num_frame
        vid_frame_label[startFrame:endFrame] = 1
    frameLabel[vid] = vid_frame_label
np.save("UCF_frame_label.npy", frameLabel)
np.save("UCF_video_label.npy", videoLabel)
