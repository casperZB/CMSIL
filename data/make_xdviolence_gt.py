import numpy as np
import os
import cv2
import io
import subprocess


def decode_url(url):
    curl_cmd = f"curl -s {url}"
    response = subprocess.check_output(curl_cmd, shell=True)
    file_object = io.StringIO(response.decode("utf-8"))
    return file_object


def get_annotations(url):
    gt = {}
    contents = [x.strip() for x in decode_url(url)]
    for content in contents:
        content = content.strip("\n").split()
        vid = content[0]
        if vid.endswith('.mp4'):  # sone lines in the annotations.txt have .mp4 suffix
            vid = vid[:-4]
        gt[vid] = []
        annotations = content[1:]
        annotations = [annotations[i : i + 2] for i in range(0, len(annotations), 2)]
        for annotation in annotations:
            gt[vid].append({"start": int(annotation[0]), "end": int(annotation[1])})
    return gt


TEXT_LIST = "./XDViolence_Test.list"
GT_TXT_URL = "https://roc-ng.github.io/XD-Violence/images/annotations.txt"  ## the url of test annotations
ZIP_FEATS = "./xdviolence_i3d_w16_s16.zip"
WINDOW = 16


zip_feats = np.load(ZIP_FEATS)
video_ids = dict.fromkeys([x.strip()[:-3] for x in open(TEXT_LIST)]).keys()
annotations = get_annotations(GT_TXT_URL)
frameLabel, videoLabel = {}, {}
for vid in video_ids:
    # get video label
    vid_label = [0] if "_label_A" in vid else [1]
    videoLabel[vid] = vid_label

    # get frame label
    num_frame = zip_feats[f"{vid}__0"].shape[0] * WINDOW
    vid_frame_label = np.zeros(num_frame, dtype=np.int8)
    vid_gt = annotations.get(vid)
    if vid_gt is not None:
        for item in vid_gt:
            startFrame = int(item["start"])
            endFrame = int(item["end"])
            if endFrame > num_frame:
                endFrame = num_frame
            vid_frame_label[startFrame:endFrame] = 1
    frameLabel[vid] = vid_frame_label
np.save("XD_frame_label.npy", frameLabel)
np.save("XD_video_label.npy", videoLabel)
