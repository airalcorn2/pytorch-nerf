import cv2
import os
import sys
import json 
import numpy as np


inputdir = sys.argv[1]
outdir = sys.argv[2]
pose_file = sys.argv[3]

images = os.listdir(inputdir)

all_images = []
all_poses = []
f = open(pose_file)

poses = json.load(f)


for image in images:
    img_path = f"{inputdir}/{image}"
    img = cv2.imread(img_path)
    img = np.array(cv2.resize(img, (100, 100)))
    all_images.append(img)

for frame in poses['frames']:
    all_poses.append(frame['transform_matrix'])

os.chdir(outdir)

np.savez_compressed('./data.npz',images=all_images,poses=all_poses,focal=138.88,camera_distance=1)

