import cv2
import os
import sys
import json 
import numpy as np


inputdir = sys.argv[1]
outdir = sys.argv[2]
pose_file = sys.argv[3]

images = os.listdir(inputdir)

all_poses = []
f = open(pose_file)

poses = json.load(f)

# creation of data for the pixlenerf
f = open(f"{outdir}/objs.txt",'w')
f.write("data_1")
f.close()

os.mkdir(outdir+"/data_1")


for image in images:
    img_path = f"{inputdir}/{image}"
    img_name = img_path.split("/")[-1]
    img = cv2.imread(img_path)
    img = np.array(cv2.resize(img, (100, 100)))
    np.save(f"{outdir}/data_1/{img_name}.npy",img)

for frame in poses['frames']:
    all_poses.append(frame['transform_matrix'])

poses = np.array([all_poses])
os.chdir(outdir)

np.savez_compressed('./poses.npz',poses=poses,focal=138.88,camera_distance=1)

