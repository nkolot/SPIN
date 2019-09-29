import os
import glob
import numpy as np
import scipy.io as sio
from .read_openpose import read_openpose

def hr_lspet_extract(dataset_path, openpose_path, out_path):

    # training mode
    png_path = os.path.join(dataset_path, '*.png')
    imgs = glob.glob(png_path)
    imgs.sort()

    # structs we use
    imgnames_, scales_, centers_, parts_, openposes_= [], [], [], [], []

    # scale factor
    scaleFactor = 1.2

    # annotation files
    annot_file = os.path.join(dataset_path, 'joints.mat')
    joints = sio.loadmat(annot_file)['joints']

    # main loop
    for i, imgname in enumerate(imgs):
        # image name
        imgname = imgname.split('/')[-1]
        # read keypoints
        part14 = joints[:,:2,i]
        # scale and center
        bbox = [min(part14[:,0]), min(part14[:,1]),
                max(part14[:,0]), max(part14[:,1])]
        center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
        scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
        # update keypoints
        part = np.zeros([24,3])
        part[:14] = np.hstack([part14, np.ones([14,1])])
        # read openpose detections
        json_file = os.path.join(openpose_path, 'hrlspet',
            imgname.replace('.png', '_keypoints.json'))
        openpose = read_openpose(json_file, part, 'hrlspet') 

        # store the data
        imgnames_.append(imgname)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        openposes_.append(openpose)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'hr-lspet_train.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       openpose=openposes_)
