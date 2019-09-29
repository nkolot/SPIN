import os
import argparse
import numpy as np
import scipy.io as sio
from .read_openpose import read_openpose

def lsp_dataset_extract(dataset_path, out_path):

    # bbox expansion factor
    scaleFactor = 1.2

    # We use LSP dataset only in testing
    imgs = range(1000,2000)

    # structs we use
    imgnames_, scales_, centers_, parts_  = [], [], [], []
    masknames_, partnames_ = [], []

    # annotation files
    annot_file = os.path.join(dataset_path, 'joints.mat')
    joints = sio.loadmat(annot_file)['joints']

    # go over all images
    for img_i in imgs:
        # image names
        img_base = 'im%04d' % (img_i+1)
        imgname = 'images/%s.jpg' % img_base
        maskname = 'data/lsp/%s_segmentation.png' % img_base
        partname = 'data/lsp/%s_part_segmentation.png' % img_base
        # keypoints
        part14 = joints[:2,:,img_i].T
        # scale and center
        bbox = [min(part14[:,0]), min(part14[:,1]),
                max(part14[:,0]), max(part14[:,1])]
        center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
        scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
        # update keypoints
        part = np.zeros([24,3])
        part[:14] = np.hstack([part14, np.ones([14,1])])

        # store data
        imgnames_.append(imgname)
        masknames_.append(maskname)
        partnames_.append(partname)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'lsp_dataset_test.npz')
    np.savez(out_file, imgname=imgnames_,
                       maskname=masknames_,
                       partname=partnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_)
