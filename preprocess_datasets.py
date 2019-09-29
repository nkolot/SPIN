#!/usr/bin/python
"""
Preprocess datasets and generate npz files to be used for training testing.
It is recommended to first read datasets/preprocess/README.md
"""
import argparse
import config as cfg
from datasets.preprocess import h36m_extract,\
                                pw3d_extract, \
                                mpi_inf_3dhp_extract, \
                                lsp_dataset_extract,\
                                lsp_dataset_original_extract, \
                                hr_lspet_extract, \
                                mpii_extract, \
                                coco_extract

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', default=False, action='store_true', help='Extract files needed for training')
parser.add_argument('--eval_files', default=False, action='store_true', help='Extract files needed for evaluation')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # define path to store extra files
    out_path = cfg.DATASET_NPZ_PATH
    openpose_path = cfg.OPENPOSE_PATH

    if args.train_files:
        # MPI-INF-3DHP dataset preprocessing (training set)
        mpi_inf_3dhp_extract(cfg.MPI_INF_3DHP_ROOT, openpose_path, out_path, 'train', extract_img=True, static_fits=cfg.STATIC_FITS_DIR)

        # LSP dataset original preprocessing (training set)
        lsp_dataset_original_extract(cfg.LSP_ORIGINAL_ROOT, openpose_path, out_path)

        # LSP Extended training set preprocessing - HR version
        hr_lspet_extract(cfg.LSPET_ROOT, openpose_path, out_path)

        # MPII dataset preprocessing
        mpii_extract(cfg.MPII_ROOT, openpose_path, out_path)

        # COCO dataset prepreocessing
        coco_extract(cfg.COCO_ROOT, openpose_path, out_path)

    if args.eval_files:
        # Human3.6M preprocessing (two protocols)
        h36m_extract(cfg.H36M_ROOT, out_path, protocol=1, extract_img=True)
        h36m_extract(cfg.H36M_ROOT, out_path, protocol=2, extract_img=False)
        
        # MPI-INF-3DHP dataset preprocessing (test set)
        mpi_inf_3dhp_extract(cfg.MPI_INF_3DHP_ROOT, openpose_path, out_path, 'test')
        
        # 3DPW dataset preprocessing (test set)
        pw3d_extract(cfg.PW3D_ROOT, out_path)

        # LSP dataset preprocessing (test set)
        lsp_dataset_extract(cfg.LSP_ROOT, out_path)
