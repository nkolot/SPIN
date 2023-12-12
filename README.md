# SPIN - SMPL oPtimization IN the loop
Code repository for the paper:  
**Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop**  
[Nikos Kolotouros](https://www.nikoskolot.com/)\*, [Georgios Pavlakos](https://geopavlakos.github.io/)\*, [Michael J. Black](https://ps.is.mpg.de/~black), [Kostas Daniilidis](http://www.cis.upenn.edu/~kostas/)  
ICCV 2019  
[[paper](https://arxiv.org/pdf/1909.12828.pdf)] [[project page](https://www.nikoskolot.com/projects/spin/)]

![teaser](teaser.png)

##  [<img src="https://i.imgur.com/QCojoJk.png" width="40"> You can run the SPIN demo in Google Colab](https://colab.research.google.com/drive/1uH2JtavOtDrFl6RsipyIncCSr19GWW4x?usp=sharing)

## Installation instructions
We suggest to use the [docker image](https://hub.docker.com/r/chaneyk/spin) we provide that has all dependencies
compiled and preinstalled.

Alternatively you can install locally:
```
git clone https://github.com/nkolot/SPIN.git
cd SPIN
git checkout origin/cuda11_fix
```

You can create a `python3` virtual environment and install all the relevant dependencies. You can adapt the dependencies based on your CUDA version:
```
python3.10 -m venv spin
source spin/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

There is also the option of using a conda environment:
```
conda create --name spin python=3.10
conda activate spin
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

If you choose to use a virtual environment, please look at the [instructions](https://pyrender.readthedocs.io/en/latest/install/index.html) for installing pyrender. 

After finishing with the installation, you can continue with running the demo/evaluation/training code.
In case you want to evaluate our approach on Human3.6M, you also need to manually install the [pycdf package of the spacepy library](https://pythonhosted.org/SpacePy/pycdf.html) to process some of the original files. If you face difficulties with the installation, you can find more elaborate instructions [here](https://stackoverflow.com/questions/37232008/how-read-common-data-formatcdf-in-python).

## Fetch data
We provide a script to fetch the necessary data for training and evaluation. You need to run:
```
./fetch_data.sh
```
The GMM prior is trained and provided by the original [SMPLify work](http://smplify.is.tue.mpg.de/), while the implementation of the GMM prior function follows the [SMPLify-X work](https://github.com/vchoutas/smplify-x). Please respect the license of the respective works.

Besides these files, you also need to download the *SMPL* model. You will need the [neutral model](http://smplify.is.tue.mpg.de) for training and running the demo code, while the [male and female models](http://smpl.is.tue.mpg.de) will be necessary for evaluation on the 3DPW dataset. Please go to the websites for the corresponding projects and register to get access to the downloads section. In case you need to convert the models to be compatible with python3, please follow the instructions [here](https://github.com/vchoutas/smplx/tree/master/tools).

## Final fits
We also release the improved fits that our method produced at the end of SPIN training. You can download them from [here](http://visiondata.cis.upenn.edu/spin/spin_fits.tar.gz). Each .npz file contains the pose and shape parameters of the SMPL model for the training examples, following the order of the training .npz files. For each example, a flag is also included, indicating whether the quality of the fit is acceptable for training (following an automatic heuristic based on the joints reprojection error).

## Run demo code
To run our method, you need a bounding box around the person. The person needs to be centered inside the bounding box and the bounding box should be relatively tight. You can either supply the bounding box directly or provide an [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) detection file. In the latter case we infer the bounding box from the detections.

In summary, we provide 3 different ways to use our demo code and models:
1. Provide only an input image (using ```--img```), in which case it is assumed that it is already cropped with the person centered in the image.
2. Provide an input image as before, together with the OpenPose detection .json (using ```--openpose```). Our code will use the detections to compute the bounding box and crop the image.
3. Provide an image and a bounding box (using ```--bbox```). The expected format for the json file can be seen in ```examples/im1010_bbox.json```.

Example with OpenPose detection .json
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.jpg --openpose=examples/im1010_openpose.json
```
Example with predefined Bounding Box
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.jpg --bbox=examples/im1010_bbox.json
```
Example with cropped and centered image
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.jpg
```

Running the previous command will save the results in ```examples/im1010_{shape,shape_side}.png```. The file  ```im1010_shape.png``` shows the overlayed reconstruction of the model on the image.  We also render a side view, saved in ```im1010_shape_side.png```.

## Run evaluation code
Besides the demo code, we also provide code to evaluate our models on the datasets we employ for our empirical evaluation. Before continuing, please make sure that you follow the [details for data preprocessing](datasets/preprocess/README.md).

Example usage:
```
python3 eval.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p1 --log_freq=20
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. 3DPW ```--dataset=3dpw```
4. LSP ```--dataset=lsp```
5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```

You can also save the results (predicted SMPL parameters, camera and 3D pose) in a .npz file using ```--result=out.npz```.

For the MPI-INF-3DHP dataset specifically, we include evaluation code only for MPJPE (before and after alignment). If
you want to evaluate on all metrics reported in the paper you should use the official MATLAB test code provided with the
dataset together with the saved detections.

## Run training code
Due to license limitiations, we cannot provide the SMPL parameters for Human3.6M (recovered using [MoSh](http://mosh.is.tue.mpg.de)). Even if you do not have access to these parameters, you can still use our training code using data from the other datasets. Again, make sure that you follow the [details for data preprocessing](datasets/preprocess/README.md).

Example usage:
```
python3 train.py --name train_example --pretrained_checkpoint=data/model_checkpoint.pt --run_smplify
```
You can view the full list of command line options by running `python3 train.py --help`. The default values are the ones used to train the models in the paper.
Running the above command will start the training process. It will also create the folders `logs` and `logs/train_example` that are used to save model checkpoints and Tensorboard logs.
If you start a Tensborboard instance pointing at the directory `logs` you should be able to look at the logs stored during training.

## Citing
If you find this code useful for your research or the use data generated by our method, please consider citing the following paper:

	@Inproceedings{kolotouros2019spin,
	  Title          = {Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop},
	  Author         = {Kolotouros, Nikos and Pavlakos, Georgios and Black, Michael J and Daniilidis, Kostas},
	  Booktitle      = {ICCV},
	  Year           = {2019}
	}
