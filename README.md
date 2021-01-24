# Towards Markerless Surgical Tool and Hand Pose Estimation: HandObjectNet Baseline

- [Project page](http://medicalaugmentedreality.org/handobject.html) <!-- - [Paper](http://arxiv.org/abs/2004.13449) -->
- [Synthetic Grasp Generation](https://github.com/jonashein/grasp_generator)
- [Synthetic Grasp Rendering](https://github.com/jonashein/grasp_renderer)
- [Real Dataset Generation](https://github.com/jonashein/handobject_dataset_creator)
- [HandObjectNet Baseline](https://github.com/jonashein/handobjectnet_baseline)
- [PVNet Baseline](https://github.com/jonashein/pvnet_baseline)
- [Combined Model Baseline](https://github.com/jonashein/baseline_combination)

## Table of Content

- [Setup](#setup)
- [Demo](#demo)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citations](#citations)

## Setup

### Download and Install Code

Retrieve the code
```sh
git clone https://github.com/jonashein/handobjectnet_baseline.git
cd handobjectnet_baseline
```

Create and activate the virtual environment with python dependencies
```sh
conda env create --file=environment.yml
conda activate handobject_env
```

### Download the MANO Model Files

- Go to [MANO website](http://mano.is.tue.mpg.de/)
- Create an account by clicking *Sign Up* and provide your information
- Download Models and Code (the downloaded file should have the format mano_v*_*.zip). Note that all code and data from this download falls under the [MANO license](http://mano.is.tue.mpg.de/license).
- unzip and copy the content of the *models* folder into the `assets/mano` folder

- Your structure should look like this:

```
handobjectnet_baseline/
  assets/
    mano/
      MANO_LEFT.pkl
      MANO_RIGHT.pkl
      fhb_skel_centeridx0.pkl
      fhb_skel_centeridx9.pkl
```

### Download Datasets
Download the synthetic and real dataset from the [project page](http://medicalaugmentedreality.org/handobject.html), 
or run:
```sh
cd data/
wget http://medicalaugmentedreality.org/datasets/syn_colibri_v1.zip
wget http://medicalaugmentedreality.org/datasets/real_colibri_v1.zip
unzip syn_colibri_v1.zip
unzip real_colibri_v1.zip
cd ../
```

## Demo
We provide pretrained models for our synthetic and real datasets, which can be downloaded [here](https://drive.google.com/file/d/1W71jGBdlrTUP8Ga3a8OzEYvhEjXiz7TH/view?usp=sharing).

Download the checkpoints and copy the `handobjectnet_*` directories to `handobjectnet_baseline/checkpoints/`:
```sh
cd checkpoints
wget https://drive.google.com/file/d/1W71jGBdlrTUP8Ga3a8OzEYvhEjXiz7TH/view?usp=sharing
unzip handobjectnet.zip
cd ../
```

To evaluate the pretrained model on the synthetic test set, run:
```sh
python3 trainmeshreg.py --block_rot --train_dataset real_colibri_v1 --val_dataset real_colibri_v1 --val_split test --evaluate --display_freq 1 --resume checkpoints/handobjectnet_pretrained_cv0/model_best.pth
```

To evaluate the refined model on the real test set, run:
```sh
python3 trainmeshreg.py --block_rot --train_dataset real_colibri_v1 --val_dataset real_colibri_v1 --val_split test --evaluate --display_freq 1 --resume checkpoints/handobjectnet_refined_cv0/model_best.pth
```
The training and evaluation will be stored with a timestamp at `checkpoints/DATASETNAME_train_mini1/YYYY_MM_DD_HH_mm/`.

Last, compute the metric averages:
```sh
python3 compute_metrics.py -m "path/to/metrics.pkl"
```

## Training

Train a model from scratch:
```sh
python3 trainmeshreg.py --block_rot --train_dataset real_colibri_v1 --val_dataset real_colibri_v1 --val_split val
```

## Evaluation

Run the evaluation:
```sh
python3 trainmeshreg.py --block_rot ---train_dataset real_colibri_v1 --val_dataset real_colibri_v1 --val_split test --evaluate --display_freq 1 --resume path/to/model_checkpoint.pth
```

The training and evaluation will be stored with a timestamp at 
```
checkpoints/DATASETNAME_train_mini1/YYYY_MM_DD_HH_mm/
  images/
  opt.txt
  metrics.pkl
  training.html
```
The training progress can be monitored using the interactive plots stored in `training.html` (updated after each validation pass). 
Qualitative results can be found in the `images` subdirectory. 

To compute the metric averages, run:
```sh
python3 compute_metrics.py -m "path/to/metrics.pkl"
```

## Citations

If you find this code useful for your research, please consider citing:

* the publication that this code was adapted for
```
@inproceedings{hein21_towards,
  title     = {Towards Markerless Surgical Tool and Hand Pose Estimation},
  author    = {Hein, Jonas and Seibold, Matthias and Bogo, Federica and Farshad, Mazda and Pollefeys, Marc and Fürnstahl, Philipp and Navab, Nassir},
  booktitle = {IPCAI},
  year      = {2021}
}
```

* the publication it builds upon and that this code was originally developed for
```
@inproceedings{hasson20_handobjectconsist,
  title     = {Leveraging Photometric Consistency over Time for Sparsely Supervised Hand-Object Reconstruction},
  author    = {Hasson, Yana and Tekin, Bugra and Bogo, Federica and Laptev, Ivan and Pollefeys, Marc and Schmid, Cordelia},
  booktitle = {CVPR},
  year      = {2020}
}
```
