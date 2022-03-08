# STEGO: Unsupervised Semantic Segmentation by Distilling Feature Correspondences

## Install

#### Clone this repository:
```
git clone https://github.com/mhamilton723/STEGO.git
cd STEGO
```

#### Install Conda Environment
Please visit the [Anaconda install page](https://docs.anaconda.com/anaconda/install/index.html) if you do not already have conda installed

```
conda env create -f environment.yml
conda activate stego
```

#### Download Pre-Trained Models

```
cd src
python download_models.py
```

#### Download Datasets

First, change the `pytorch_data_dir` variable to your 
systems pytorch data directory where datasets are stored. 

```
python download_data.py
cd /YOUR/PYTORCH/DATA/DIR
unzip cocostuff.zip
```


## Run Evaluation

From inside STEGO/src pleas run the following:
```
python eval_segmentation.py
```


