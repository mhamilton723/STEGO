# STEGO: Unsupervised Semantic Segmentation by Distilling Feature Correspondences
### [Project Page](https://mhamilton.net/stego.html) | [Paper](https://arxiv.org/abs/2203.08414) | [Video](https://aka.ms/stego-video) | [ICLR 2022](https://iclr.cc/virtual/2022/poster/6068) 

	
[Mark Hamilton](https://mhamilton.net/),
[Zhoutong Zhang](https://ztzhang.info/),
[Bharath Hariharan](http://home.bharathh.info/),
[Noah Snavely](https://www.cs.cornell.edu/~snavely/),
[William T. Freeman](https://billf.mit.edu/about/bio)

This is the official implementation of the paper "Unsupervised Semantic Segmentation by Distilling Feature Correspondences".


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mhamilton723/STEGO/blob/master/src/STEGO_Colab_Demo.ipynb) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-semantic-segmentation-by-2/unsupervised-semantic-segmentation-on)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-on?p=unsupervised-semantic-segmentation-by-2)\
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-semantic-segmentation-by-2/unsupervised-semantic-segmentation-on-coco-4)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-on-coco-4?p=unsupervised-semantic-segmentation-by-2) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-semantic-segmentation-by-2/unsupervised-semantic-segmentation-on-potsdam-1)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-on-potsdam-1?p=unsupervised-semantic-segmentation-by-2)


[![Overview Video](https://marhamilresearch4.blob.core.windows.net/stego-public/graphics/STEGO%20Header%20video%20(2).jpg)](https://youtu.be/NPub4E4o8BA)

## Contents
<!--ts-->
   * [Install](#install)
   * [Evaluation](#evaluation)
   * [Training](#training)
      * [Bringing your own data](#bringing-your-own-data)
   * [Understanding STEGO](#understanding-stego)
      * [Unsupervised Semantic Segmentation](#unsupervised-semantic-segmentation)
      * [Deep features connect objects across images](#deep-features-connect-objects-across-images)
      * [The STEGO architecture](#the-stego-architecture)
      * [Results](#results)
   * [Citation](#citation)
   * [Contact](#contact)
<!--te-->

## Install

### Clone this repository:
```shell script
git clone https://github.com/mhamilton723/STEGO.git
cd STEGO
```

### Install Conda Environment
Please visit the [Anaconda install page](https://docs.anaconda.com/anaconda/install/index.html) if you do not already have conda installed

```shell script
conda env create -f environment.yml
conda activate stego
```

### Download Pre-Trained Models

```shell script
cd src
python download_models.py
```

### Download Datasets

First, change the `pytorch_data_dir` variable to your 
systems pytorch data directory where datasets are stored. 

```shell script
python download_datasets.py
```

Once downloaded please navigate to your pytorch data dir and unzip the resulting files:

```shell script
cd /YOUR/PYTORCH/DATA/DIR
unzip cocostuff.zip
unzip cityscapes.zip
unzip potsdam.zip
unzip potsdamraw.zip
```


## Evaluation

To evaluate our pretrained models please run the following in `STEGO/src`:
```shell script
python eval_segmentation.py
```
One can change the evaluation parameters and model by editing [`STEGO/src/configs/eval_config.yml`](src/configs/eval_config.yml)

## Training

To train STEGO from scratch, please first generate the KNN indices for the datasets of interest. Note that this requires generating a cropped dataset first, and you may need to modify `crop datasets.py` to specify the dataset that you are cropping:

```shell script
python crop_datasets.py
python precompute_knns.py
```

Then you can run the following in `STEGO/src`:
```shell script
python train_segmentation.py
```
Hyperparameters can be adjusted in [`STEGO/src/configs/train_config.yml`](src/configs/train_config.yml)

To monitor training with tensorboard run the following from `STEGO` directory:

```shell script
tensorboard --logdir logs
```

### Bringing your own data

To train STEGO on your own dataset please create a directory in your pytorch data root with the following structure. Note, if you do not have labels, omit the `labels` directory from the structure:

```
dataset_name
|── imgs
|   ├── train
|   |   |── unique_img_name_1.jpg
|   |   └── unique_img_name_2.jpg
|   └── val
|       |── unique_img_name_3.jpg
|       └── unique_img_name_4.jpg
└── labels
    ├── train
    |   |── unique_img_name_1.png
    |   └── unique_img_name_2.png
    └── val
        |── unique_img_name_3.png
        └── unique_img_name_4.png
```

Next in [`STEGO/src/configs/train_config.yml`](src/configs/train_config.yml) set the following parameters:

```yaml
dataset_name: "directory"
dir_dataset_name: "dataset_name"
dir_dataset_n_classes: 5 # This is the number of object types to find
```

If you want to train with cropping to increase spatial resolution run our [cropping utility](src/crop_datasets.py).

Finally, uncomment the custom dataset code and run `python precompute_knns.py`
 from `STEGO\src` to generate the prerequisite KNN information for the custom dataset.
 
You can now train on your custom dataset using:
```shell script
python train_segmentation.py
```

## Understanding STEGO

### Unsupervised semantic segmentation
Real-world images can be cluttered with multiple objects making classification feel arbitrary. Furthermore, objects in the real world don't always fit in bounding boxes. Semantic segmentation methods aim to avoid these challenges by assigning each pixel of an image its own class label. Conventional semantic segmentation methods are notoriously difficult to train due to their dependence on densely labeled images, which can take 100x longer to create than bounding boxes or class annotations. This makes it hard to gather sizable and diverse datasets impossible in domains where humans don't know the structure a-priori. We sidestep these challenges by learning an ontology of objects with pixel-level semantic segmentation through only self-supervision.

### Deep features connect objects across images
Self-supervised contrastive learning enables algorithms to learn intelligent representations for images without supervision. STEGO builds on this work by showing that representations from self-supervised visual transformers like  Caron et. al.’s  DINO are already aware of the relationships between objects. By computing the cosine similarity between image features, we can see that similar semantic regions such as grass, motorcycles, and sky are “linked” together by feature similarity.

![Feature connection GIF](https://mhamilton.net/images/Picture3.gif)


### The STEGO architecture
The STEGO unsupervised segmentation system learns by distilling correspondences between images into a set of class labels using a contrastive loss. In particular we aim to learn a segmentation that respects the induced correspondences between objects. To achieve this we train a shallow segmentation network on top of the DINO ViT backbone with three contrastive terms that distill connections between an image and itself, similar images, and random other images respectively. If two regions are strongly coupled by deep features we encourage them to share the same class.

![Architecture](results/figures/stego.svg)

### Results

We evaluate the STEGO algorithm on the CocoStuff, Cityscapes, and Potsdam semantic segmentation datasets. Because these methods see no labels, we use a Hungarian matching algorithm to find the best mapping between clusters and dataset classes. We find that STEGO is capable of segmenting complex and cluttered scenes with much higher spatial resolution and sensitivity than the prior art, [PiCIE](https://sites.google.com/view/picie-cvpr2021/home). This not only yields a substantial qualitative improvement, but also more than doubles the mean intersection over union (mIoU). For results on Cityscapes, and Potsdam see [our paper](https://arxiv.org/abs/2203.08414).

![Cocostuff results](results/figures/cocostuff27_results.jpg)


## Citation

```
@inproceedings{hamilton2022unsupervised,
	title={Unsupervised Semantic Segmentation by Distilling Feature Correspondences},
	author={Mark Hamilton and Zhoutong Zhang and Bharath Hariharan and Noah Snavely and William T. Freeman},
	booktitle={International Conference on Learning Representations},
	year={2022},
	url={https://openreview.net/forum?id=SaKO6z6Hl0c}
}
```

## Contact

For feedback, questions, or press inquiries please contact [Mark Hamilton](mailto:markth@mit.edu)
