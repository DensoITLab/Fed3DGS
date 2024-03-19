# Fed3DGS: Scalable 3D Gaussian Splatting with Federated Learning
Official implementation of Fed3DGS in PyTorch.  
arXiv: https://arxiv.org/abs/2403.11460

## Demo
![](https://github.com/DensoITLab/Fed3DGS/blob/asset/gif/lq-building.gif)
![](https://github.com/DensoITLab/Fed3DGS/blob/asset/gif/lq-residence.gif)

Other videos are available [here](https://github.com/DensoITLab/Fed3DGS/tree/asset/gif)

## Introduction
![](https://github.com/DensoITLab/Fed3DGS/blob/asset/images/overview.png)
Fed3DGS is a federated learning framework for 3D reconstruction using 3DGS.
In Fed3DGS, multiple clients collaboratively reconstruct 3D scenes under the orchestration of a central server.
To update global 3DGS with local 3DGS, we propose a distillation-based model update scheme.
We update global 3DGS to minimize the difference between images rendered by local and global 3DGS.
By repeating this update scheme, Fed3DGS continuously improves the global model.

## Pretrained Models
The pretrained models can be downloaded [here](https://d-itlab.s3.ap-northeast-1.amazonaws.com/Fed3DGS/global-models.zip).

## Usage
In the following, please set values to variables (`$DATASET_DIR`, etc.) or replace them according to your environment.

### Setup
Install [Colmap](https://colmap.github.io/) following the [installation](https://colmap.github.io/install.html), which is required to compute initial points of 3DGS.

Clone this repository with `recursive` option:
```sh
git clone --recursive --single-branch -b main https://github.com/DensoITLab/Fed3DGS
cd Fed3DGS
```

Then, create conda environment:
```sh
conda env create --file environment.yml
conda activate fed3dgs
```

### Dataset Preparation
Download the datasets following [the Mega-NeRF repository](https://github.com/cmusatyalab/mega-nerf).

After downloading, run the following code for each dataset:
```sh
python tools/merge_val_train.py -d $DATASET_DIR
```
This code merges training data and validation data because the left half of the validation images is used to train the model in the Mega-NeRF setting.
`$DATASET_DIR` denotes the path to dataset (e.g., `./datasets/building-pixsfm`).

Then, generate image lists of local data by running following code:
```sh
python tools/gen_client_data.py -d $DATASET_DIR -o $IMAGE_LIST_DIR --n-clients $N_CLIENTS
```
The generated image lists are saved in `$IMAGE_LIST_DIR` (e.g., `image-lists/building`).
`$N_CLIENTS` denotes the number of clients (set to 200 or 400 in the paper).

### Training Local Models
After generating image lists of local data, train local models by running following code:
```sh
bash scripts/client_training.sh 0 $N_CLIENTS $COLMAP_RESULTS_DIR $DATASET_DIR $IMAGE_LIST_DIR $LOCAL_MODEL_DIR
```
Colmap results used to initialize 3DGS and trained local models are saved in `$COLMAP_RESULTS_DIR` and `$LOCAL_MODEL_DIR`, respectively.

### Training Global Model
After training local models, run:
```sh
python gaussian-splatting/build_global_model.py -w -o $GLOBAL_MODEL_DIR -m $LOCAL_MODEL_DIR -i $IMAGE_LIST_DIR -data $DATASET_DIR
```
The global model is saved at `${GLOBAL_MODEL_DIR}/global_model.pth`.

### Evaluation
```sh
python eval.py -w -o $OUTPUT_DIR -g ${GLOBAL_MODEL_DIR}/global_model.pth -data $DATASET_DIR
```
The evaluation results are saved in `$OUTPUT_DIR`.

## Citation
```
@article{suzuki2024fed3dgs,
  title={{Fed3DGS: Scalable 3D Gaussian Splatting with Federated Learning}},
  author={Suzuki, Teppei},
  journal={arXiv preprint arXiv:2403.11460},
  year={2024}
}
```

## License
See [LICENCE](./LICENSE).
Note that a large part of this project relies on [the 3DGS repository](https://github.com/graphdeco-inria/gaussian-splatting).
The code related to 3DGS is subject to 3DGS's [licence](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md).
