# Learning Human Mesh Recovery in 3D Scenes

### [Project Page](https://zju3dv.github.io/sahmr/) | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_Learning_Human_Mesh_Recovery_in_3D_Scenes_CVPR_2023_paper.pdf)

![teaser](https://zju3dv.github.io/sahmr/images/teaser_homepage.jpg)

> [Learning Human Mesh Recovery in 3D Scenes](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_Learning_Human_Mesh_Recovery_in_3D_Scenes_CVPR_2023_paper.pdf)  
> Zehong Shen, Zhi Cen, Sida Peng, Qing Shuai, Hujun Bao, Xiaowei Zhou  
> CVPR 2023

## Setup

<details><summary>Environment</summary>

```bash
conda create -y -n sahmr python=3.8
conda activate sahmr
pip install -r requirements.txt
pip install -e . 

# torchsparse==1.4.0, please refer to https://github.com/mit-han-lab/torchsparse
sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```
</details>
 

<details><summary>Weights and data</summary>

\
🚩 [Google drive link](https://drive.google.com/drive/folders/1CluXFrJliem1awumjBt7gvitSbSI92cZ?usp=sharing)

### Model Weights
We provide the pretrained rich and prox models for evaluation under the `release` folder.

### RICH/PROX dataset

#### Evaluation
1. You need to agree and follow the [RICH dataset license](https://rich.is.tue.mpg.de/license.html) and the [PROX dataset license](https://prox.is.tue.mpg.de/license.html) to use the data.

2. Here, we provide the minimal and pre-propcessed `RICH/sahmr_support` and `PROX/quantitative/sahmr_support` for reproducing the metrics in the paper. By downloading, you agree to the [RICH dataset license](https://rich.is.tue.mpg.de/license.html) and the [PROX dataset license](https://prox.is.tue.mpg.de/license.html).

#### Training ✨

1. You need to submit a request to the authors from MPI and use their links for downloading the full datasets. 

2. RICH: We use the JPG format image. We downsampled the image to **one-forth** of its original dimensions.

### Link weights and data to the project folder

``` bash
datasymlinks
├── RICH
│   ├── images_ds4      # see comments below
│   │   ├── train
│   │   └── val
│   ├── bodies          # included in the RICH_train.zip
│   │   ├── train
│   │   └── val
│   └── sahmr_support
│       ├── scene_info  # included in the RICH.zip
│       ├── test_split  # included in the RICH.zip
│       ├── train_split # included in the RICH_train.zip
│       └── val_split   # included in the RICH_train.zip
├── PROX                # included in the PROX.zip
└── checkpoints
    ├── release         # included in the `release`
    │   ├── sahmr_rich_e30.pth
    │   └── sahmr_prox_e30.pth
    └── metro           # see comments below
        └── metro_3dpw_state_dict.bin 
```

- `images_ds4`: Please download the *train* and *val* datasets and downsample the images to one-forth of its original dimensions.
- `bodies`: We provide the fitted smplh parameters for each image. We will shift to the original smplx parameters in the future.
- `metro_3dpw_state_dict.bin`: You only need this if you want to do training. 
    <details><summary>Download the pretrained weights of METRO</summary>

    ```bash
    mkdir -p datasymlinks/checkpoints/metro
    # See https://github.com/microsoft/MeshTransformer/blob/main/LICENSE
    # See https://github.com/microsoft/MeshTransformer/blob/main/scripts/download_models.sh
    wget -nc https://datarelease.blob.core.windows.net/metro/models/metro_3dpw_state_dict.bin -O datasymlinks/checkpoints/metro/metro_3dpw_state_dict.bin
    ```
    </details>

```bash
ln -s path-to-models(smpl-models) models

mkdir datasymlinks
mkdir -p datasymlinks/checkpoints
ln -s path-to-release(weights) datasymlinks/checkpoints/release

# the RICH folder should contain the original RICH dataset in the training phase,
# and the `RICH/sahmr_support` is enough for evaluation
mkdir -p datasymlinks/RICH 
ln -s path-to-rich-sahmr_support datasymlinks/RICH/sahmr_support
# for the training parts, please refer to the folder structure above

# the `PROX/quantitative/sahmr_support` is enough for evaluation
mkdir -p datasymlinks/PROX/quantitative
ln -s path-to-prox-sahmr_support datasymlinks/PROX/quantitative/sahmr_support
```
</details>


## Usage

<details><summary>Evaluation</summary>

```bash
# RICH model
python tools/dump_results.py -c configs/pose/sahmr_eval/rich.yaml 
python tools/eval_results.py -c configs/pose/sahmr_eval/rich.yaml

# PROX model
python tools/dump_results.py -c configs/pose/sahmr_eval/prox.yaml
python tools/eval_results.py -c configs/pose/sahmr_eval/prox.yaml
```
</details>

<details><summary>Training</summary>

```bash
# We provide a training example on RICH dataset
python train_net.py -c configs/pose/rich/rcnet.yaml 
python train_net.py -c configs/pose/rich/sahmr.yaml 
```
</details>


## Citation

```bibtex
@article{shen2023sahmr,
    title={Learning Human Mesh Recovery in 3D Scenes},
    author={Shen, Zehong and Cen, Zhi and Peng, Sida and Shuai, Qing and Bao, Hujun and Zhou, Xiaowei},
    journal={CVPR},
    year={2023}
}
```