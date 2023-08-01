# Learning Human Mesh Recovery in 3D Scenes

### [Project Page](https://zju3dv.github.io/sahmr/) | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_Learning_Human_Mesh_Recovery_in_3D_Scenes_CVPR_2023_paper.pdf)

![teaser](https://zju3dv.github.io/sahmr/images/teaser_homepage.jpg)

> [Learning Human Mesh Recovery in 3D Scenes](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_Learning_Human_Mesh_Recovery_in_3D_Scenes_CVPR_2023_paper.pdf)  
> Zehong Shen, Zhi Cen, Sida Peng, Qing Shuai, Hujun Bao, Xiaowei Zhou  
> CVPR 2023

### TODO List and ETA
- [x] Inference code and weights
- [ ] Training code (expected 2023-7-12)

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
You need to agree and follow the [RICH dataset license](https://rich.is.tue.mpg.de/license.html) and the [PROX dataset license](https://prox.is.tue.mpg.de/license.html) to use the data.

Here, we provide the minimal and pre-propcessed `RICH/sahmr_support` and `PROX/quantitative/sahmr_support` for reproducing the metrics in the paper. By downloading, you agree to the [RICH dataset license](https://rich.is.tue.mpg.de/license.html) and the [PROX dataset license](https://prox.is.tue.mpg.de/license.html).

If you want to train the model, you still need to submit a request to the authors from MPI and use their links for downloading the full datasets.

### Link weights and data to the project folder
```bash
ln -s path-to-models(smpl-models) models

mkdir datasymlinks
ln -s path-to-release(weights) datasymlinks/release

# the RICH folder should contain the original RICH dataset in the training phase,
# and the `RICH/sahmr_support` is enough for evaluation
mkdir -p datasymlinks/RICH 
ln -s path-to-rich-sahmr_support datasymlinks/RICH/sahmr_support

# the `PROX/quantitative/sahmr_support` is enough for evaluation
mkdir -p datasymlinks/PROX/quantitative
ln -s path-to-prox-sahmr_support datasymlinks/PROX/quantitative/sahmr_support
```
</details>


## Usage

<details><summary>Evaluation</summary>

```bash
# RICH model
python tools/dump_results.py -c configs/pose/sahmr_eval/rich.yaml -d datasymlinks/release/sahmr_rich_e30.pth 
python tools/eval_results.py -c configs/pose/sahmr_eval/rich.yaml

# PROX model
python tools/dump_results.py -c configs/pose/sahmr_eval/prox.yaml -d datasymlinks/release/sahmr_prox_e30.pth --data_name prox_quant
python tools/eval_results.py -c configs/pose/sahmr_eval/prox.yaml --data_name prox_quant
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