<p align="center">
<h1 align="center"><strong>N3D-VLM: Native 3D Grounding Enables
Accurate Spatial Reasoning in Vision-Language Models</strong></h1>
<!-- <h3 align="center">Arxiv 2025</h3> -->

<p align="center">
    <a href="https://w-ted.github.io/">Yuxin Wang</a><sup>1,2</sup>,
    <a href="https://www.kelei.site/">Lei Ke</a><sup>2</sup>,
    <a href="https://cyrilsterling.github.io/">Boqiang Zhang</a><sup>2</sup>,
    <a href="https://openreview.net/profile?id=~Tianyuan_Qu2">Tianyuan Qu</a><sup>2,3</sup>,
    <a href="https://hanxunyu.github.io/">Hanxun Yu</a><sup>2,4</sup>,
    <br>
    <a href="https://openreview.net/profile?id=~Zhenpeng_Huang1">Zhenpeng Huang</a><sup>2,5</sup>,
    <a href="https://raymond-myu.github.io/">Meng Yu</a><sup>2</sup>,
    <a href="https://www.danxurgb.net/">Dan Xu</a><sup>1✉️</sup>,
    <a href="https://sites.google.com/view/dongyu888/">Dong Yu</a><sup>2</sup>
    <br>
    <sup>1</sup>HKUST,
    <sup>2</sup>Tencent AI Lab,
    <sup>3</sup>CUHK,
    <sup>4</sup>ZJU,
    <sup>5</sup>NJU
</p>

<div align="center">
    <a href='https://arxiv.org/abs/2512.16561' target="_blank"><img src='https://img.shields.io/badge/arXiv-2512.16561-b31b1b.svg'></a>  
    <a href='https://n3d-vlm.github.io' target="_blank"><img src='https://img.shields.io/badge/Project-Page-Green'></a>  
    <a href='https://huggingface.co/yuxinhk/N3D-VLM' target="_blank">
        <img src='https://img.shields.io/badge/Hugging%20Face-Models-blue'>
    </a>
</div>
</p>




https://github.com/user-attachments/assets/077b6a9c-d1b3-4ebe-a9f1-e652b654b7d1


## Overview
**N3D-VLM** is a unified vision-language model for **native 3D grounding** and **3D spatial reasoning**. By incorporating native 3D grounding, our model enables precise spatial reasoning, allowing users to query object relationships, distances, and attributes directly within complex 3D environments.


## Updates

- **`2025/12/19`**: We released this repo with the pre-trained model and inference code.


## Installation

```
git clone --recursive https://github.com/W-Ted/N3D-VLM.git
cd N3D-VLM

conda env create -n n3d_vlm python=3.11 -y
conda activate n3d_vlm
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Pre-trained model
We provide the pre-trained model [here](https://huggingface.co/yuxinhk/N3D-VLM). 


## Inference 
We provide three examples for inference of N3D-VLM. You could check the source files in `data` directory, where `*.jpg` are the source images and `*.npz` are the monocular point clouds obtained by using [MoGe2](https://github.com/microsoft/moge). 
```
# inference 
python demo.py
```

### Demo 1


https://github.com/user-attachments/assets/e86306f2-152d-4337-a8d2-d165a26ce305


### Demo 2


https://github.com/user-attachments/assets/1bc0ee64-7a15-4592-941d-1037a26fb108


### Demo 3


https://github.com/user-attachments/assets/ba7ece12-4288-411d-9964-c676b78c6d5c


After running the code above, the inference results will be saved in the `outputs` directory, including generated answers in `*.json` format, and 3D grounding results in `*.rrd` format. 
The rrd files can be visualized by using [Rerun](https://rerun.io):
```
rerun outputs/demo1.rrd
```

<!-- If you want to do the 3D Detection / Grouding only, please check the prompt in the first example.  -->

## Citation

```BibTeX
@article{wang2025n3d,
    title={N3D-VLM: Native 3D Grounding Enables Accurate Spatial Reasoning in Vision-Language Models},
    author={Wang, Yuxin and Ke, Lei and Zhang, Boqiang and Qu, Tianyuan and Yu, Hanxun and Huang, Zhenpeng and Yu, Meng and Xu, Dan and Yu, Dong},
    journal={arXiv preprint arXiv:2512.16561},
    year={2025}
}
```



