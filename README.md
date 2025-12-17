<p align="center">
<h1 align="center"><strong>N3D-VLM: Native 3D Grounding Enables
Accurate Spatial Reasoning</strong></h1>
<!-- <h3 align="center">Arxiv 2025</h3> -->

<p align="center">
    <a href="https://w-ted.github.io/">Yuxin Wang</a><sup>1,2</sup>,
    <a href="https://www.kelei.site/">Lei Ke</a><sup>2</sup>,
    <a href="https://cyrilsterling.github.io/">Boqiang Zhang</a><sup>2</sup>,
    <a href="https://openreview.net/profile?id=~Tianyuan_Qu2">Tianyuan Qu</a><sup>2,3</sup>,
    <a href="https://hanxunyu.github.io/">Hanxun Yu</a><sup>2,4</sup>,
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
    <a href='https://arxiv.org/abs/25xx.xxxxx' target="_blank"><img src='https://img.shields.io/badge/arXiv-25xx.xxxxx-b31b1b.svg'></a>  
    <a href='https://n3d-vlm.github.io' target="_blank"><img src='https://img.shields.io/badge/Project-Page-Green'></a>  
    <a href='https://huggingface.co/yuxinhk/N3D-VLM' target="_blank">
        <img src='https://img.shields.io/badge/Hugging%20Face-Models-blue'>
    </a>
</div>
</p>




https://github.com/user-attachments/assets/77c2abac-17c8-402a-8fda-ad10262484fe


## Overview
**N3D-VLM** is a unified vision-language models for **native 3D grounding** and **3D spatial reasoning**. By incorporating native 3D grounding, our model enables precise spatial reasoning, allowing users to query object relationships, distances, and attributes directly within complex 3D environments.


## Updates

- **`2025/12/20`**: We released this repo with the pre-trained model and inference code.


## Installation

```
git clone https://github.com/W-Ted/N3D-VLM.git

cd N3D-VLM
conda env create -n n3d_vlm -f n3d-vlm.yaml
```

## Pre-trained model
We provide the pre-trained model [here](https://huggingface.co/yuxinhk/N3D-VLM). 


## Inference 
We provide three examples for inference of N3D-VLM. You could check the source files in `data` directory, where `*.jpg` are the source images and `*.npz` are the monocular point clouds obtained by using [MoGe2](https://github.com/microsoft/moge)
```
# inference 
python demo.py
```

### Demo 1


https://github.com/user-attachments/assets/ce660743-c951-4773-affb-61e532751758


### Demo 2


https://github.com/user-attachments/assets/c02af68b-d95c-4510-abbd-c557e1faf9c7


### Demo 3


https://github.com/user-attachments/assets/1574a4af-280f-4fa5-bf35-07cdd5bc5913


After running the code above, the inference results will be saved in the `outputs` directory, including generated answers in `*.json` format, and 3D grounding results in `*.rrd` format. 
The rrd files can be visualized by using [Rerun](https://rerun.io):
```
rerun outputs/demo1.rrd
```

## Citation

```BibTeX
@article{wang2025n3d,
    title={N3D-VLM: Native 3D Grounding Enables Accurate Spatial Reasoning},
    author={Wang, Yuxin and Ke, Lei and Zhang, Boqiang and Qu, Tianyuan and Yu, Hanxun and Huang, Zhenpeng and Yu, Meng and Xu, Dan and Yu, Dong},
    journal={arXiv preprint arXiv:25xx.xxxxx},
    year={2024}
}
```



