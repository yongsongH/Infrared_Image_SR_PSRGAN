# [PSRGAN](https://ieeexplore.ieee.org/abstract/document/9424970/) 

Official PyTorch implementation of the paper 
[Infrared Image Super-Resolution via Transfer Learning and PSRGAN](https://ieeexplore.ieee.org/abstract/document/9424970/) accepted in [*IEEE SPL*](https://signalprocessingsociety.org/tags/ieee-spl-article).

:bell:[Yongsong HUANG's Homepage](https://hyongsong.work/) :pushpin:

## Introduction

Recent advances in single image super-resolution (SISR) demonstrate the power of deep learning for achieving better performance. Because it is costly to recollect the training data and retrain the model for infrared (IR)  image super-resolution, the availability of only a few samples for restoring IR images presents an important challenge in the field of SISR. To solve this problem, we first propose the progressive super-resolution generative adversarial network (PSRGAN)  that includes the main path and branch path.  The depthwise  residual  block (DWRB)   is used to represent the features of the  IR  image in the main path. Then, the novel  shallow  lightweight  distillation  residual  block  (SLDRB)  is used to extract the features of the readily available visible image in the other path. Furthermore, inspired by transfer learning, we propose the multistage transfer learning strategy for bridging the gap between different high-dimensional feature spaces that can improve the PSRGAN performance. Finally, quantitative and qualitative evaluations of two public datasets show that PSRGAN can achieve better results compared to the SR methods.

## Approach overview

![PSRGAN](https://user-images.githubusercontent.com/23012102/132626645-154ae3f0-db4d-4bac-b0e8-4c8f2a5baf8e.png)

## Main results

![vis](https://user-images.githubusercontent.com/23012102/132626774-299f9dc1-e2da-440f-8189-f39172601396.png)

## Requirements and dependencies
 * Python  3.7
 * Pytorch 0.4.1
 * CUDA Version 10.2
 * TITAN X (Pascal)
 * Win10

## Dataset prepare

Please check my [homepage](https://hyongsong.work/).

## Model

Pre-trained models can be downloaded from this [site](https://figshare.com/articles/dataset/Pre-trained_models/16591973).

## Evaluation
Creating a new folder named `model_zoo` is necessary, 
please check the [log file](https://github.com/yongsongH/Infrared-Image_PSRGAN/blob/master/results/results-A_75000_G/results-A_75000_G.log) for more information about the settings.

Setting up the following directory structure:

    .
    ├── model_zoo                   
    |   ├──75000_G         # X4
    |   |——5000_G          # X2 
    
***
Run 
```
  main_test_kdsrgan.py
```

## Citation

```
@ARTICLE{9424970, 
author={Huang, Yongsong and Jiang, Zetao and Lan, Rushi and Zhang, 
Shaoqin and Pi, Kui}, 
journal={IEEE Signal Processing Letters}, 
title={Infrared Image Super-Resolution via Transfer Learning 
and PSRGAN}, 
year={2021}, 
volume={28}, 
number={}, 
pages={982-986}, 
doi={10.1109/LSP.2021.3077801}}
```

## Contact

If you meet any problems, please describe them and contact me. 

hyongsong.work@gmail.com

**Impolite emails are not welcome. Thank you for understanding.**

## Acknowledgement
Thanks to [Kai Zhang](https://scholar.google.com.hk/citations?user=0RycFIIAAAAJ&hl) for his work.
