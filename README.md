
# PSMB-Net
Recently, the U-Shaped networks has been widely explored in remote sensing image dehazing and obtained promising performance. However, most of the existing dehazing methods based on U-Shaped framework lack the reconstruction constraints of haze areas, which is particularly important to restore haze-free images. Moreover, their encoding and decoding layers cannot effectively fuse multi-scale features, resulting in deviations in the color and texture of the dehazing image. To address these issues, in this paper, we propose a Partial Siamese with Multiscale Bi-codec Dehazing Network (PSMB-Net) which is mainly composed of a Partial Siamese Framework (PSF) and a Multiscale Bi-codec Information Fusion (MBIF) module. Specifically, the PSF is proposed to create dehazing prior information to guide the network to build Siamese constraints and achieve improved dehazing results. Furthermore, we design a MBIF module which can enhance feature extraction, and the multi-scale information is used to improve the reconstruction ability of the network for the color and texture of the dehazing image. Experimental results on challenging benchmark datasets demonstrate the superiority of our PSMB-Net over state-of-the-art image dehazing methods.
<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/thislzm/PSMB-Net/">
    <img src="images/psf.png" alt="Logo" width="800" height="500">
  </a>
  <h3 align="center">Partial Siamese Framework</h3>
  <p align="center">
  <a href="https://github.com/thislzm/PSMB-Net/">
    <img src="images/mbif.png" alt="Logo" width="1000" height="640">
  </a>
  </p>
  <h3 align="center">Multiscale Bi-codec Information Fusion module</h3>

  <p align="center">
    Partial Siamese Networks with Multiscale Bi-codec Information Fusion Module for Remote Sensing Single Image Dehazing
    <br />
    <a href="https://github.com/thislzm/PSMB-Net"><strong>Exploring the documentation for PSMB-Net »</strong></a>
    <br />
    <br />
    <a href="https://github.com/thislzm/PSMB-Net">Check Demo</a>
    ·
    <a href="https://github.com/thislzm/PSMB-Net/issues">Report Bug</a>
    ·
    <a href="https://github.com/thislzm/PSMB-Net/issues">Pull Request</a>
  </p>

</p>

## Contents

- [Dependencies](#dependences)
- [Filetree](#filetree)
- [Pretrained Model](#pretrained-weights-and-dataset)
- [Train](#train)
- [Test](#test)
- [Clone the repo](#clone-the-repo)
- [Qualitative Results](#qualitative-results)
  - [Results on HRSD-DHID remote sensing Dehazing Challenge testing images:](#results-on-hrsd-dhid-remote-sensing-dehazing-challenge-testing-images)
  - [Results on HRSD-LHID remote sensing Dehazing Challenge testing images:](#results-on-hrsd-lhid-remote-sensing-dehazing-challenge-testing-images)
  - [Results on Statehaze1k-Thin remote sensing Dehazing Challenge testing images:](#results-on-statehaze1k-thin-remote-sensing-dehazing-challenge-testing-images)
  - [Results on Statehaze1k-Moderate remote sensing Dehazing Challenge testing images:](#results-on-statehaze1k-moderate-remote-sensing-dehazing-challenge-testing-images)
  - [Results on Statehaze1k-Thick remote sensing Dehazing Challenge testing images:](#results-on-statehaze1k-thick-remote-sensing-dehazing-challenge-testing-images)
  - [Results on NTIRE 2021 NonHomogeneous Dehazing Challenge testing images:](#results-on-ntire-2021-nonhomogeneous-dehazing-challenge-testing-images)
  - [Results on RESIDE-Outdoor Dehazing Challenge testing images:](#results-on-reside-outdoor-dehazing-challenge-testing-images)
- [Copyright](#copyright)
- [Thanks](#thanks)

### Dependences

1. Pytorch 1.8.0
2. Python 3.7.1
3. CUDA 11.7
4. Ubuntu 18.04

### Filetree

```
├── README.md
├── /PSMB-Net/
|  ├── train.py
|  ├── test.py
|  ├── Model.py
|  ├── Model_util.py
|  ├── perceptual.py
|  ├── train_dataset.py
|  ├── test_dataset.py
|  ├── utils_test.py
|  ├── make.py
│  ├── /pytorch_msssim/
│  │  ├── __init__.py
│  ├── /datasets_train/
│  │  ├── /hazy/
│  │  ├── /clean/
│  ├── /datasets_test/
│  │  ├── /hazy/
│  │  ├── /clean/
│  ├── /output_result/
├── LICENSE.txt
└── /images/
```

### Pretrained Weights and Dataset

Download our model weights on Baidu cloud disk: https://pan.baidu.com/s/1dePHGG4MYvyuLW5rZ0D8VA?pwd=lzms

Download our test datasets on Baidu cloud disk: https://pan.baidu.com/s/1HK1oy4SjZ99N-Dh-8_s0hA?pwd=lzms


### Train

```shell
python train.py -train_batch_size 4 --gpus 0 --type 5
```

### Test

 ```shell
python test.py --gpus 0 --type 5
 ```

### Clone the repo

```sh
git clone https://github.com/thislzm/PSMB-Net.git
```

### Qualitative Results

#### Results on HRSD-DHID remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/DHID.png" style="display: inline-block;" />
</div>

#### Results on HRSD-LHID remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/LHID.png" style="display: inline-block;" />
</div>

#### Results on Statehaze1k-Thin remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/thin.png" style="display: inline-block;" />
</div>

#### Results on Statehaze1k-Moderate remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/moderate.png" style="display: inline-block;" />
</div>

#### Results on Statehaze1k-Thick remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/thick.png" style="display: inline-block;" />
</div>

#### Results on NTIRE 2021 NonHomogeneous Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/nhhaze.png" style="display: inline-block;" />
</div>

#### Results on RESIDE-Outdoor Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/reside.png" style="display: inline-block;" />
</div>




### Copyright

The project has been licensed by MIT. Please refer to for details. [LICENSE.txt](https://github.com/thislzm/PSMB-Net/LICENSE.txt)

### Thanks


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)


<!-- links -->
[your-project-path]:thislzm/PSMB-Net
[contributors-shield]: https://img.shields.io/github/contributors/thislzm/PSMB-Net.svg?style=flat-square
[contributors-url]: https://github.com/thislzm/PSMB-Net/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/thislzm/PSMB-Net.svg?style=flat-square
[forks-url]: https://github.com/thislzm/PSMB-Net/network/members
[stars-shield]: https://img.shields.io/github/stars/thislzm/PSMB-Net.svg?style=flat-square
[stars-url]: https://github.com/thislzm/PSMB-Net/stargazers
[issues-shield]: https://img.shields.io/github/issues/thislzm/PSMB-Net.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/thislzm/PSMB-Net.svg
[license-shield]: https://img.shields.io/github/license/thislzm/PSMB-Net.svg?style=flat-square
[license-url]: https://github.com/thislzm/PSMB-Net/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian




