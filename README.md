
# PSBDN
In this paper, we propose a Partial Siamese and Bi-codec Dehazing Network (PSBDN) which is mainly constructed by a Partial Siamese Dehazing Framework (PSDF) and a Bi-codec Multi-scale Information Fusion Module (BMIFM). Specifically, the PSDF is proposed to create dehazing prior information to guide the network to build siamese constraints and achieve better dehazing results. Furthermore, we design a BMIFM which can enhance feature extraction and the multi-scale information is used to improve the reconstruction ability of the network for the color and texture of the dehazing image.

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/thislzm/PSBDN/">
    <img src="images/psdf.png" alt="Logo" width="80" height="80">
  </a>
  <h3 align="center">Partial Siamese Dehazing Framework</h3>
  <p align="center">
  <a href="https://github.com/thislzm/PSBDN/">
    <img src="images/psdf.png" alt="Logo" width="80" height="80">
  </a>
  </p>
  <h3 align="center">Bi-codec Multi-scale Information Fusion Module</h3>

  <p align="center">
    Partial Siamese Network with Bi-codec Multi-scale Information Fusion Module for Single Image Dehazing
    <br />
    <a href="https://github.com/thislzm/PSBDN"><strong>Exploring the documentation for PSBDN »</strong></a>
    <br />
    <br />
    <a href="https://github.com/thislzm/PSBDN">Check Demo</a>
    ·
    <a href="https://github.com/thislzm/PSBDN/issues">Report Bug</a>
    ·
    <a href="https://github.com/thislzm/PSBDN/issues">Pull Request</a>
  </p>

</p>

 
## Contents

- [Dependencies](#dependences)
- [filetree](#filetree)
- [Train](#train)
- [Test](#test)
- [Clone the repo](#clone-the-repo)
- [Qualitative Results](#qualitative-results)
  - [Results on NTIRE 2021 NonHomogeneous Dehazing Challenge testing images:](#results-on-ntire-2021-nonhomogeneous-dehazing-challenge-testing-images)
  - [Results on RESIDE-Outdoor Dehazing Challenge testing images:](#results-on-reside-outdoor-dehazing-challenge-testing-images)
  - [Results on Statehaze1k Dehazing Challenge testing images:](#results-on-statehaze1k-dehazing-challenge-testing-images)
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
├── /PSBDN/
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
### Train

```shell
python train.py --data_dir data -train_batch_size 8 --model_save_dir train_result
```

### Test

 ```shell
python test.py --model_save_dir results
 ```

### Clone the repo

```sh
git clone https://github.com/thislzm/PSBDN.git
```

### Qualitative Results

#### Results on NTIRE 2021 NonHomogeneous Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/nhhaze.png" style="display: inline-block;" />
</div>

#### Results on RESIDE-Outdoor Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/reside.png" style="display: inline-block;" />
</div>

#### Results on Statehaze1k Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/haze1k.png" style="display: inline-block;" />
</div>


### Copyright

The project has been licensed by MIT. Please refer to for details. [LICENSE.txt](https://github.com/thislzm/PSBDN/LICENSE.txt)

### Thanks


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)


<!-- links -->
[your-project-path]:thislzm/PSBDN
[contributors-shield]: https://img.shields.io/github/contributors/thislzm/PSBDN.svg?style=flat-square
[contributors-url]: https://github.com/thislzm/PSBDN/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/thislzm/PSBDN.svg?style=flat-square
[forks-url]: https://github.com/thislzm/PSBDN/network/members
[stars-shield]: https://img.shields.io/github/stars/thislzm/PSBDN.svg?style=flat-square
[stars-url]: https://github.com/thislzm/PSBDN/stargazers
[issues-shield]: https://img.shields.io/github/issues/thislzm/PSBDN.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/thislzm/PSBDN.svg
[license-shield]: https://img.shields.io/github/license/thislzm/PSBDN.svg?style=flat-square
[license-url]: https://github.com/thislzm/PSBDN/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian




