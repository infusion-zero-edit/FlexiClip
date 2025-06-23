# FlexiClip: Locality-Preserving Free-Form Character Animation (ICML'25)
<a href="https://creative-gen.github.io/flexiclip.github.io/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://openreview.net/forum?id=xtxCM4XZ82"><img src="https://img.shields.io/badge/OpenReview-ICML25-b31b1b.svg"></a>
<a href="https://www.apache.org/licenses/LICENSE-2.0.txt"><img src="https://img.shields.io/badge/License-Apache-yellow"></a>
<!-- Official implementation. -->
<br>
<p align="center">
<img src="repo_images/FlexiClip_Poster.png" width="90%"/>  
  
> <a href="https://livesketch.github.io/">**FlexiClip: Locality-Preserving Free-Form Character Animation**</a>
>
<a href="https://www.linkedin.com/in/anant-khandelwal-iitd/">Anant Khandelwal</a>,
> <br>
>  Given a still image in vector format and a text prompt describing a desired action, our method automatically animates the drawing with respect to the prompt
</p>

# Setup
```
git clone https://github.com/kingnobro/FlexiClip.git
cd FlexiClip
```

## Environment
To set up our environment, please run:
```
conda env create -f environment.yml
```
Next, you need to install diffvg:
```bash
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
python setup.py install
```

## Run
Single-layer animation:
```
bash scripts/run_flexiclip.sh
```
Multi-layer animation:
```
bash scripts/run_layer_flexiclip.sh
```


## Keypoint Detection
For humans, we use [UniPose](https://github.com/IDEA-Research/UniPose?tab=readme-ov-file). Take a look at our example SVG input. Specifically, we merge 5 points on face (`tools.merge_unipose_ske.py`) due to the limitations of mesh-based algorithms in altering emotions, alongside the video diffusion model's inability to precisely direct facial expressions.

For broader categories, first install scikit-geometry:
```
conda install -c conda-forge scikit-geometry
```

Then put your SVG files under `svg_input`. For example, if your download SVG from the Internet and its name is `cat`, then you create the folder `svg_input/cat` and there is a file `cat.svg` in this folder.

Then, modify the `target` in `preprocess/keypoint_detection.py` and run:
```
python -m preprocess.keypoint_detection
```
You can adjust `epsilon`, `max_iter` and `factor` to adjust the complexity of the skeleton.

## SVG Preprocess
For SVG downloaded from the Internet, there may exist complex grammars.

For a file `cat_input.svg`, we first use [picosvg](https://github.com/googlefonts/picosvg) to remove grammars like `group` and `transform`:
```
picosvg cat_input.svg > cat.svg
```
Then you modify the SVG to `256x256` by running:
```
python -m preprocess.svg_resize 
```
## Citation
If you find this useful for your research, please cite the following:
```bibtex
@inproceedings{
khandelwal2025flexiclip,
title={FlexiClip: Locality-Preserving Free-Form Character Animation},
author={Anant Khandelwal},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=xtxCM4XZ82}
}
```
