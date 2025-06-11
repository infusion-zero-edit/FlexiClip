# AniClipart: Clipart Animation with Text-to-Video Priors

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

