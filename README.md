# Instructions for use

## Overview
This repository provides functionality for a 3 step conversion process to convert the [U2Net model](https://github.com/xuebinqin/U-2-Net) from its original PyTorch format to TensorFlowLite format.

## Pre-Requisites
1. Install the required dependencies via `pip install -r requirements.txt`
1. Download the pre-trained u2net model [here](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing) and place it in `content/`.
2. Place an image of choice(.jpg) in `content/`
3. Run the `pth_onnx` script. The output model should appear in the root directory.
4. Verify the successfulness of the conversion by examining the console statements.
5. Run the `convert_onnx_tf` script. The converted model will appear in the root directory.
6. Run the `convert_tf_tflite` script. The converted model will appear in the root directory.

## Notes
The repository also includes `model-dimensions.py`, which is a python script that prints the input requirements of a model. This is to facilitiate ease of integration of the model into your program.

## Credits
Source code taken from farmaker47, which can be found [here](https://github.com/farmaker47/Portrait_creator/blob/master/U2Net_to_TFLite.ipynb)