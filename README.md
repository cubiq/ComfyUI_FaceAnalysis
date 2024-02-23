# Face Analysis for ComfyUI

This extension uses [DLib](http://dlib.net/) to calculate the Euclidean and Cosine *distance* between two faces.

Please read the results as follow:

- **Lower values are better**
- The minimum thresholds are: **EUC 0.6**, **COS 0.07**
- In my tests a value of Euc <0.3 is very good

## Installation

The following instructions are written with the assumption that you have [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed, but it's not required.

1. The extension requires `dlib` installation in your ComfyUI environment. Click 'Manager', click 'Install pip packages', and type: `cmake dlib`.
2. To install the extension itself, click 'Manager', click 'Install custom nodes', and search for 'face analysis'. You should see this repository in the results. Click 'Install'.
3. After installing the extension you should now have folder `ComfyUI\custom_nodes\ComfyUI_FaceAnalysis\dlib`. Download [Shape Predictor](https://huggingface.co/matt3ounstable/dlib_predictor_recognition/resolve/main/shape_predictor_68_face_landmarks.dat?download=true) and the [Face Recognition](https://huggingface.co/matt3ounstable/dlib_predictor_recognition/resolve/main/dlib_face_recognition_resnet_model_v1.dat?download=true) models and place them into the `dlib` directory.
4. Load the 'face_analysis.json' workflow file. It uses IPAdapter to generate a few images and return the distance to the reference face.

![face analysis](./face_analysis.jpg)

## Important notes

There are many ways to do this. At the moment I'm using DLib as it's fast and easy to use, if there's an actual interest I will release more options (insightface?).

Also, I'm not an engineer and I don't know what I'm doing, hopefully someone more experienced can chime in.
