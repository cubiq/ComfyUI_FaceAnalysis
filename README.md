# Face Analysis for ComfyUI

This extension uses [DLib](http://dlib.net/) or [InsightFace](https://github.com/deepinsight/insightface) to calculate the Euclidean and Cosine *distance* between two faces.

The best way to evaluate generated faces is to first send a batch of 3 reference images to the node and compare them to a forth reference (all actual pictures of the person). That will give you a baseline number that you can use to compare to generated images.

## Installation

You need to install either InsightFace or Dlib (or both).

For DLIB download [Shape Predictor](https://huggingface.co/matt3ounstable/dlib_predictor_recognition/resolve/main/shape_predictor_68_face_landmarks.dat?download=true), [Face Predictor 5 landmarks](https://huggingface.co/matt3ounstable/dlib_predictor_recognition/resolve/main/shape_predictor_5_face_landmarks.dat?download=true), [Face Predictor 81 landmarks](https://huggingface.co/matt3ounstable/dlib_predictor_recognition/resolve/main/shape_predictor_81_face_landmarks.dat?download=true) and the [Face Recognition](https://huggingface.co/matt3ounstable/dlib_predictor_recognition/resolve/main/dlib_face_recognition_resnet_model_v1.dat?download=true) models and place them into the `dlib` directory.

Precompiled Dlib for Windows can be found [here](https://github.com/z-mahmud22/Dlib_Windows_Python3.x).

![face analysis](./face_analysis.jpg)
