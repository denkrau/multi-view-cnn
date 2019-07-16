# multi-view-cnn
Classify an object based on multiple views of its 3D representation. These views are combined to a so-called multi-view and used as input for a convolutional neural network.
In particular, there are duplicates of each object with different color marks. The possible classification classes include each color mark of an object.

If, for example, there wil be classified two objects where each duplicate has either a green or red color mark, the possible prediction classes include objA_green and objA_red.

## Configuration
All parameters for training and evaluation are stored in params.py. They may need to be changed before proceeding.

## Training
Train a CNN with

    python train.py

## Evaluate 

Evaluate a trained network regarding its accuracy among others with

    python stats.py [-w]

    -w: write to disk

## Predict Single Objects

Show the view scores, grouping, saliency maps and activations of single predictions with

    python predict.py [-w] [-g] [-s] [-f] [-c] mulit-view1 [multi-view2] ...

    -w: write to disk (includes -g -s -f)
    -g: show grouping
    -s: show saliency maps
    -f: show activations
    -c: network's checkpoint file

## Generate a Dataset

Generate a dataset with

    python generate_views.py

using the Blender API. Hence, this needs to be called inside Blender's Python console in general.
The view generation is based on a polygon mesh representation of objects like the ones from ModelNet.

> **Note:** Some parameters inside this file may need to be changed to acquire a certain behaviour.