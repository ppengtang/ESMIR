# Learning Super Shared Middle-level Image Representation for Scene Classification

## Introduction

The implemention codes on MIT-indoor 67 dataset.

A novel method to learn very compact and discriminative image representation.

The paper is submitted to the journal "IEEE TRANSACTIONS ON CYBERNETICS".


## Dependencies
The MIT-indoor 67 dataset can be downloaded on http://web.mit.edu/torralba/www/indoor.html and should be put under the "data" path.

The caffe (http://caffe.berkeleyvision.org/) is needed to be installed.

The CNN model can be downloaded on https://gist.github.com/ksimonyan/fd8800eeb36e276cd6f9#file-readme-md 
and be put under the "feature_extraction_imagenet" path.

VLFeat (http://www.vlfeat.org/) and LIBLINEAR (https://www.csie.ntu.edu.tw/~cjlin/liblinear/) are also needed.


## Trained models
We put 4 different trained models "best_2.mat", "best_4.mat", "best_6.mat", "best_8.mat" under the "data" path, 
which correspond to 2 patterns, 4 patterns, 6 patterns, 8 patterns per class respectively.


## Testing
Run the "MITindoor_classification.m" to test the trained model.


## Training
Run the "exp_MITindoorft.m" to train the model.