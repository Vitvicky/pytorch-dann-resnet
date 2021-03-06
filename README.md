# PyTorch-DANN

This repo is mostly based on https://github.com/wogong/pytorch-dann. 

A PyTorch implementation for paper *[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/)*

    InProceedings (icml2015-ganin15)
    Ganin, Y. & Lempitsky, V.
    Unsupervised Domain Adaptation by Backpropagation
    Proceedings of the 32nd International Conference on Machine Learning, 2015

## Environment

- Python 3.8.5
- PyTorch 1.6.0

## Note

Before running the training code, make sure that `DATASETDIR` environment variable is set to dataset directory.

- `MNISTmodel()`
    - basically the same network structure as proposed in the paper, expect for adding dropout layer in feature extractor
    - large gap exsits between with and w/o dropout layer
    - better result than paper
- `SVHNmodel()`
    - network structure proposed in the paper may be wrong for both 32x32 and 28x28 inputs
    - change last conv layer's filter to 4x4, get similar(actually higher) result
- `GTSRBmodel()`
- `AlexModel`
    - not successful, mainly due to the pretrained model difference
- `ResNet50`
    - Better and more stable results than AlexNet.

## Run

For training on Office dataset using ResNet-50, first set configs in `experiments/office.py`, then run
```
$ python experiments/office.py
```

## Result

|                      | MNIST-MNISTM   | SVHN-MNIST | SYNDIGITS-SVHN | SYNSIGNS-GTSRB |
| :------------------: | :------------: | :--------: |:-------------: |:-------------: |
| Source Only          |   0.5225       |  0.5490    | 0.8674         | 0.7900         |
| DANN (paper)         |   0.7666       |  0.7385    | 0.9109         | 0.8865         |
| This Repo Source Only|   -            |  -         | -              | 0.9100         |
| This Repo            |   0.8400       |  0.7339    | 0.8200         | -              |

|                      | AMAZON-WEBVCAM |  DSLR-WEBCAM | WEBCAM-DSLR |
| :------------------: | :------------: |:-----------: |:----------: |
| Source Only          |   0.6420       |  0.9610      | 0.9780      |
| DANN (paper)         |   0.7300       |  0.9640      | 0.9920      |
| This Repo (ResNet50) |   0.8151       |  -           | -           |

## Credit

- <https://github.com/wogong/pytorch-dann>
- <https://github.com/fungtion/DANN>
- <https://github.com/hehai131/DANN>
- <https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py>
- <https://github.com/corenel/torchsharp>
- <https://github.com/corenel/pytorch-starter-kit>
