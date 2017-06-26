<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Object Detection & Tracking Exercises](#object-detection--tracking-exercises)
  - [Setup](#setup)
    - [Testing](#testing)
    - [Training](#training)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Object Detection & Tracking Exercises

Note: This code is a fork of the following repository:

https://github.com/rykov8/ssd_keras

It has been adapted for educational purposes. You can find below several IPython notebook tutorials on object detection and tracking.

## Setup

### Testing

- Download model weights file ```weights_SSD300.hdf5``` [here](https://mega.nz/#F!7RowVLCL!q3cEVRK9jyOSB9el3SssIA).
- Put it under ```data/files/``` or create a symbolic link there.

### Training

- Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar;
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar;
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

- Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar;
	tar xvf VOCtest_06-Nov-2007.tar;
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

- It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```
	
- Put this folder under the ```data``` folder or create a symbolic link there.