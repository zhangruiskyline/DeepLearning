<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [SORT](#sort)
    - [Introduction](#introduction)
    - [License](#license)
    - [Citing SORT](#citing-sort)
    - [Usage:](#usage)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

SORT
=====

A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences.
See an example [video here](https://motchallenge.net/movies/ETH-Linthescher-SORT.mp4).

By Alex Bewley  
[DynamicDetection.com](http://www.dynamicdetection.com)

*Note: This fork has been modified for educational purposes*

### Introduction

SORT is a barebones implementation of a visual multiple object tracking framework based on rudimentary data association and state estimation techniques. It is designed for online tracking applications where only past and current frames are available and the method produces object identities on the fly. While this minimalistic tracker doesn't handle occlusion or re-entering objects its purpose is to serve as a baseline and testbed for the development of future trackers.

SORT was initially described in an [arXiv tech report](http://arxiv.org/abs/1602.00763). At the time of the initial publication, SORT was ranked the best *open source* multiple object tracker on the [MOT benchmark](https://motchallenge.net/results/2D_MOT_2015/).

This code has been tested on Mac OSX 10.10, and Ubuntu 14.04, with Python 2.7 (anaconda).

**Note:** A significant proportion of SORT's accuracy is attributed to the detections.
For your convenience, this repo also contains *Faster* RCNN detections for the MOT benchmark sequences in the benchmark format. To run the detector yourself please see the original [*Faster* RCNN project](https://github.com/ShaoqingRen/faster_rcnn) or the python reimplementation of [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) by Ross Girshick.

### License

SORT is released under the GPL License (refer to the LICENSE file for details) to promote the open use of the tracker and future improvements. If you require a permissive license contact Alex (alex@dynamicdetection.com).

### Citing SORT

If you find this repo useful in your research, please consider citing:

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }


### Usage:

To run the tracker with the provided detections:

- Download the [2D MOT 2015 benchmark dataset](https://motchallenge.net/data/2D_MOT_2015/#download)
	```
	wget https://motchallenge.net/data/2DMOT2015.zip
	unzip 2DMOT2015.zip
	```