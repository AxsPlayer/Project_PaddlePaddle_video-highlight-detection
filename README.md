# Project_PaddlePaddle_video-highlight-detection
Using DL to automatically detect the highlight part of video.

## Motivation?
In a lot of variety shows, editor should find the highlight part in videos by themselves. It's a heavy workload for editors. Therefore, if there are some AIs, which can allevate pain of editing, it's valuable and desirable. 

## Aims.
The aim of this project is to find the highlight part in videos, automatically.

## Data used.
The original dataset are vectors extracted from pictures using ResNet, as well as vectors extracted from audios.

## The structure of Neural Network.
The structure of Neural Network is: 
Firstly, predict every second's probabilty as highlight time. The method is to using 1D-CNN to train and predict. 
Secondly, using TAG(Temporal Actioness Grouping) to create proposals and select final result.

## About the tool.
The tool used in this project is PaddlePaddle, created by Baidu, China.
