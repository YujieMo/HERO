# Self-supervised Heterogeneous Graph Learning:  a Homogeneity and Heterogeneity Perspective 

This repository contains the reference code for the manuscript ``Self-supervised Heterogeneous Graph Learning:  a Homogeneity and Heterogeneity Perspective" 

## Contents

0. [Installation](#installation)
0. [Preparation](#Preparation)
0. [Training](#train)


## Installation
* pip install -r requirements.txt 
* Unzip the datasets (the heterogeneous graph datasets can be found in ../dataset/, and the homogeneous graph datasets will be automatically downloaded from the public website)

## Preparation
Important args:
* `--use_pretrain` Test checkpoints to reproduce the results 
* `--dataset` Heterogeneous graph: ACM, Yelp, DBLP, Aminer || Homogeneous graph: photo, computers, cs, physics
* `--custom_key` Node: node classification

## Training
python main.py


