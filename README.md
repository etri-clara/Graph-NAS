# (Work in process)Boosting semi-supervised preformance prediction via Graph of Architecture

This repository is pytorch implementation for our ongoing research "Boosting semi-supervised preformance prediction via Graph of Architecture".

## Summary
We are developing neural performance predictor for NAS. There are lots of model-based performance predictor, but they need enough amount of architecture-performance pair. However, this scenario is intractable under the limited budget. Therefore, we propose Neural architecture performance predictor on semi-supervised setting, which is boosted by constructing Graph of architecture.

## Prerequisites
1. Download our pre-defined data split from [link](https://drive.google.com/file/d/1e6C5CETMuICeOYMII5c2KVI71kr3N_8S/view?usp=sharing)
2. Extract the zip file into ```./data/```
3. Build anaconda env via ```conda env create -f conda_env.yaml``` 
4. Activate conda virtual env with the command ```conda activate gnn```.

## Quick Start
1. Edit config(```setting201.json```) file with your desired setting.
2. Run as ```python main.py setting.json 50```