# (Work in process)Boosting semi-supervised preformance prediction via Graph of Architecture

This repository is pytorch implementation for our ongoing research "Boosting semi-supervised preformance prediction via Graph of Architecture".

## Summary
We are developing neural performance predictor for NAS. There are lots of model-based performance predictor, but they need enough amount of architecture-performance pair. However, this scenario is intractable under the limited budget. Therefore, we propose Neural architecture performance predictor on semi-supervised setting, which is boosted by constructing Graph of architecture.

## Prerequisites
1. Download our pre-defined data split from [link](https://drive.google.com/file/d/1t8wrmdq_pnHtJKO4s7CGm1-z6vM0cdZI/view?usp=share_link)
2. Extract the zip file into ```./data/```
3. Build anaconda env via ```conda env create -f conda_env.yaml``` 
4. Activate conda virtual env with the command ```conda activate gnn```.

## Quick Start
1. Edit config(```setting.json```) file with your desired setting.
2. Run as ```python main.py setting.json 50```