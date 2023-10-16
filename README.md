# JetNN

[WIP] A library for running Codeformer model.

## Supported tasks

- [x] Method name prediction
- [x] Code modelling
- [ ] Comment generation
- [ ] And more...

## How to run

Simply install the dependecies from requirements.txt and then run run_methodnaming.py script from jetnn/runners/ folder using simmilar command: 
```bash
python3 run_methodnaming.py train -c ../../configs/config_methodnaming.yaml -cd 0 -dr ../../datasets/python/python_small -et BigBird -dt plain_code -mcp 2048 -bs 16 -wk 5e2675f8a3c8340913e59614f17ee02a7b7f4351 -opt Momentum -lr 0.01 -wd 0.0001 -mss 14 -msn 384
```
