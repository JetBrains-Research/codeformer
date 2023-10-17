# Codeformer

[WIP] A library for running Codeformer model.

## Supported tasks

- [x] Method name prediction
- [x] Code modeling
- [x] Language modeling

## Installation
0. Go to the root of the project:
```bash
cd codeformer
```
1. Install the dependencies from requirements.txt:
```bash
pip -r requirements.txt
```
2. Install spacy and download additional packages:
```bash
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```
3. Update git submodules to fetch tree-sitter:
```bash
git submodule update --init --recursive
```

## How to run

Run the run_task.py script from jetnn/runners/ folder using similar command: 
```bash
python3 run_task.py -task=language_modeling -mode=train -c ../../configs/config_language_modeling.yaml -cd 0
```
