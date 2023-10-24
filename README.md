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


## Natural Language Modelling

To evaluate the NLM model use the the `generate(input_ids: LongTensor, max_new_tokens: int)` can be used. 
Example of usage:
```python
from lm.model import CodeformerLM; from transformers import AutoTokenizer

max_new_tokens = 10
model_name = 'microsoft/deberta-v3-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = CodeformerLM(model_name)
# Only single batch is supported by now
token_ids = tokenizer.encode('The man', return_tensors='pt', add_special_tokens=False)
pred_ids = model.generate(token_ids, max_new_tokens)
# If you want to get only predicted text without prefix 
# uncomment the following line
# pred_ids = pred_ids[:, token_ids.shape[1]:]
pred_text = tokenizer.decode(pred_ids[0])
```

Currently only for interfaces purposes, the output is random:
```python
>>> print(pred_text)
The man Sandal Hickenlooper hauntedFIS despicable Asp
```
