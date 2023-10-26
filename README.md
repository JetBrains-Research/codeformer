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

import torch
from lm.utils import get_model_from_config, get_tokenizer_from_config
from omegaconf import OmegaConf

# device = torch.device('cuda:3')  # third GPU
device = torch.device('cpu')
config = OmegaConf.load('/mnt/data/shared-data/codeformer/models/rand_init_codeformer.yaml')
print(config.load_path)
# >>> /mnt/data/shared-data/codeformer/models/rand_init_codeformer.pt
# If needed you can:
# config.load_path = '/your/path.pt'
model = get_model_from_config(config)
tokenizer = get_tokenizer_from_config(config)

max_new_tokens = 20
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
