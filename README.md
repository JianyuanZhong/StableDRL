# StableDRL

This repository contains the implementation of StableDRL for diffusion language models (specifically LLaDA).

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Training

To train the model using SVPO on Sudoku or other tasks, you can use the provided `train.py` script with the corresponding configuration file.

Example for Sudoku:

```bash
accelerate launch --num_processes 8 --config_file configs/accelerate/fsdp.yaml train.py --config configs/sudoku_spg_snis.yaml
```

You can find other configurations in the `configs/` directory.

### Inference

To generate samples using the trained model, you can use the `generate_spg` function from `networks.llada_svpo`.

```python
import torch
from transformers import AutoTokenizer
from networks.llada_svpo import LLaDASVPO, generate_spg

# Load model and tokenizer
model_path = "path/to/your/checkpoint" # or pretrained model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LLaDASVPO.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

# Prepare input
prompt_text = "Your prompt here"
prompt = tokenizer(prompt_text, return_tensors="pt").input_ids.cuda()

# Generate
output = generate_spg(
    model,
    prompt,
    steps=64,
    gen_length=64,
    temperature=1.0
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```
