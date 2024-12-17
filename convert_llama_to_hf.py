import glob
import os
from convert_to_hf import convert_to_hf
from transformers import AutoTokenizer
import torch

# specify the directory where the checkpoint files are stored
ckpt_dir = './llama_2b/'

# find all checkpoint files in the directory
ckpt_files = glob.glob(os.path.join(ckpt_dir, 'model_*.pt'))

# extract the numbers from the file names and find the largest
latest_ckpt_num = max(int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]) for f in ckpt_files)

# construct the path to the latest checkpoint file
# use zfill to add leading zeros to the checkpoint number
ckp_path = os.path.join(ckpt_dir, f'model_{str(latest_ckpt_num).zfill(5)}.pt')

model_name = 'llama'
hf_model = convert_to_hf(model_name, ckp_path)


# Target vocabulary size
new_vocab_size = 50257

# Update the embedding layer
old_embed_tokens = hf_model.model.embed_tokens.weight.data
hf_model.model.embed_tokens = torch.nn.Embedding(new_vocab_size, old_embed_tokens.size(1))
hf_model.model.embed_tokens.weight.data = old_embed_tokens[:new_vocab_size, :]

# Update the lm_head
old_lm_head = hf_model.lm_head.weight.data
hf_model.lm_head = torch.nn.Linear(old_lm_head.size(1), new_vocab_size, bias=False)
hf_model.lm_head.weight.data = old_lm_head[:new_vocab_size, :]

hf_model.config.vocab_size = new_vocab_size


# construct the save folder name based on the model name and checkpoint number
sav_fold = f'/src/dmas-{model_name}-{latest_ckpt_num}_steps'

# create the save folder if it doesn't exist
os.makedirs(sav_fold, exist_ok=True)

hf_model.save_pretrained(sav_fold)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.save_pretrained(sav_fold)