from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig,LlamaForCausalLM
from models.gpt2 import GPT
from models.llama import LlamaTransformer
from utils import load_model
import torch


def convert_to_hf(model_name,ckp_path):
    # Load the model from the checkpoint
    checkpoint = torch.load(ckp_path, map_location=torch.device('cpu'),weights_only=False)
    config = checkpoint['config']
    if model_name == 'gpt2':

        state_dict = checkpoint['model']
        model = GPT(config)
        model.load_state_dict(state_dict)
        config = GPT2Config(
            vocab_size=50304,
            n_positions=1024,
            n_ctx=config.block_size,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
        )
        model_hf = GPT2LMHeadModel(config=config)
    else:
        model = LlamaTransformer(config)
        model = load_model(model, ckp_path)
        config = LlamaConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.dim,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.n_layers,
            num_attention_heads=config.n_heads,
            num_key_value_heads=config.n_kv_heads,
            max_position_embeddings=config.max_seq_len,
            rms_norm_eps=config.norm_eps,
            rope_theta=config.rope_theta)

        model_hf = LlamaForCausalLM(config=config)

    sd = model.state_dict()
    sd_keys = sd.keys()
    if model_name == 'gpt2':
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
    else:
        sd_keys = [k for k in sd_keys if not k.endswith('.bias')]

    # Initialize the model
    #model_size = sum(t.numel() for t in model_hf.parameters())

    sd_hf = model_hf.state_dict()

    # copy while ensuring all of the parameters are aligned and match in names and shapes
    sd_keys_hf = sd_hf.keys()
    if model_name == 'gpt2':
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    # this means that we have to transpose these weights when we import them
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
        if model_name == 'gpt2' and any(k.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd_hf[k].copy_(sd[k].t())
        else:
            # vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k].shape, f"mismatched shape of {k}: {sd_hf[k].shape} != {sd[k].shape}"
            with torch.no_grad():
                sd_hf[k].copy_(sd[k])

    return model_hf

