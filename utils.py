import torch

import torch.nn.functional as F

from hellaswag import render_example, iterate_examples
from models.kvcache import KVCache
import requests

import time



def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm
def evaluate_hella_swag(model, device, device_type,model_imp, dist=None,ddp=False,master_process=False,ddp_rank=0,ddp_world_size=1,log_file=None,step=0):
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                if model_imp == 'hf':
                    logits, _ = model(tokens, return_dict=False)
                else:
                    logits = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    # reduce the stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} hella {acc_norm:.4f}\n")
    return num_correct_norm,num_total,acc_norm


def extract_json(content):
    # Initialize counters for curly braces
    open_brace_count = 0
    close_brace_count = 0
    json_start_index = None
    json_end_index = None

    # Iterate over the content by index and character
    for index, char in enumerate(content):
        if char == '{':
            open_brace_count += 1
            # Mark the start of JSON content
            if json_start_index is None:
                json_start_index = index
        elif char == '}':
            close_brace_count += 1
            # If the counts match, we've found the end of the JSON content
            if open_brace_count == close_brace_count:
                json_end_index = index + 1  # Include the closing brace
                break

    # If we found a start and end, extract and parse the JSON
    if json_start_index is not None and json_end_index is not None:
        json_str = content[json_start_index:json_end_index]
        try:
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON content") from e
    else:
        raise ValueError("No JSON content found")


def completion(model, enc, prompt, device, device_type,model_imp, generate_max_length,num_return_sequences,rank=0, greedy=False, use_cache=False):
    model.eval()
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    cpu_device = torch.device("cpu")
    if greedy:
        num_return_sequences = 1
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    if use_cache:
        cache = KVCache(model.config.n_layers)
    else:
        cache = None
    result = tokens
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + rank)
    t1 = time.time()
    if greedy:
        # greedy decoding
        while result.size(1) < generate_max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    if model_imp == 'hf':
                        logits, _ = model(xgen, return_dict=False)
                    else:
                        logits = model(xgen,use_cache=use_cache,cache=cache)
                # take the logits at the last position
                logits = logits[:, -1, :]
                # take the argmax
                ix = torch.argmax(logits, dim=-1, keepdim=True)
                xcol = ix
                if use_cache:
                    xgen = xcol
                    result = torch.cat((result, xgen.to(cpu_device)), dim=1)
                else:
                    xgen = torch.cat((xgen, xcol), dim=1)
                    result = xgen
        ret = []
        tokens = result[0, :generate_max_length].tolist()
        decoded = enc.decode(tokens)
        #print(f"rank {rank} sample 0: {decoded}")
        ret.append(decoded)
        t2 = time.time()
        print(f"generating {num_return_sequences} greedy completions of length {generate_max_length} took {t2 - t1:.2f} seconds")
        return ret


    while result.size(1) < generate_max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                if model_imp == 'hf':
                    logits, _ = model(xgen, return_dict=False)
                else:
                    logits = model(xgen,use_cache=use_cache,cache=cache)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            if use_cache:
                xgen = xcol
                result = torch.cat((result, xgen.to(cpu_device)), dim=1)
            else:
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
                result = xgen
    # print the generated text
    results = []
    for i in range(num_return_sequences):
        tokens = result[i, :generate_max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"rank {rank} sample {i}: {decoded}")
        results.append(decoded)
    t2 = time.time()
    print(f"generating {num_return_sequences} completions of length {generate_max_length} took {t2 - t1:.2f} seconds")
    return results





def load_model(model, ckp_path,with_step = False):
    checkpoint = torch.load(ckp_path, map_location=torch.device('cpu'),weights_only=False)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    if with_step:
        return model, checkpoint['step']
    return model
