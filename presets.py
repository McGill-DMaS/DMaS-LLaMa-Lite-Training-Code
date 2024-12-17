from models.gpt2 import GPTConfig
from models.llama import LlamaConfig





model_presets = {
    'gpt2': {'10m': GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=50304, block_size=1024),
             '124m': GPTConfig(n_layer=12, n_head=12, n_embd=768, vocab_size=50304, block_size=1024),
             '500m': GPTConfig(n_layer=36, n_head=16, n_embd=1024, vocab_size=50304, block_size=1024),
             'large': GPTConfig(n_layer=36, n_head=16, n_embd=1280, vocab_size=50304, block_size=1024),
             'xl': GPTConfig(n_layer=48, n_head=16, n_embd=1600, vocab_size=50304, block_size=1024)},

    'llama': {'124m': LlamaConfig(n_layers=12, n_heads=12, dim=768, vocab_size=50304, max_seq_len=1024, intermediate_size=4 * 768),
              '978m': LlamaConfig(n_layers=36, n_heads=20, dim=1280, vocab_size=50304, max_seq_len=1024, intermediate_size=5120,n_kv_heads=4),
              '2b': LlamaConfig(n_layers=36, n_heads=32, dim=2048, vocab_size=50304, max_seq_len=1024, intermediate_size=5120,n_kv_heads=8)}

}