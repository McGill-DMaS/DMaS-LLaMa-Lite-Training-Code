import torch
import torch.nn as nn


class KVCache:
    def __init__(self, n_layer):
        self.n_layer = n_layer
        self.key_list = [[] for _ in range(n_layer)]
        self.value_list = [[] for _ in range(n_layer)]

    def reset(self):
        """Clear the cache."""
        self.key_list.clear()
        self.value_list.clear()

    def update(self ,layer, keys, values):
        """Append new keys and values to the cache."""
        self.key_list[layer] = keys
        self.value_list[layer] = values

    def get_key_values(self,layer):
        """Retrieve all cached keys."""
        return self.key_list[layer], self.value_list[layer]

