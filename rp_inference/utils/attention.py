# modeling_XX.py from transformers==4.48.3
# for replacing torch.matmul to self.matmul
import math
import warnings
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg

@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def llama_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if self.config._attn_implementation != "eager":
        raise NotImplementedError

    # eager_attention_forward
    if self.training:
        dropout = self.attention_dropout
    else:
        dropout=0.0
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    #attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    attn_weights = self.matmul1(query_states, key_states.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=self.training)
    #attn_output = torch.matmul(attn_weights, value_states)
    attn_output = self.matmul2(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    # end eager_attention_forward

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
