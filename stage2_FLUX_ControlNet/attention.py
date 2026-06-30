import inspect
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
import cv2
import os
from diffusers.models.attention_processor import Attention
import numpy as np


target_combinations = {17: [7], 52: [4], 23: [14], 50: [22], 53: [4], 13: [20], 14: [7], 46: [1]}

def create_neighborhood_mask(N, H=64, W=64, kernel_size=3):
    assert N == H * W
    identity = torch.eye(N)
    identity = identity.reshape(N, 1, H, W)
    padding = (kernel_size - 1) // 2
    # F.max_pool2d may be more intuitive, but avg_pool2d has the same effect here.
    pooled = F.avg_pool2d(identity, kernel_size=kernel_size, stride=1, padding=padding)
    mask = pooled.reshape(N, N)
    binary_mask = (mask > 0).float()
    return binary_mask

def my_logsumexp_with_hard_clamp(
    input, 
    dim, 
    keepdim=False, 
    safe_min=-1e6, 
    safe_max=1e6
):
    """
    Numerically stable logsumexp implementation.

    Before computation, all input values, including inf and NaN,
    are clamped to the [safe_min, safe_max] range.
    """
    
    # --- Step 1: Preprocess and clamp. ---
    
    # 1.1. Handle NaN.
    # torch.nan_to_num(..., nan=0.0) only replaces NaN and does not touch inf.
    # First convert NaN to a safe finite value, 0.0.
    no_nan_input = torch.nan_to_num(input, nan=0.0)
    
    # 1.2. Apply hard clamping.
    # torch.clamp maps:
    #   - +inf   ->  safe_max
    #   - -inf   ->  safe_min
    #   - values outside the range -> safe_max or safe_min
    #   - values inside the range  -> unchanged
    clamped_input = torch.clamp(no_nan_input, min=safe_min, max=safe_max)
    
    # --- Step 2: Core LogSumExp logic. ---
    
    # 2.1. Find the maximum value m.
    # Since inputs are clamped, max_val is always finite, at most safe_max.
    max_val, _ = torch.max(clamped_input, dim=dim, keepdim=True)
    
    # 2.2. Compute x_i - m.
    # This step is now safe and does not produce NaN.
    # The worst case is safe_min - safe_max, which is a finite negative value.
    stable_input = clamped_input - max_val
    
    # 2.3. Compute exp(x_i - m).
    exp_stable_input = torch.exp(stable_input)
    
    # 2.4. Compute sum(...).
    sum_exp = torch.sum(exp_stable_input, dim=dim, keepdim=True)
    
    # 2.5. Compute log(...).
    tiny_eps = 1e-6
    log_sum = torch.log(sum_exp+ tiny_eps)
    
    # 2.6. Add m.
    result = max_val + log_sum
    
    # 2.7. Handle dimensions.
    if not keepdim:
        result = result.squeeze(dim)
        
    return result    

def scaled_dot_product_attention(query, key, value, attn_mask=None,  
                                 dropout_p=0.0,is_causal=False,return_mask=False,c_choose=None, 
                                 scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    B = query.size(0)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(B,24,L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        # if attn_mask.dtype == torch.bool:
        #     attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        # else:
        #     attn_bias = attn_mask + attn_bias
        epsilon = 1e-6
        attn_bias = torch.log(attn_mask + epsilon)

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    if return_mask:
        attn_weight_soft = torch.softmax(attn_weight, dim=-1)[:,c_choose,512:,:512]
    else:
        attn_weight_soft =None

    N_txt =512
    # 4.1. Split logits: L_img_txt (image query, text key).
    # Assume L = S = N_txt + N_img.
    # L_img_txt shape: (B, H, N_img, N_txt).
    L_img_txt = attn_weight[:, :, N_txt:, :N_txt]
    
    # 4.2. Split bias: bias_img_txt, which is the mask.
    # Shape: (B, H, N_img, N_txt).
    bias_img_txt = attn_bias[:, :, N_txt:, :N_txt]
    
    # 4.3. Compute the original energy of the Img-Txt block (logsumexp).
    # Shape: (B, H, N_img, 1).
    lse_txt_orig = my_logsumexp_with_hard_clamp(L_img_txt, dim=-1, keepdim=True)

    # 4.4. Apply the mask to obtain masked logits.
    L_img_txt_masked = L_img_txt + bias_img_txt

    # 4.5. Compute the masked energy.
    # Shape: (B, H, N_img, 1).
    lse_txt_masked = my_logsumexp_with_hard_clamp(L_img_txt_masked, dim=-1, keepdim=True)

    # 4.6. Compute compensation C = masked energy - original energy.
    # Shape: (B, H, N_img, 1).
    C = lse_txt_masked - lse_txt_orig

    # 4.7. Handle NaN when both lse_txt_orig and lse_txt_masked are -inf.
    # In this case C should be 0 because both original and masked energies are 0 in log space.
    C = torch.nan_to_num(C, nan=0.0)
    
    # 4.8. Add compensation C to the Img-Img block of attn_bias.
    # C with shape (B, H, N_img, 1) is broadcast to (B, H, N_img, N_img).
    attn_bias[:, :, N_txt:, N_txt:] += C
    
    attn_weight += attn_bias

    attn_weight = torch.softmax(attn_weight, dim=-1)

    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    return attn_weight @ value,attn_weight_soft
    


class FluxAttnProcessor2_0(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        return_mask=None,
        layer =None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        
        c_choose = None
        if layer not in target_combinations:
                return_mask =False
        if return_mask:
            c_choose = target_combinations[layer]

        hidden_states,cross_attn = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=0.0, is_causal=False,return_mask=return_mask,
            c_choose = c_choose
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states,cross_attn
        else:
            return hidden_states,cross_attn
