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

def atten_adj(attn_weight, attn_bias):
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
    lse_txt_orig = torch.logsumexp(L_img_txt, dim=-1, keepdim=True)

    # 4.4. Apply the mask to obtain masked logits.
    L_img_txt_masked = L_img_txt + bias_img_txt

    # 4.5. Compute the masked energy.
    # Shape: (B, H, N_img, 1).
    lse_txt_masked = torch.logsumexp(L_img_txt_masked, dim=-1, keepdim=True)

    # 4.6. Compute compensation C = masked energy - original energy.
    # Shape: (B, H, N_img, 1).
    C = lse_txt_masked - lse_txt_orig

    # 4.7. Handle NaN when both lse_txt_orig and lse_txt_masked are -inf.
    # In this case C should be 0 because both original and masked energies are 0 in log space.
    C = torch.nan_to_num(C, nan=0.0)
    
    # 4.8. Add compensation C to the Img-Img block of attn_bias.
    # C with shape (B, H, N_img, 1) is broadcast to (B, H, N_img, N_img).
    attn_bias[:, :, N_txt:, N_txt:] += C
    return attn_bias

def atten_adj_1(attn_weight,attn_bias):
    
    num_txt_tokens = 512
    bias_img_txt = attn_bias[:, :, num_txt_tokens:, :num_txt_tokens]
    # 3.1. Separate scores for text queries and image queries.
    txt_query_scores = attn_weight[:, :, :num_txt_tokens, :]
    img_query_scores = attn_weight[:, :, num_txt_tokens:, :]

    # 3.2. Apply standard softmax to the text-query part.
    final_weights_txt = F.softmax(txt_query_scores, dim=-1)

    # --- Two-step normalization for image queries. ---
    
    # 3.3. Further split image-query scores into img->txt and img->img.
    img_to_txt_scores = img_query_scores[:, :, :, :num_txt_tokens]
    img_to_img_scores = img_query_scores[:, :, :, num_txt_tokens:]

    # 3.4. Compute the original weight budget before applying position_mask.
    # Use logsumexp for a numerically stable sum.
    logsumexp_it = torch.logsumexp(img_to_txt_scores, dim=-1, keepdim=True)
    logsumexp_ii = torch.logsumexp(img_to_img_scores, dim=-1, keepdim=True)

    # Compute softmax between the two parts to get their total weight proportions.
    total_logsumexp = torch.cat([logsumexp_it, logsumexp_ii], dim=-1)
    partition_weights = F.softmax(total_logsumexp, dim=-1)
    
    original_it_weight_sum = partition_weights[..., 0:1] # shape: (B, H, num_img, 1)
    original_ii_weight_sum = partition_weights[..., 1:2] # shape: (B,

    weights_it_internal = F.softmax(img_to_txt_scores+bias_img_txt, dim=-1)
    weights_ii_internal = F.softmax(img_to_img_scores, dim=-1)

    # 3.7. Scale internal weights with the original budget.
    final_weights_it = weights_it_internal * original_it_weight_sum
    final_weights_ii = weights_ii_internal * original_ii_weight_sum

    # 3.8. Concatenate final weights for image queries.
    final_weights_img = torch.cat([final_weights_it, final_weights_ii], dim=-1)
    
    # 3.9. Reassemble the full attention weight matrix.
    attn_weight_final = torch.cat([final_weights_txt, final_weights_img], dim=-2)
    return attn_weight_final

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

    

def scaled_dot_product_attention(query, key, value, attn_mask=None, 
                                 indices_list =None,position_mask=None,
                                 timestep=None,num =None,part_prompt=None,layer =0, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
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
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor


    attn_weight_1 = atten_adj_1(attn_weight,attn_bias)

    attn_bias = atten_adj(attn_weight,attn_bias)
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)


    attn_weight = torch.dropout(attn_weight_1, dropout_p, train=True)
    return attn_weight @ value


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
        indices_list : Optional[torch.Tensor] = None,
        position_mask: Optional[torch.Tensor] = None,
        part_prompt : Optional[str] = None,
        timestep=None,
        layer = None,
        num =None
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

        
        
        hidden_states = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            indices_list=indices_list,
            position_mask= position_mask,
            timestep= timestep,
            num = num,
            part_prompt= part_prompt,
            layer = layer,
            dropout_p=0.0, is_causal=False
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

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
