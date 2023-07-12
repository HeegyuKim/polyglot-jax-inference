from __future__ import annotations

from typing import Any, NamedTuple

import chex
import flax.traverse_util
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec

from modeling_hf import GPTNeoXForCausalLM


class ConversionRule(NamedTuple):
    name: str
    transpose: bool = False
    slicing: tuple[slice, ...] | None = None
    unflatten: dict[str, Any] | None = None
    unflatten_first: bool = False
    dtype: jnp.dtype = jnp.bfloat16


def convert_weights(
    state_dict: dict[str, torch.Tensor], rules: dict[str, ConversionRule]
) -> chex.ArrayTree:
    params = {}
    for name, rule in rules.items():
        param = state_dict[rule.name]
        if rule.transpose:
            param = param.transpose(0, 1)
        if rule.unflatten_first:
            if rule.unflatten is not None:
                param = param.unflatten(**rule.unflatten)
            if rule.slicing is not None:
                param = param[rule.slicing]
        else:
            if rule.slicing is not None:
                param = param[rule.slicing]
            if rule.unflatten is not None:
                param = param.unflatten(**rule.unflatten)
        params[name] = param.numpy().astype(jnp.bfloat16)
    return flax.traverse_util.unflatten_dict(params, sep=".")


def get_conversion_rules(model: GPTNeoXForCausalLM) -> dict[str, ConversionRule]:
    head_dim = model.dim // model.heads
    WEIGHT_CONVERSION_RULES = {
        "gpt_neox.embed_in.embedding": ConversionRule("gpt_neox.embed_in.weight"),
        "gpt_neox.layers.{}.attention.query_key_value.kernel": ConversionRule(
            "gpt_neox.layers.{}.attention.query_key_value.weight",
            transpose=True,
        ),
        "gpt_neox.layers.{}.attention.dense.kernel": ConversionRule(
            "gpt_neox.layers.{}.attention.dense.weight",
            transpose=True,
        ),
        "gpt_neox.layers.{}.attention.query_key_value.bias": ConversionRule(
            "gpt_neox.layers.{}.attention.query_key_value.bias",
        ),
        "gpt_neox.layers.{}.attention.dense.bias": ConversionRule(
            "gpt_neox.layers.{}.attention.dense.bias"
        ),

        "gpt_neox.layers.{}.mlp.dense_h_to_4h.kernel": ConversionRule(
            "gpt_neox.layers.{}.mlp.dense_h_to_4h.weight", transpose=True
        ),
        "gpt_neox.layers.{}.mlp.dense_4h_to_h.kernel": ConversionRule(
            "gpt_neox.layers.{}.mlp.dense_4h_to_h.weight", transpose=True
        ),
        "gpt_neox.layers.{}.mlp.dense_h_to_4h.bias": ConversionRule(
            "gpt_neox.layers.{}.mlp.dense_h_to_4h.bias"
        ),
        "gpt_neox.layers.{}.mlp.dense_4h_to_h.bias": ConversionRule(
            "gpt_neox.layers.{}.mlp.dense_4h_to_h.bias"
        ),

        "gpt_neox.layers.{}.input_layernorm.scale": ConversionRule(
            "gpt_neox.layers.{}.input_layernorm.weight"
        ),
        "gpt_neox.layers.{}.input_layernorm.bias": ConversionRule(
            "gpt_neox.layers.{}.input_layernorm.bias"
        ),

        "gpt_neox.layers.{}.post_attention_layernorm.scale": ConversionRule(
            "gpt_neox.layers.{}.post_attention_layernorm.weight"
        ),
        "gpt_neox.layers.{}.post_attention_layernorm.bias": ConversionRule(
            "gpt_neox.layers.{}.post_attention_layernorm.bias"
        ),

        "embed_out.kernel": ConversionRule("embed_out.weight", transpose=True),
        "gpt_neox.final_layer_norm.scale": ConversionRule("gpt_neox.final_layer_norm.weight"),
        "gpt_neox.final_layer_norm.bias": ConversionRule("gpt_neox.final_layer_norm.bias"),
    }

    conversion_rules = {}
    for k, v in WEIGHT_CONVERSION_RULES.items():
        for i in range(model.layers):
            conversion_rules[k.format(i)] = ConversionRule(v[0].format(i), *v[1:])
    return conversion_rules


def get_sharding_rules(model: GPTNeoXForCausalLM) -> chex.ArrayTree:
    WEIGHT_SHARDING_RULES = {
        "gpt_neox.embed_in.embedding": ("mp", None),
        "gpt_neox.layers.{}.attention.query_key_value.kernel": (None, "mp"),
        "gpt_neox.layers.{}.attention.dense.kernel": ("mp", None),
        "gpt_neox.layers.{}.attention.query_key_value.bias": ("mp",),
        "gpt_neox.layers.{}.attention.dense.bias": (None,),
        "gpt_neox.layers.{}.mlp.dense_h_to_4h.kernel": (None, "mp"),
        "gpt_neox.layers.{}.mlp.dense_4h_to_h.kernel": ("mp", None),
        "gpt_neox.layers.{}.mlp.dense_h_to_4h.bias": ("mp",),
        "gpt_neox.layers.{}.mlp.dense_4h_to_h.bias": (None,),
        "gpt_neox.layers.{}.input_layernorm.scale": (None,),
        "gpt_neox.layers.{}.input_layernorm.bias": (None,),
        "gpt_neox.layers.{}.post_attention_layernorm.scale": (None,),
        "gpt_neox.layers.{}.post_attention_layernorm.bias": (None,),
        "embed_out.kernel": (None, "mp"),
        "gpt_neox.final_layer_norm.scale": (None,),
        "gpt_neox.final_layer_norm.bias": (None,),
    }

    sharding_rules = {}
    for k, v in WEIGHT_SHARDING_RULES.items():
        for i in range(model.layers):
            sharding_rules[k.format(i)] = PartitionSpec(*v)
    return flax.traverse_util.unflatten_dict(sharding_rules, sep=".")
