from __future__ import annotations

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class GPTNeoXAttention(nn.Module):
    dim: int
    heads: int
    rotary: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.query_key_value = nn.DenseGeneral(self.dim * 3, dtype=self.dtype)
        self.dense = nn.DenseGeneral(self.dim, dtype=self.dtype)

        x = np.arange(0, self.rotary, 2) / self.rotary
        x = np.outer(np.arange(10000), 1.0 / 10000.0**x)
        self.freqs_cis = jnp.asarray(np.cos(x) + 1j * np.sin(x), dtype=jnp.complex64)

    def update_cache(self, name: str, x: chex.Array) -> chex.Array:
        if (cache := self.get_variable("cache", name)) is not None:
            x = jnp.roll(cache, -x.shape[1], axis=1).at[:, -x.shape[1] :].set(x)
        self.put_variable("cache", name, x)
        return x

    def apply_rotary_embedding(self, x: chex.Array) -> chex.Array:
        z = x.astype(jnp.float32).reshape(*x.shape[:-1], 2, -1)
        z = jax.lax.complex(z[..., 0, :], z[..., 1, :])

        z = z * self.freqs_cis[None, -x.shape[1] :, None, :]
        z = jnp.stack((jnp.real(z), jnp.imag(z)), axis=-1)
        return z.reshape(x.shape).astype(x.dtype)

    def __call__(self, x: chex.Array, attn_bias: chex.Array) -> chex.Array:
        qkv = self.query_key_value(x)
        qkv = qkv.reshape(x.shape[:2] + (self.heads, -1))
        head_size = qkv.shape[-1] // 3

        # [b, s, h, d]
        q = qkv[..., :head_size]
        k = qkv[..., head_size : 2 * head_size]
        v = qkv[..., 2 * head_size :]

        k = self.update_cache("k", k)
        v = self.update_cache("v", v)

        q_rot = self.apply_rotary_embedding(q[..., : self.rotary])
        k_rot = self.apply_rotary_embedding(k[..., : self.rotary])

        q = jnp.concatenate((q_rot, q[..., self.rotary :]), axis=3)
        k = jnp.concatenate((k_rot, k[..., self.rotary :]), axis=3)

        p = jnp.einsum("bqhd,bkhd->bhqk", q, k) / k.shape[3] ** 0.5
        # print(p.shape, attn_bias.shape)
        attn_w = nn.softmax(p + attn_bias, axis=3)
        jax.debug.print(str(attn_w) + "attn_w={w}", w=attn_w)
        x = jnp.einsum("bhqk,bkhd->bqhd", attn_w, v)
        x = x.reshape(x.shape[:2] + (-1,))
        return self.dense(x)


class GPTNeoXMLP(nn.Module):
    dim: int
    hidden: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense_h_to_4h = nn.Dense(self.hidden, dtype=self.dtype)
        self.dense_4h_to_h = nn.Dense(self.dim, dtype=self.dtype)

    def __call__(self, x: chex.Array) -> chex.Array:
        return self.dense_4h_to_h(nn.gelu(self.dense_h_to_4h(x), approximate=False))


class GPTNeoXBlock(nn.Module):
    dim: int
    heads: int
    hidden: int
    rotary: int
    eps: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = GPTNeoXAttention(self.dim, self.heads, self.rotary)
        self.mlp = GPTNeoXMLP(self.dim, self.hidden)

        self.input_layernorm = nn.LayerNorm(self.eps, dtype=self.dtype)
        self.post_attention_layernorm = nn.LayerNorm(self.eps, dtype=self.dtype)

    def __call__(self, x: chex.Array, attn_bias: chex.Array) -> chex.Array:
        return x + self.attention(self.input_layernorm(x), attn_bias) + self.mlp(self.post_attention_layernorm(x))

class GPTNeoXBlockCollection(nn.Module):
    num_layers: int
    dim: int
    heads: int
    hidden: int
    rotary: int
    eps: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        layer_args = (self.dim, self.heads, self.hidden, self.rotary, self.eps)
        self.blocks = [
            GPTNeoXBlock(*layer_args, name=str(i), dtype=self.dtype) for i in range(self.num_layers)
        ]

    def __call__(self, x: chex.Array, attn_bias: chex.Array | None = None) -> chex.Array:
        for layer in self.blocks:
            x = layer(x, attn_bias)
        return x


class GPTNeoXModel(nn.Module):
    vocab_size: int
    num_layers: int
    dim: int
    heads: int
    rotary: int
    hidden: int
    eps: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_in = nn.Embed(self.vocab_size, self.dim, dtype=self.dtype)
        self.layers = GPTNeoXBlockCollection(self.num_layers, self.dim, self.heads, self.hidden, self.rotary, self.eps)
        self.final_layer_norm = nn.LayerNorm(self.eps, dtype=self.dtype)

    def __call__(self, x: chex.Array, mask: chex.Array | None = None) -> chex.Array:
        if mask is None:
            mask = self.get_variable("cache", "mask")
            mask = jnp.roll(mask, -x.shape[1], axis=1).at[:, -x.shape[1] :].set(True)
        self.put_variable("cache", "mask", mask)

        # Create an attention bias to mask the attention probability which should be
        # ignored. To mask the future tokens, `jnp.tril` is used to the extended
        # attention bias array. We use `-1e9` which is relatively high penalty to make
        # the exponential value to zero.
        attn_bias = jnp.repeat(mask[:, None, None, :], x.shape[1], axis=2)
        attn_bias = jnp.tril(attn_bias, k=attn_bias.shape[3] - attn_bias.shape[2])
        attn_bias = -1e9 * (1 - attn_bias.astype(jnp.bfloat16))
        x = self.embed_in(x)
        x = self.layers(x, attn_bias)
        return self.final_layer_norm(x)

class GPTNeoXForCausalLM(nn.Module):
    vocab_size: int
    layers: int
    dim: int
    heads: int
    rotary: int
    hidden: int
    eps: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.gpt_neox = GPTNeoXModel(self.vocab_size, self.layers, self.dim, self.heads, self.rotary, self.hidden, self.eps, self.dtype)
        self.embed_out = nn.Dense(self.vocab_size, use_bias=False, dtype=self.dtype)

    def __call__(self, x: chex.Array, mask: chex.Array | None = None) -> chex.Array:
        return self.embed_out(self.gpt_neox(x, mask))