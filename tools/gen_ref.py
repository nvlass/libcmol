#!/usr/bin/env python3
"""
gen_ref.py — reference logit generator for libcmol model tests.

Runs a fixed sequence of token IDs through a pure-Python transformer
(using gguf.quants.dequantize as the dequant ground truth) and writes
per-position logits to a binary file for comparison with tests/test_model_ref.c.

Usage:
    python3 tools/gen_ref.py <model.gguf> <output.bin> [--n-gen N] [--tokens A,B,...]

Binary format (all little-endian):
    uint32  magic    = 0x4D4F4C52
    uint32  version  = 1
    int32   n_prompt  (prefill token count)
    int32   vocab     (vocabulary size)
    int32   n_gen     (generation steps recorded after prefill)
    int32   tokens[n_prompt]
    # n_gen+1 records — last prefill pos, then n_gen generation positions:
    for each record:
        int32    best_tok   (greedy argmax at this position)
        float32  logits[vocab]

Requirements:  pip install gguf numpy
"""

import sys
import struct
import argparse
import numpy as np

try:
    import gguf
    from gguf.quants import dequantize
except ImportError:
    sys.exit("Error: pip install gguf")

MAGIC   = 0x4D4F4C52
VERSION = 1

# Reproducible prompt: BOS (1) + 9 tokens spanning the vocabulary.
# All IDs must be < vocab_size (49152 for SmolLM2/3).
DEFAULT_PROMPT = [1, 3000, 7500, 12000, 18000, 25000, 30000, 35000, 40000, 45000]


# ---------------------------------------------------------------------------
# GGUF helpers
# ---------------------------------------------------------------------------

def field_scalar(reader, key, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    return f.parts[-1][0]


def field_string(reader, key, default=""):
    f = reader.fields.get(key)
    if f is None:
        return default
    return bytes(f.parts[-1]).decode("utf-8", errors="replace")


def load_hparams(reader):
    arch = field_string(reader, "general.architecture", "llama")
    p    = arch + "."
    d      = int(field_scalar(reader, p + "embedding_length",               960))
    n_lay  = int(field_scalar(reader, p + "block_count",                     32))
    n_h    = int(field_scalar(reader, p + "attention.head_count",             15))
    n_kv   = int(field_scalar(reader, p + "attention.head_count_kv",           5))
    d_ffn  = int(field_scalar(reader, p + "feed_forward_length",            2560))
    eps    = float(field_scalar(reader, p + "attention.layer_norm_rms_epsilon", 1e-5))
    freq   = float(field_scalar(reader, p + "rope.freq_base",             100000.0))
    nrope_interval = int(field_scalar(reader, p + "no_rope_layer_interval",      0))

    # vocab_size from the embedding tensor shape (GGUF: shape[0]=d, shape[1]=vocab)
    tmap = {t.name: t for t in reader.tensors}
    embd = tmap["token_embd.weight"]
    # total elements / d_model = vocab_size
    vocab = int(np.prod(embd.shape)) // d

    return dict(
        arch=arch, d=d, n_layers=n_lay, n_heads=n_h, n_kv_heads=n_kv,
        d_head=d // n_h, d_ffn=d_ffn, vocab=vocab,
        eps=eps, freq=freq, nrope_interval=nrope_interval,
        tie_embd="output.weight" not in tmap,
    )


# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------

class RefModel:
    def __init__(self, reader, hp, max_ctx=512):
        self.hp    = hp
        self._tmap = {t.name: t for t in reader.tensors}
        self._W    = {}          # dequantized weight cache

        kv_dim = hp["n_kv_heads"] * hp["d_head"]
        self.kv_k = [np.zeros((max_ctx, kv_dim), np.float32)
                     for _ in range(hp["n_layers"])]
        self.kv_v = [np.zeros((max_ctx, kv_dim), np.float32)
                     for _ in range(hp["n_layers"])]

    def get(self, name):
        """Return weight as float32 ndarray.

        2-D tensors are returned as [out_features, in_features] (matmul-ready).
        1-D tensors (norms) are returned as [n].
        """
        if name not in self._W:
            t    = self._tmap[name]
            flat = dequantize(t.data, t.tensor_type).astype(np.float32)
            if t.shape.ndim == 0 or len(t.shape) == 1:
                self._W[name] = flat.reshape(-1)
            else:
                # GGUF convention: shape[0]=in (innermost), shape[1]=out (outermost)
                n_in  = int(t.shape[0])
                n_out = int(np.prod(t.shape)) // n_in
                self._W[name] = flat.reshape(n_out, n_in)
            print(f"  dequant  {name:50s}  {self._W[name].shape}", flush=True)
        return self._W[name]

    def rms_norm(self, x, w_name):
        w  = self.get(w_name)
        ss = float(np.mean(x * x)) + self.hp["eps"]
        return w * (x / np.sqrt(ss))

    def forward(self, token_id, pos):
        hp = self.hp
        d, n_h, n_kv, dh = hp["d"], hp["n_heads"], hp["n_kv_heads"], hp["d_head"]
        half = dh // 2
        freq = hp["freq"]

        # 1. Token embedding
        embd = self.get("token_embd.weight")   # [vocab, d]
        x    = embd[token_id].copy()

        # 2. Transformer layers
        for layer in range(hp["n_layers"]):
            pfx = f"blk.{layer}."

            # ---- Attention ------------------------------------------------
            x_norm = self.rms_norm(x, pfx + "attn_norm.weight")

            q = self.get(pfx + "attn_q.weight")      @ x_norm   # [d]
            k = self.get(pfx + "attn_k.weight")      @ x_norm   # [kv_dim]
            v = self.get(pfx + "attn_v.weight")      @ x_norm   # [kv_dim]

            # Optional QK-norm (SmolLM3)
            qn = self._tmap.get(pfx + "attn_q_norm.weight")
            kn = self._tmap.get(pfx + "attn_k_norm.weight")
            if qn is not None:
                w = self.get(pfx + "attn_q_norm.weight")
                for h in range(n_h):
                    s = h * dh
                    qh = q[s:s+dh]
                    ss = float(np.mean(qh * qh)) + hp["eps"]
                    q[s:s+dh] = w * (qh / np.sqrt(ss))
            if kn is not None:
                w = self.get(pfx + "attn_k_norm.weight")
                for h in range(n_kv):
                    s = h * dh
                    kh = k[s:s+dh]
                    ss = float(np.mean(kh * kh)) + hp["eps"]
                    k[s:s+dh] = w * (kh / np.sqrt(ss))

            # RoPE (skip for NoPE layers in SmolLM3)
            nrope_skip = (hp["nrope_interval"] > 0 and
                          (layer + 1) % hp["nrope_interval"] == 0)
            if not nrope_skip:
                for h in range(n_h):
                    qh = q[h*dh:(h+1)*dh]
                    for i in range(half):
                        theta = pos / (freq ** (2.0 * i / dh))
                        cs, sn = np.cos(theta), np.sin(theta)
                        q0, q1 = float(qh[i]), float(qh[i+half])
                        qh[i]       = q0*cs - q1*sn
                        qh[i+half]  = q0*sn + q1*cs
                for h in range(n_kv):
                    kh = k[h*dh:(h+1)*dh]
                    for i in range(half):
                        theta = pos / (freq ** (2.0 * i / dh))
                        cs, sn = np.cos(theta), np.sin(theta)
                        k0, k1 = float(kh[i]), float(kh[i+half])
                        kh[i]       = k0*cs - k1*sn
                        kh[i+half]  = k0*sn + k1*cs

            # Write KV cache
            self.kv_k[layer][pos] = k
            self.kv_v[layer][pos] = v

            # GQA scaled dot-product attention
            seqlen   = pos + 1
            scale    = 1.0 / np.sqrt(float(dh))
            attn_out = np.zeros(d, np.float32)

            for h in range(n_h):
                h_kv = h * n_kv // n_h
                qh   = q[h*dh:(h+1)*dh]
                K    = self.kv_k[layer][:seqlen, h_kv*dh:(h_kv+1)*dh]  # [seqlen, dh]
                V    = self.kv_v[layer][:seqlen, h_kv*dh:(h_kv+1)*dh]

                scores  = (K @ qh) * scale          # [seqlen]
                scores -= scores.max()
                w_attn  = np.exp(scores)
                w_attn /= w_attn.sum()
                attn_out[h*dh:(h+1)*dh] = w_attn @ V

            x = x + self.get(pfx + "attn_output.weight") @ attn_out

            # ---- FFN (SwiGLU) --------------------------------------------
            x_norm = self.rms_norm(x, pfx + "ffn_norm.weight")

            gate = self.get(pfx + "ffn_gate.weight") @ x_norm  # [d_ffn]
            up   = self.get(pfx + "ffn_up.weight")   @ x_norm  # [d_ffn]
            silu = gate / (1.0 + np.exp(-gate))                # SiLU
            x    = x + self.get(pfx + "ffn_down.weight") @ (silu * up)

        # 3. Final norm + LM head
        x_norm = self.rms_norm(x, "output_norm.weight")
        lm     = "token_embd.weight" if hp["tie_embd"] else "output.weight"
        return self.get(lm) @ x_norm   # [vocab]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate reference logits for libcmol model tests.")
    ap.add_argument("gguf_path",  help="Path to GGUF model file")
    ap.add_argument("out_path",   help="Output binary file")
    ap.add_argument("--n-gen",    type=int, default=3,
                    help="Generation steps to record after prefill (default 3)")
    ap.add_argument("--tokens",   default=None,
                    help="Comma-separated prompt token IDs (default: built-in)")
    args = ap.parse_args()

    prompt = (list(map(int, args.tokens.split(",")))
              if args.tokens else DEFAULT_PROMPT)
    n_prompt = len(prompt)
    n_gen    = args.n_gen

    print(f"Loading {args.gguf_path} ...", flush=True)
    reader = gguf.GGUFReader(args.gguf_path)
    hp     = load_hparams(reader)
    print(f"  arch={hp['arch']}  d={hp['d']}  layers={hp['n_layers']}"
          f"  heads={hp['n_heads']}/{hp['n_kv_heads']}"
          f"  vocab={hp['vocab']}  freq={hp['freq']}")
    print(f"  tie_embd={hp['tie_embd']}  nrope_interval={hp['nrope_interval']}")

    model  = RefModel(reader, hp, max_ctx=n_prompt + n_gen + 4)
    vocab  = hp["vocab"]

    # Validate token IDs
    for tok in prompt:
        if tok < 0 or tok >= vocab:
            sys.exit(f"Token ID {tok} out of range [0, {vocab})")

    # --- Prefill ---
    print(f"\nPrefilling {n_prompt} tokens: {prompt}", flush=True)
    logits = None
    for i, tok in enumerate(prompt):
        logits = model.forward(tok, i)
        best   = int(np.argmax(logits))
        print(f"  pos={i:3d}  tok={tok:5d}  best={best:5d}  "
              f"logit[best]={logits[best]:.4f}", flush=True)

    records = []

    # Record for last prefill position
    best_tok = int(np.argmax(logits))
    records.append((best_tok, logits.copy()))
    print(f"\nLast prefill pos={n_prompt-1}: best_tok={best_tok}"
          f"  logit={logits[best_tok]:.4f}")

    # --- Generation ---
    tok = best_tok
    for g in range(n_gen):
        pos    = n_prompt + g
        logits = model.forward(tok, pos)
        best   = int(np.argmax(logits))
        records.append((best, logits.copy()))
        print(f"Gen pos={pos}: input={tok}  best={best}  "
              f"logit={logits[best]:.4f}", flush=True)
        tok = best

    # --- Write binary ---
    with open(args.out_path, "wb") as f:
        f.write(struct.pack("<II", MAGIC, VERSION))
        f.write(struct.pack("<iii", n_prompt, vocab, n_gen))
        f.write(struct.pack(f"<{n_prompt}i", *prompt))
        for best_tok, lgs in records:
            f.write(struct.pack("<i", best_tok))
            f.write(lgs.astype(np.float32).tobytes())

    total_bytes = 8 + 12 + n_prompt*4 + len(records)*(4 + vocab*4)
    print(f"\nWrote {args.out_path}  ({len(records)} records, {total_bytes} bytes)")


if __name__ == "__main__":
    main()
