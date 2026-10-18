#!/usr/bin/env python3
"""Requantize a BF16 mmproj GGUF to Q8_0 with llama.cpp's own tensor rules.

Rules (conversion/base.py and conversion/qwenvl.py in the pinned llama.cpp):
  * 1-D tensors, *_norm.weight and anything not ending in .weight (biases): F32
  * position embeddings (v.position_embd.*): F32
  * convolution weights (patch embedding, .conv in the HF name -> v.patch_embd.* in GGUF): F16
  * every other 2-D .weight: Q8_0
Metadata is copied as is, except general.file_type which becomes MOSTLY_Q8_0.
"""
import sys
from pathlib import Path
import numpy as np

LLAMA = Path('/home/aim/opt/UnbiasedStrata/third_party/llama.cpp/gguf-py')
sys.path.insert(0, str(LLAMA))
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as T, GGUFReader, GGUFWriter, quants  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
r = GGUFReader(str(src))
arch = None
for f in r.fields.values():
    if f.name == 'general.architecture':
        arch = bytes(f.parts[f.data[0]]).decode()
w = GGUFWriter(str(dst), arch or 'clip')


def bf16_to_f32(a: np.ndarray) -> np.ndarray:
    u16 = a.view(np.uint16).astype(np.uint32) << 16
    return u16.view(np.float32)


skipped_keys = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count', 'general.architecture', 'general.file_type'}
for f in r.fields.values():
    if f.name in skipped_keys:
        continue
    vt = f.types[0]
    if vt == gguf.GGUFValueType.ARRAY:
        et = f.types[1]
        vals = [f.parts[i] for i in f.data]
        if et == gguf.GGUFValueType.STRING:
            w.add_array(f.name, [bytes(v).decode() for v in vals])
        else:
            w.add_array(f.name, [v[0].item() for v in vals])
    elif vt == gguf.GGUFValueType.STRING:
        w.add_string(f.name, bytes(f.parts[f.data[0]]).decode())
    else:
        val = f.parts[f.data[0]][0].item()
        adder = {gguf.GGUFValueType.UINT8: w.add_uint8, gguf.GGUFValueType.INT8: w.add_int8,
                 gguf.GGUFValueType.UINT16: w.add_uint16, gguf.GGUFValueType.INT16: w.add_int16,
                 gguf.GGUFValueType.UINT32: w.add_uint32, gguf.GGUFValueType.INT32: w.add_int32,
                 gguf.GGUFValueType.UINT64: w.add_uint64, gguf.GGUFValueType.INT64: w.add_int64,
                 gguf.GGUFValueType.FLOAT32: w.add_float32, gguf.GGUFValueType.FLOAT64: w.add_float64,
                 gguf.GGUFValueType.BOOL: w.add_bool}[vt]
        adder(f.name, val)
w.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)

counts = {}
for t in r.tensors:
    shape = [int(x) for x in reversed(t.shape.tolist())]   # gguf stores ne reversed vs numpy
    data = np.array(t.data)
    if t.tensor_type == T.BF16:
        f32 = bf16_to_f32(data.reshape(-1)).reshape(shape)
    elif t.tensor_type == T.F16:
        f32 = data.astype(np.float32).reshape(shape)
    elif t.tensor_type == T.F32:
        f32 = data.reshape(shape).astype(np.float32)
    else:
        raise SystemExit(f'{t.name}: unexpected source type {t.tensor_type.name}')
    n_dims = len(shape)
    name = t.name
    if n_dims <= 1 or name.endswith('_norm.weight') or not name.endswith('.weight') or 'position_embd' in name:
        q = T.F32
    elif 'patch_embd' in name:
        q = T.F16
    else:
        q = T.Q8_0
        if shape[-1] % 32 != 0:
            q = T.F16
    if q == T.F32:
        out = f32
    elif q == T.F16:
        out = f32.astype(np.float16)
    else:
        out = quants.quantize(f32, T.Q8_0)
    w.add_tensor(name, out, raw_shape=None if q != T.Q8_0 else None, raw_dtype=q if q == T.Q8_0 else None)
    counts[q.name] = counts.get(q.name, 0) + 1

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file(progress=False)
w.close()
print('tensors by type:', counts)
print('bytes', src.stat().st_size, '->', dst.stat().st_size)
