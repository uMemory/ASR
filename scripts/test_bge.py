"""快速测试 BGE-M3 加载和编码。"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
torch.cuda.empty_cache()

from src.retrieval.embedding import EmbeddingEncoder

print("加载 BGE-M3 (CUDA)...")
t0 = time.time()
enc = EmbeddingEncoder(model_path="E:/ASR/models/bge-m3", device="cuda")
enc.load()
print(f"  加载完成: {time.time()-t0:.1f}s")

t0 = time.time()
r = enc.encode(["测试文本", "另一个测试", "今天讨论的内容是关于预算的问题"])
print(f"  编码完成: {time.time()-t0:.1f}s")
print(f"  Dense shape: {r.dense_vecs.shape}")
norm = (r.dense_vecs[0] ** 2).sum()
print(f"  L2 norm (应接近 1.0): {norm:.4f}")

enc.unload()
print("✓ BGE-M3 测试通过")
