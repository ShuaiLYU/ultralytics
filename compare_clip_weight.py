
import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)


from ultralytics.nn.text_model import  MobileCLIPTS,TextModel,OpenCLIP
import torch
import numpy as np

import os,sys
import open_clip
import copy
import numpy as np
from ultralytics.utils.torch_utils import smart_inference_mode


device="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES","0")!="cpu" else "cpu"

print("Loading mobileclip2_b.pt...")
model=OpenCLIP(device, "mobileclip2_b.pt")

print("Loading mobileclip2_b.ts...")
model2=MobileCLIPTS(device,"mobileclip2_b.ts")

# 测试文本
test_texts = ["a photo of a cat", "a photo of a dog", "car", "person walking"]

print("\n" + "="*60)
print("Testing text encoding...")
print("="*60)

# 对比文本编码
with torch.no_grad():
    # OpenClip model
    tokens1 = model.tokenize(test_texts)
    embeddings1 = model.encode_text(tokens1)
    
    # MobileCLIPTS model  
    tokens2 = model2.tokenize(test_texts)
    embeddings2 = model2.encode_text(tokens2)
    
    print(f"\nOpenClip (.pt) embeddings shape: {embeddings1.shape}")
    print(f"MobileCLIPTS (.ts) embeddings shape: {embeddings2.shape}")
    
    # 对比差异
    if embeddings1.shape == embeddings2.shape:
        diff = (embeddings1 - embeddings2).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\nMax absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        
        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=-1)
        print(f"Cosine similarity (per text):")
        for i, text in enumerate(test_texts):
            print(f"  '{text}': {cos_sim[i].item():.6f}")
        
        # 归一化后对比
        emb1_norm = torch.nn.functional.normalize(embeddings1, dim=-1)
        emb2_norm = torch.nn.functional.normalize(embeddings2, dim=-1)
        norm_diff = (emb1_norm - emb2_norm).abs()
        print(f"\nNormalized max diff: {norm_diff.max().item():.6f}")
        print(f"Normalized mean diff: {norm_diff.mean().item():.6f}")
        
        if max_diff < 1e-5:
            print("\n✓ Models produce nearly identical outputs!")
        elif max_diff < 1e-3:
            print("\n⚠ Models have small differences (may be due to precision)")
        else:
            print("\n✗ Models produce significantly different outputs!")
    else:
        print("\n✗ Shape mismatch! Models are incompatible.")
        
print("\n" + "="*60)
print("Model information:")
print("="*60)
print(f"OpenClip device: {next(model.model.parameters()).device}")
print(f"MobileCLIPTS device: {next(model2.model.parameters()).device if hasattr(model2.model, 'parameters') else 'N/A (TorchScript)'}")

# 输出示例 embedding
print(f"\nSample embedding from OpenClip (first 10 dims):")
print(embeddings1[0, :10].cpu().numpy())
print(f"\nSample embedding from MobileCLIPTS (first 10 dims):")
print(embeddings2[0, :10].cpu().numpy())

