
import os ,sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yoloe26_weights import yoloe26n_pf

from ultralytics import YOLO
import torch


yoloe1=YOLO(yoloe26n_pf)
yoloe2=YOLO("yoloe-26n.yaml").load(yoloe26n_pf)

print("\n" + "="*60)
print("Model Comparison")
print("="*60)

# 比较模型结构
print("\n1. Model Structure:")
print(f"   Model 1 (direct load): {type(yoloe1.model).__name__}")
print(f"   Model 2 (yaml+load):   {type(yoloe2.model).__name__}")

# 比较参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

params1 = count_parameters(yoloe1.model)
params2 = count_parameters(yoloe2.model)

print(f"\n2. Parameter Count:")
print(f"   Model 1: {params1:,} parameters")
print(f"   Model 2: {params2:,} parameters")
print(f"   Difference: {abs(params1-params2):,} parameters")

# 比较head配置
head1 = yoloe1.model.model[-1]
head2 = yoloe2.model.model[-1]

print(f"\n3. Head Configuration:")
print(f"   Model 1 head type: {type(head1).__name__}")
print(f"   Model 2 head type: {type(head2).__name__}")
print(f"   Model 1 nc: {head1.nc}")
print(f"   Model 2 nc: {head2.nc}")
print(f"   Model 1 is_fused: {getattr(head1, 'is_fused', 'N/A')}")
print(f"   Model 2 is_fused: {getattr(head2, 'is_fused', 'N/A')}")
print(f"   Model 1 has lrpc: {hasattr(head1, 'lrpc')}")
print(f"   Model 2 has lrpc: {hasattr(head2, 'lrpc')}")
print(f"   Model 1 training mode: {yoloe1.model.training}")
print(f"   Model 2 training mode: {yoloe2.model.training}")

# 检查 cv3 层数（是否被删除最后一层）
if hasattr(head1, 'cv3') and head1.cv3 is not None:
    print(f"   Model 1 cv3[0] layers: {len(head1.cv3[0]) if isinstance(head1.cv3[0], torch.nn.Sequential) else 'N/A'}")
if hasattr(head2, 'cv3') and head2.cv3 is not None:
    print(f"   Model 2 cv3[0] layers: {len(head2.cv3[0]) if isinstance(head2.cv3[0], torch.nn.Sequential) else 'N/A'}")

# 检查分支
print(f"\n4. Branch Configuration:")
print(f"   Model 1 has cv2: {hasattr(head1, 'cv2')}")
print(f"   Model 1 has cv3: {hasattr(head1, 'cv3')}")
print(f"   Model 1 has one2one_cv2: {hasattr(head1, 'one2one_cv2')}")
print(f"   Model 1 has one2one_cv3: {hasattr(head1, 'one2one_cv3')}")
print(f"   Model 2 has cv2: {hasattr(head2, 'cv2')}")
print(f"   Model 2 has cv3: {hasattr(head2, 'cv3')}")
print(f"   Model 2 has one2one_cv2: {hasattr(head2, 'one2one_cv2')}")
print(f"   Model 2 has one2one_cv3: {hasattr(head2, 'one2one_cv3')}")

# 比较权重
print(f"\n5. Weight Comparison:")
with torch.no_grad():
    # 选择一些层来比较
    state1 = yoloe1.model.state_dict()
    state2 = yoloe2.model.state_dict()
    
    # 检查键是否相同
    keys1 = set(state1.keys())
    keys2 = set(state2.keys())
    
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common = keys1 & keys2
    
    print(f"   Common keys: {len(common)}")
    print(f"   Only in Model 1: {len(only_in_1)}")
    print(f"   Only in Model 2: {len(only_in_2)}")
    
    if only_in_1:
        print(f"\n   Keys only in Model 1 ({len(only_in_1)} total):")
        for key in sorted(only_in_1):
            print(f"     - {key}")
    
    if only_in_2:
        print(f"\n   Keys only in Model 2 ({len(only_in_2)} total):")
        # 按模块分组显示
        from collections import defaultdict
        grouped = defaultdict(list)
        for key in sorted(only_in_2):
            # 提取模块前缀（如 model.23.cv3, model.23.lrpc 等）
            parts = key.split('.')
            if len(parts) >= 3:
                prefix = '.'.join(parts[:3])
            else:
                prefix = 'other'
            grouped[prefix].append(key)
        
        for prefix, keys in sorted(grouped.items()):
            print(f"\n     [{prefix}] - {len(keys)} keys:")
            for key in keys[:10]:  # 每个模块最多显示10个
                print(f"       - {key}")
            if len(keys) > 10:
                print(f"       ... and {len(keys)-10} more")
    
    # 比较共同权重的差异
    if common:
        max_diff = 0
        max_diff_key = ""
        shape_mismatches = []
        for key in list(common)[:10]:  # 只检查前10个
            try:
                if state1[key].shape != state2[key].shape:
                    shape_mismatches.append((key, state1[key].shape, state2[key].shape))
                    continue
                diff = (state1[key] - state2[key]).abs().max().item()
                if diff > max_diff:
                    max_diff = diff
                    max_diff_key = key
            except Exception as e:
                print(f"   ⚠️ Error comparing {key}: {e}")
        
        print(f"\n   Max weight difference (first 10 keys): {max_diff:.2e}")
        if max_diff > 0:
            print(f"   Key with max diff: {max_diff_key}")
        
        if shape_mismatches:
            print(f"\n   ⚠️ Shape mismatches found ({len(shape_mismatches)}):")
            for key, shape1, shape2 in shape_mismatches[:3]:
                print(f"     {key}: {shape1} vs {shape2}")

# 测试推理
print(f"\n6. Inference Test:")
test_input = torch.randn(1, 3, 640, 640)
if torch.cuda.is_available():
    test_input = test_input.cuda()
    yoloe1.model.cuda()
    yoloe2.model.cuda()

try:
    with torch.no_grad():
        out1 = yoloe1.model(test_input)
        out2 = yoloe2.model(test_input)
        
    print(f"   Model 1 output type: {type(out1)}")
    print(f"   Model 2 output type: {type(out2)}")
    
    # 分析输出差异的原因
    print(f"\n   Output Type Analysis:")
    print(f"   Model 1 is_fused={head1.is_fused}, has_lrpc={hasattr(head1, 'lrpc')}")
    print(f"   Model 2 is_fused={head2.is_fused}, has_lrpc={hasattr(head2, 'lrpc')}")
    print(f"   → Model 1 returns tuple because it's a fused PF model (forward_lrpc)")
    print(f"   → Model 2 returns dict because it's not fused (forward_head)")
    
    if isinstance(out1, torch.Tensor) and isinstance(out2, torch.Tensor):
        print(f"\n   Model 1 output shape: {out1.shape}")
        print(f"   Model 2 output shape: {out2.shape}")
        if out1.shape == out2.shape:
            diff = (out1 - out2).abs().max().item()
            print(f"   Max output difference: {diff:.2e}")
    elif isinstance(out1, (tuple, list)) and isinstance(out2, (tuple, list)):
        print(f"\n   Model 1 outputs: {len(out1)} tensors")
        print(f"   Model 2 outputs: {len(out2)} tensors")
        if len(out1) == len(out2):
            for i, (o1, o2) in enumerate(zip(out1, out2)):
                if isinstance(o1, torch.Tensor) and isinstance(o2, torch.Tensor):
                    diff = (o1 - o2).abs().max().item()
                    print(f"   Output {i} diff: {diff:.2e}, shape: {o1.shape}")
    elif isinstance(out1, tuple) and isinstance(out2, dict):
        print(f"\n   ⚠️ Cannot directly compare tuple vs dict outputs")
        print(f"   Model 1 (tuple): {len(out1)} elements")
        if len(out1) > 0:
            print(f"     - Element 0 type: {type(out1[0])}")
            if isinstance(out1[0], torch.Tensor):
                print(f"       Shape: {out1[0].shape}")
            if len(out1) > 1:
                print(f"     - Element 1 type: {type(out1[1])}")
                if isinstance(out1[1], dict):
                    print(f"       Dict keys: {list(out1[1].keys())}")
        
        print(f"   Model 2 (dict): {len(out2)} keys")
        print(f"     - Keys: {list(out2.keys())}")
        for key, val in out2.items():
            if isinstance(val, torch.Tensor):
                print(f"       {key}: {val.shape}")
            elif isinstance(val, dict):
                print(f"       {key}: dict with keys {list(val.keys())}")
    elif isinstance(out1, dict) and isinstance(out2, dict):
        print(f"\n   Both outputs are dicts")
        print(f"   Model 1 keys: {list(out1.keys())}")
        print(f"   Model 2 keys: {list(out2.keys())}")
        
        common_keys = set(out1.keys()) & set(out2.keys())
        for key in common_keys:
            if isinstance(out1[key], torch.Tensor) and isinstance(out2[key], torch.Tensor):
                if out1[key].shape == out2[key].shape:
                    diff = (out1[key] - out2[key]).abs().max().item()
                    print(f"   {key} diff: {diff:.2e}, shape: {out1[key].shape}")
                else:
                    print(f"   {key} shape mismatch: {out1[key].shape} vs {out2[key].shape}")
                    
except Exception as e:
    print(f"   ✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)



