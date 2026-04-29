import torch, json, sys
train = torch.load('/Users/palakaggarwal/Documents/musically-aligned-translation/data_gathering/output/disney_train_data.pt', weights_only=False)
test = torch.load('/Users/palakaggarwal/Documents/musically-aligned-translation/data_gathering/output/disney_test_data.pt', weights_only=False)

print(f"Train: {len(train)} examples")
print(f"Test:  {len(test)} examples")
print()

def describe(ex, idx):
    print(f"--- Example {idx} ---")
    for k, v in ex.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: tensor shape {tuple(v.shape)} dtype {v.dtype}")
        elif isinstance(v, str):
            s = v if len(v) < 220 else v[:217] + "..."
            print(f"  {k}: {s!r}")
        elif isinstance(v, (list, tuple)):
            tn = type(v).__name__
            sample = v[:5] if len(v) > 0 else v
            print(f"  {k}: {tn} length {len(v)} sample {sample}")
        else:
            print(f"  {k}: {v!r}")
    print()

for i, ex in enumerate(train):
    describe(ex, i)

print("=== TEST ===")
for i, ex in enumerate(test):
    describe(ex, i)

try:
    fma = torch.load('/Users/palakaggarwal/Documents/musically-aligned-translation/src/data/processed/fma_train_data.pt', weights_only=False)
    print(f"\nFMA train: {len(fma)} examples")
    print("FMA[0] keys:", list(fma[0].keys()))
except Exception as e:
    print(f"\nCould not load FMA for comparison: {e}")
