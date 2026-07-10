"""Quick script to check what's in the checkpoint file"""
import torch

checkpoint_path = 'checkpoints/final_model.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("="*70)
print("CHECKPOINT CONTENTS")
print("="*70)
print(f"\nKeys in checkpoint: {list(checkpoint.keys())}\n")

for key, value in checkpoint.items():
    if isinstance(value, dict):
        print(f"{key}: (dict with {len(value)} items)")
        if len(value) < 20:  # Only show if not too many
            for k in list(value.keys())[:5]:
                print(f"  - {k}")
            if len(value) > 5:
                print(f"  ... and {len(value)-5} more")
    elif isinstance(value, (int, float, str)):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {type(value)}")

print("\n" + "="*70)
