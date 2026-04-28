import torch

ckpt = torch.load(
    "training_data/trained_models/best_model_z_refactored.pth",
    map_location="cpu", weights_only=False
)
print("Top-level keys:", list(ckpt.keys()))
print()
for k, v in ckpt["model_state"].items():
    print(f"  {k}: {v.shape}")
