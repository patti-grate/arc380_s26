import sys, torch, torch.nn as nn, math

FEATURE_DIM = 33
HIDDEN_DIM  = 128
N_HEADS     = 4
N_LAYERS    = 2
FF_DIM      = 256
DROPOUT     = 0.1
K_MIXTURES  = 5
POSE_DIM    = 4

class NextBrickModel(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM,
                 n_heads=N_HEADS, n_layers=N_LAYERS, ff_dim=FF_DIM,
                 dropout=DROPOUT, K=K_MIXTURES, pose_dim=POSE_DIM,
                 max_layer_classes=13):
        super().__init__()
        self.n_layers_cls = max_layer_classes
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder   = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.b_head    = nn.Sequential(nn.Linear(hidden_dim,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,2))
        self.layer_head= nn.Sequential(nn.Linear(hidden_dim,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,max_layer_classes))
        mdn_in = hidden_dim + 3
        self.mdn_pi        = nn.Linear(mdn_in, K)
        self.mdn_mu        = nn.Linear(mdn_in, K * pose_dim)
        self.mdn_log_sigma = nn.Linear(mdn_in, K * pose_dim)

ckpt_path = "training_data/trained_models/best_model_z_refactored.pth"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
max_layer_classes = int(ckpt.get("max_layer_classes", 13))
print(f"max_layer_classes from checkpoint: {max_layer_classes}")
print(f"std_uv in ckpt: {ckpt.get('std_uv', 'NOT FOUND')}")

model = NextBrickModel(max_layer_classes=max_layer_classes)
missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=True)
print(f"Missing keys:    {missing}")
print(f"Unexpected keys: {unexpected}")
print("Load OK")

# quick forward pass
tokens = torch.zeros(1, 5, FEATURE_DIM)
mask   = torch.zeros(1, 6, dtype=torch.bool)
with torch.no_grad():
    x    = model.input_proj(tokens)
    cls  = model.cls_token.expand(1, -1, -1)
    x    = torch.cat([cls, x], dim=1)
    x    = model.encoder(x, src_key_padding_mask=mask)
    scene = x[:, 0]
    b_logits     = model.b_head(scene)
    layer_logits = model.layer_head(scene)
    b_pred     = b_logits.argmax(dim=-1).float().unsqueeze(-1)
    layer_pred = layer_logits.argmax(dim=-1).float().unsqueeze(-1) / max_layer_classes
    has_pair   = torch.zeros(1, 1)
    cond = torch.cat([scene, b_pred, layer_pred, has_pair], dim=-1)
    pi   = model.mdn_pi(cond)
    mu   = model.mdn_mu(cond)
    lsig = model.mdn_log_sigma(cond)
    print(f"pi shape: {pi.shape}  (expected [1,5])")
    print(f"mu shape: {mu.shape}  (expected [1,20])")
    print(f"lsig shape: {lsig.shape}  (expected [1,20])")
print("Forward pass OK")
