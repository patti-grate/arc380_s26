# Work Plan: Small Transformer/Set-Transformer Model for Next-Brick Pose Prediction

**Small Set/Transformer Encoder + Mixture Density Network next-pose head + hard validation**

## 0. Project Goal

Train a lightweight behavior-cloning model that predicts a plausible next brick pose from all previously placed brick poses in a partial structure.

The model should not predict a single deterministic “correct” next brick. Instead, it should predict a **multimodal distribution** of plausible next poses.

At inference time, the model proposes candidate poses, and the existing validation pipeline checks whether each candidate satisfies:

* geometric validity;
* no collision with existing bricks;
* structural or quasi-static stability;
* robot reachability;
* robot path collision avoidance;
* layer and assembly-sequence constraints.

The core principle is:

> The neural model learns where plausible next bricks are likely to be placed; the validator enforces whether the candidate is physically and robotically valid.

---

## 1. Problem Formulation

Each complete generated sequence is represented as:

```text
[p1, p2, p3, ..., pT]
```

where each brick pose is:

```text
p = [x, y, z, b, r]
```

with:

```text
x, y, z = brick center position
b       = binary state, laying or standing
r       = yaw rotation in the world XY plane
```

The supervised learning task is:

```text
placed brick history -> next brick pose
```

or:

```text
[p1, ..., pt] -> p(t+1)
```

This is a conditional generative modeling problem:

```text
p(next brick pose | current partial structure)
```

rather than simple regression.

---

## 2. Available Dataset

Current expected dataset:

```text
original valid sequences (validated_simPhysics from batch x-y)
20 to 40 bricks per sequence
```

Approximate number of supervised next-step samples before augmentation:

```text
25 sequences × 30 average bricks ≈ 750 training pairs
```

After augmentation:

```text
750 samples × 100 augmentations ≈ 75,000 augmented samples
```

Important note:

```text
75,000 augmented samples are not 75,000 independent design examples.
```

The augmentation mainly teaches spatial invariance/equivariance under translation and rotation. It does not create fundamentally new assembly logic. Therefore, the model should remain small and strongly regularized.

---

## 3. Dataset Construction

### 3.1 Convert Full Sequences into Next-Step Training Pairs

For each complete valid sequence:

```text
[p1, p2, p3, ..., pT]
```

create training pairs:

```text
[]                  -> p1
[p1]                -> p2
[p1, p2]            -> p3
[p1, p2, p3]        -> p4
...
[p1, ..., p(T-1)]   -> pT
```

Each sample contains:

```text
input:  placed brick history
target: next brick pose
```

The input length varies from 0 to T-1.

---

### 3.2 Sequence-Level Train/Validation/Test Split

Do not randomly split augmented samples.

Use sequence-level splitting to avoid data leakage:

```text
20 original sequences -> training
5 original sequences  -> validation
testing will be sequential brick spawning in simulation
```

Only the training sequences should be augmented.

Validation and test sequences should remain unaugmented or only use deterministic canonical normalization.

Reason:

```text
If augmented versions of the same original sequence appear in both training and validation, validation accuracy will be artificially high.
```

---

## 4. Pose Canonicalization

### 4.1 Problem

A single physical brick pose may have multiple equivalent parameterizations because of brick symmetry.

For example, due to geometric symmetry, the same placed brick may be represented by several flipped or rotated local coordinate frames.

If these equivalent representations are treated as different labels, the model may see contradictory targets.

Example issue:

```text
same physical brick geometry
same position in the structure
different pose parameterization
```

This can confuse the network during training.

---

### 4.2 Canonicalization Strategy

Before training, convert every brick pose to a deterministic canonical representation.

For each 5d brick pose (x,y,z, b, r):

we replace the r term with 0 to pi sin,cos term: (abs(sin(r)), cos(r)*sin(r)/abs(sin(r)))

Use the same canonicalization rule for:

```text
input history poses
target next pose
inference-time observed poses
model-generated candidate poses
```

---

## 5. Pose Encoding

### 5.1 Avoid Raw Angle Regression

Do not directly regress raw yaw angle `r`.

Raw angle regression is problematic because of circular discontinuity.

For example:

```text
r = 0
r = π
```

represent nearly identical rotations but have very different numeric values.

Use sine/cosine encoding:

```text
sin_r = sin(r)
cos_r = cos(r)
```

Each brick pose becomes (r is adjusted to 0 to pi):

```text
[x, y, z, b, sin_r, cos_r]
```

---

### 5.2 Recommended Brick Token Features

Each placed brick should be encoded as one token:

```text
[x, y, z, b, sin_r, cos_r, layer_id, time_index, prior_layer_brick_number, current_layer_brick_number, prior_layer_average_b, prior_prior_layer_average_b]
```

```text
where the brick is
how it is oriented
whether it is standing or laying
which layer it belongs to
when it was placed
```

---

## 6. Layer and History Organization

### 6.1 Why Layer Information Matters

The structure follows a repeating layer pattern:

```text
layer 1: laying
layer 2: standing
layer 3: laying
repeat
```

Therefore, the model needs to understand not only chronological history but also layer organization.

---

### 6.2 Recommended Sorting

Before feeding tokens into the model, sort bricks using a deterministic rule.

obtain layer information by monitoring z changes over the sequence. 

---

### 6.3 Variable-Length History

Histories have variable length.

Use padding and attention masks.

demo format:

```text
tokens: [demo_size, max_num_bricks, feature_dim]
mask:   [demo_size, max_num_bricks]
```

Where:

```text
mask = 1 for real brick tokens
mask = 0 for padded tokens
```

The attention mask prevents the model from attending to padded positions.

---

## 7. Data Augmentation

### 7.1 Rigid SE(2) Augmentation

Apply global XY translation and Z-axis rotation centered on sample bbox center to the entire sample.

For each sample in training data:

```text
input history: [p1, ..., pt]
target pose:   p(t+1)
```

apply the same rigid transform to both:

```text
all input bricks
target next brick
```

The augmentation should preserve physical relations.

Transform:

```text
Rz(theta), translation(tx, ty)
```

where:

```text
theta = random yaw angle (0 to 360 degrees, then 5d pose r should be converted to between 0 to pi)
tx    = random incremental translation in X (-0.0100 to 0.0100)
ty    = random incremental translation in Y (-0.0100 to 0.0100)
```

Do not randomly perturb Z unless the structure generator explicitly supports vertical variation.

---

### 7.2 Number of Augmentations

Initial setting:

```text
100 transformed variants per next-step sample
```

---

### 7.3 Important Limitation

Rigid SE(2) augmentation teaches:

```text
the same structure can be built at different XY locations and global orientations
```

It does not teach:

```text
new topology
new layer logic
new stability behavior
new robot constraints
```

Therefore, the model should be evaluated on held-out original sequences, not only augmented variants.

---

## 9. Model Architecture

### 9.1 Recommended Architecture

Use a small attention-based encoder.

Overall structure:

```text
placed brick tokens
    ↓
MLP token embedding
    ↓
small Transformer encoder or Set Transformer encoder
    ↓
global pooling / context token
    ↓
prediction heads
```

The model should be small because the number of original sequences is limited.

---

### 9.2 Initial Model Size

Recommended starting configuration:

```text
input feature dimension: 12 as defined ealier
hidden dimension: 128
attention layers: 2
attention heads: 4
dropout: 0.1
feedforward dimension: 256
```

Alternative smaller version:

```text
hidden dimension: 64
attention layers: 2
attention heads: 4
dropout: 0.1 to 0.2
feedforward dimension: 128
```

Do not start with a large transformer.

Avoid:

```text
8+ layers
large hidden dimensions
large numbers of heads
large autoregressive language-model-style architecture
```

because the original dataset is too small.

---

### 9.3 Transformer Encoder vs Set Transformer

Use:

```text
small Transformer encoder
```

For this project, a practical first implementation can use a standard Transformer encoder with:

```text
deterministically sorted brick tokens
padding mask
layer/time features
```

This is easier to implement than a full Set Transformer and should be sufficient for the first prototype.

---

### 9.4 Global Scene Embedding

After attention layers, aggregate token embeddings into a single scene context vector.

Options:

```text
mean pooling over valid tokens
max pooling over valid tokens
learned [CLS] token
attention pooling
```

Recommended first choice:

```text
learned [CLS] token
```

The [CLS] token attends to all brick tokens and becomes the global scene representation.

---

## 10. Prediction Head

The output should represent multiple possible next poses.

Use:

```text
binary state head + mixture density pose head
```

---

### 10.1 Binary State Head

Predict the next brick state:

```text
b = laying
b = standing
```

Output:

```text
P(b = laying)
P(b = standing)
```

Loss:

```text
cross entropy
```

If the layer pattern already strongly determines `b`, this head may be easy to train. Still include it because it helps separate discrete state prediction from continuous pose prediction.

---

### 10.2 Continuous Pose Head

Predict a mixture distribution over:

```text
[x, y, z, sin_r, cos_r]
```

Use a Mixture Density Network head.

For K mixture components, output:

```text
mixture weights: π1, π2, ..., πK
means:           μ1, μ2, ..., μK
variances:       σ1, σ2, ..., σK
```

Initial setting:

```text
K = 5
```

If the model under-represents multimodality, increase to:

```text
K = 10
```

Do not start with too many modes because the dataset is limited.

---

### 10.3 Optional Conditional Head by Brick State

A stronger version is hierarchical:

```text
1. predict b
2. predict continuous pose conditioned on b
```

This is useful because laying and standing bricks may follow different spatial distributions.

First implementation can use a shared continuous pose head.

Second implementation can split into:

```text
MDN head for laying
MDN head for standing
```

---

## 11. Loss Function

Total loss:

```text
L = L_pose_mdn + λb * L_b + λreg * L_reg
```

Where:

```text
L_pose_mdn = negative log likelihood of target pose under predicted mixture (L2 norm-weighted?)
L_b        = cross entropy for binary state
L_reg      = regularization term
```

Initial weights:

```text
λb   = 1.0
λreg = 1e-4
```

---

### 11.1 MDN Negative Log Likelihood

For target continuous pose:

```text
y = [x, y, z, sin_r, cos_r]
```

the model predicts:

```text
p(y | scene) = Σ_k π_k * Normal(y | μ_k, σ_k)
```

The pose loss is:

```text
L_pose_mdn = -log p(y | scene)
```

Use numerical stabilization:

```text
log-sum-exp
minimum variance clamp
gradient clipping
```

Clamp variance:

```text
σ_min = 1e-3 or 1e-4 after normalization
```

---

### 11.2 Rotation Normalization Loss

Because `sin_r` and `cos_r` should lie on the unit circle, add:

```text
L_circle = (sin_r^2 + cos_r^2 - 1)^2
```

However, if the MDN predicts `sin_r` and `cos_r` as continuous values, this constraint is more important during sampling/post-processing than during likelihood training.

At inference, normalize sampled rotation vector:

```text
v = [sin_r, cos_r]
v = v / ||v||
r = atan2(sin_r, cos_r)
```

---

## 12. Training Setup

### 12.1 Recommended Hyperparameters

Initial training configuration:

```text
optimizer: AdamW
learning rate: 3e-4
batch size: 64
epochs: 150
weight decay: 1e-4
gradient clipping: 1.0
dropout: 0.1
```

Learning-rate schedule:

```text
ReduceLROnPlateau
```

or:

```text
cosine decay
```

Early stopping:

```text
stop if validation loss does not improve for 20 epochs
```

---

### 12.2 Normalization Statistics

Compute normalization statistics only from training data:

```text
mean_x, std_x
mean_y, std_y
mean_z, std_z
```

Use the same statistics for:

```text
training
validation
test
inference
```

Do not compute statistics from validation or test data.

---

### 12.3 Regularization

Because the real dataset is small, use:

```text
dropout
weight decay
early stopping
small model size
sequence-level validation split
```

Avoid overly large networks.

---

## 13. Inference Pipeline

At runtime:

```text
1. receive current placed brick poses
2. canonicalize each brick pose
3. assign layer_id and time_index
4. sort/group tokens by layer and spatial order
5. normalize scene into local frame
6. run model
7. sample K candidate next poses from predicted mixture
8. convert sin/cos back to yaw angle
9. convert local predictions back to world frame
10. convert canonical pose to runtime pose format if needed
11. run hard validation
12. select valid candidate
```

Suggested sample count:

```text
K = 50 to 200
```

---

### 13.1 Candidate Selection

For each sampled candidate, evaluate:

```text
model likelihood
collision validity
stability validity
reachability validity
motion-planning validity
layer-pattern validity
```

Candidate ranking can initially be:

```text
1. discard invalid candidates
2. choose highest model-likelihood valid candidate
```

Alternative ranking:

```text
score = model_log_likelihood
      - α * distance_to_preferred_region
      - β * robot_motion_cost
      - γ * stability_margin_penalty
```

Start simple:

```text
highest-likelihood valid candidate
```

---

### 13.2 Failure Handling

If no sampled candidate passes validation:

```text
1. increase K and resample
2. use lower-temperature or higher-temperature sampling
3. fall back to rule-based valid sequence generator
4. request replanning from current partial structure
```

Initial fallback:

```text
sample K = 100
if none valid, sample K = 500
if still none valid, call rule-based generator
```

---

## 14. Evaluation Metrics

Evaluate the model at both learning and system levels.

Pure prediction error is not enough because multiple next poses may be valid.

---

### 14.1 Learning-Level Metrics

Track:

```text
validation negative log likelihood
binary state accuracy
average position error to target
average yaw error to target
minimum distance to equivalent target pose
```

However, these metrics are limited because the target sequence gives only one valid next pose, while many other next poses may also be valid.

---

### 14.2 Candidate Validity Metrics

More important metrics:

```text
top-1 validity rate
top-5 validity rate
top-10 validity rate
top-50 validity rate
top-100 validity rate
```

Definitions:

```text
top-1 validity rate:
    percentage of samples where the highest-likelihood candidate is valid

top-10 validity rate:
    percentage of samples where at least one of the top 10 sampled candidates is valid

top-100 validity rate:
    percentage of samples where at least one of the top 100 sampled candidates is valid
```

---

### 14.3 Constraint-Specific Metrics

Report:

```text
collision-free rate
stability-valid rate
robot-reachable rate
path-collision-free rate
layer-pattern-valid rate
```

This helps diagnose which constraints the model struggles with.

---

### 14.4 Rollout Metrics

The most important evaluation is full sequence rollout.

Starting from:

```text
empty scene
```

or:

```text
partial seed scene
```

repeatedly predict and validate the next brick.

Track:

```text
average completed bricks before failure
full sequence success rate
number of validation rejections per accepted brick
average sampling count needed per accepted brick
diversity of generated structures
```

A useful metric:

```text
rollout success at N bricks
```

Example:

```text
successfully builds 10 bricks
successfully builds 20 bricks
successfully builds 30 bricks
```

---

## 15. Baselines

Compare against several baselines.

### 15.1 Rule-Based Generator

Use the existing valid sequence generator as the strongest non-learning baseline.

Metrics:

```text
validity rate
rollout success
diversity
runtime
```

---

### 15.2 Nearest-Neighbor Retrieval

Given a partial structure, retrieve the most similar partial structure from training data and use its next brick.

This is a direct comparison against KNN-like behavior.

Expected limitation:

```text
poor generalization to new partial structures
high sensitivity to pose-history alignment
```

---

### 15.3 Plain MLP Regression

Use pooled brick features and regress one next pose.

Expected limitation:

```text
mode averaging
invalid in-between predictions
poor multimodal behavior
```

---

### 15.4 Small Transformer with Single Gaussian Head

This tests whether attention helps but without multimodal output.

Expected limitation:

```text
better context understanding than MLP
still weak under multimodal next-pose distributions
```

---

### 15.5 Small Transformer with MDN Head

Main proposed model.

Expected advantage:

```text
captures multiple plausible next-pose modes
works with candidate sampling and hard validation
```

---

## 16. Implementation Milestones

### Milestone 1: Dataset Processor

Deliverables:

```text
load original valid sequences
convert sequences to next-step training pairs
canonicalize equivalent poses
encode r as sin_r and cos_r
assign layer_id
assign time_index
sort history tokens
split by original sequence
```

Expected output:

```text
train_samples.pkl
val_samples.pkl
test_samples.pkl
```

---

### Milestone 2: Augmentation Module

Deliverables:

```text
random SE(2) transformation
apply transform to history and target together
verify geometry is preserved
visualize augmented samples
```

Validation:

```text
plot original and augmented sequences
check that relative brick layout is unchanged
check that target remains consistent with transformed history
```

---

### Milestone 3: Normalization Module

Deliverables:

```text
local-frame normalization
inverse transform recovery
training-set normalization statistics
world-to-local conversion
local-to-world conversion
```

Test cases:

```text
normalize then inverse-transform should recover original poses
rotation should recover correct yaw
translation should recover correct x/y
```

---

### Milestone 4: Baseline Transformer Encoder

Deliverables:

```text
token embedding MLP
small transformer encoder
CLS token or mean pooling
binary b head
single Gaussian continuous pose head
training loop
validation loop
```

Purpose:

```text
establish a simple working model before adding MDN complexity
```

---

### Milestone 5: MDN Prediction Head

Deliverables:

```text
mixture weights output
mixture means output
mixture variances output
MDN negative log likelihood loss
stable log-sum-exp implementation
candidate sampling function
```

Initial setting:

```text
K = 5 mixture components
```

---

### Milestone 6: Candidate Visualization

Deliverables:

```text
visualize current partial structure
visualize ground-truth next pose
visualize sampled model candidates
color valid vs invalid candidates
```

This is critical for debugging.

Useful plots:

```text
top-down XY plot
3D brick pose plot
layer-colored plot
candidate density plot
```

---

### Milestone 7: Validator Integration

Deliverables:

```text
sample K candidate next poses
convert local predictions to world poses
run collision validation
run stability validation
run robot reachability validation
run robot path validation
rank valid candidates
```

Initial inference:

```text
sample K = 100 candidates
choose highest-likelihood valid candidate
```

---

### Milestone 8: Rollout Testing

Deliverables:

```text
start from empty scene
predict next brick
validate candidate
append accepted brick
repeat until failure or target length
```

Report:

```text
completed bricks before failure
number of rejected candidates per step
failure reason
generated structure visualization
```

---

## 17. Recommended First Experiment

### 17.1 Input Features

Use:

```text
[x, y, z, b, sin_r, cos_r, layer_id, time_index]
```

Normalize:

```text
x, y, z
```

Keep:

```text
b as 0/1
sin_r, cos_r in [-1, 1]
layer_id normalized or embedded
time_index normalized or embedded
```

---

### 17.2 Model

Use:

```text
2-layer transformer encoder
hidden_dim = 128
num_heads = 4
dropout = 0.1
feedforward_dim = 256
CLS token pooling
```

Prediction head:

```text
binary classifier for b
MDN with K = 5 components for [x, y, z, sin_r, cos_r]
```

---

### 17.3 Training

Use:

```text
optimizer = AdamW
learning_rate = 3e-4
batch_size = 64
epochs = 150
weight_decay = 1e-4
gradient_clipping = 1.0
early_stopping_patience = 20
```

---

### 17.4 Inference

Use:

```text
sample 100 candidates
run hard validator
choose highest-likelihood valid candidate
```

If failure:

```text
sample 500 candidates
if still failure, fall back to rule-based generator
```

---

## 18. Expected Outcomes

### 18.1 What This Model Should Learn

The model should learn:

```text
layer-wise placement tendencies
common spatial relationships between neighboring bricks
where new bricks are likely to appear
whether the next brick is likely laying or standing
how yaw relates to the current layer pattern
```

---

### 18.2 What This Model May Not Fully Learn

With only 50 original sequences, the model may not fully learn:

```text
rare structural configurations
long-horizon stability constraints
robot motion-planning constraints
unseen topology changes
complex global design intent
```

Therefore, hard validation remains necessary.

---

## 19. Risks and Mitigations

### Risk 1: Overfitting to 50 Original Sequences

Mitigation:

```text
small model
dropout
weight decay
early stopping
sequence-level validation split
rigid SE(2) augmentation
local-frame normalization
```

---

### Risk 2: Mode Collapse

The model may predict only the most common next-pose mode.

Mitigation:

```text
use MDN instead of single regression
sample multiple candidates
increase mixture components from K=5 to K=10 if needed
monitor candidate diversity
```

---

### Risk 3: Invalid Sampled Candidates

The model may produce plausible-looking but invalid poses.

Mitigation:

```text
hard validation loop
candidate rejection
fallback to rule-based generator
optional future feasibility classifier
```

---

### Risk 4: Confusion from Symmetric Pose Representations

Mitigation:

```text
deterministic canonicalization
consistent runtime conversion
optional symmetry-aware loss
```

---

### Risk 5: Validation/Test Leakage Through Augmentation

Mitigation:

```text
split by original sequence before augmentation
augment only training sequences
evaluate on held-out original sequences
```

---

## 20. Future Extension: Feasibility Classifier

After the MDN generator works, add a learned feasibility scorer.

Input:

```text
partial structure + candidate next pose
```

Output:

```text
valid / invalid
```

Training data:

```text
positive examples: valid next poses from generated sequences
negative examples: perturbed invalid poses
```

Negative examples can be generated by:

```text
small XY shifts causing collision
large XY shifts causing unsupported placement
wrong Z values causing intersection
wrong yaw values violating layer pattern
poses outside robot reach
poses that cause path collision
```

Inference pipeline becomes:

```text
MDN generator proposes candidates
feasibility classifier ranks candidates
hard validator confirms candidates
```

This can improve runtime efficiency by reducing the number of hard validation calls.

---

## 21. Future Extension: Conditional Diffusion

If the MDN model cannot represent enough multimodality, replace the MDN head with a conditional diffusion model over next pose.

However, do not start with diffusion.

Reason:

```text
the original dataset is small
diffusion is more complex to train
MDN is easier to debug
MDN integrates naturally with candidate sampling
```

Diffusion becomes more attractive if:

```text
more sequences are generated
the next-pose distribution is highly multimodal
MDN candidates lack diversity
rollout performance saturates
```

---

## 22. Recommended Repository Structure

```text
brick_pose_prediction/
│
├── data/
│   ├── raw_sequences/
│   ├── processed/
│   ├── splits/
│   └── augmentation_cache/
│
├── configs/
│   ├── dataset.yaml
│   ├── model_transformer_mdn.yaml
│   ├── training.yaml
│   └── inference.yaml
│
├── src/
│   ├── data/
│   │   ├── sequence_loader.py
│   │   ├── sequence_to_pairs.py
│   │   ├── pose_canonicalization.py
│   │   ├── pose_encoding.py
│   │   ├── augmentation.py
│   │   ├── normalization.py
│   │   └── batching.py
│   │
│   ├── models/
│   │   ├── token_embedding.py
│   │   ├── transformer_encoder.py
│   │   ├── mdn_head.py
│   │   └── next_pose_model.py
│   │
│   ├── losses/
│   │   ├── mdn_loss.py
│   │   └── pose_losses.py
│   │
│   ├── train/
│   │   ├── train.py
│   │   ├── validate.py
│   │   └── callbacks.py
│   │
│   ├── inference/
│   │   ├── sample_candidates.py
│   │   ├── candidate_ranking.py
│   │   └── rollout.py
│   │
│   ├── validation/
│   │   ├── collision_check.py
│   │   ├── stability_check.py
│   │   ├── reachability_check.py
│   │   └── motion_check.py
│   │
│   └── visualization/
│       ├── plot_sequence.py
│       ├── plot_candidates.py
│       └── plot_rollout.py
│
├── scripts/
│   ├── preprocess_sequences.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── run_rollout.py
│   └── visualize_predictions.py
│
├── notebooks/
│   ├── inspect_dataset.ipynb
│   ├── debug_augmentation.ipynb
│   ├── debug_model_predictions.ipynb
│   └── rollout_analysis.ipynb
│
└── README.md
```

---

## 23. Final Recommended Starting Point

Start with the following concrete setup:

```text
dataset:
    50 original sequences
    split = 40 train / 5 val / 5 test
    augmentation = 50 SE(2) variants for training only

input:
    [x, y, z, b, sin_r, cos_r, layer_id, time_index]

normalization:
    local centroid frame
    training-set xyz normalization

model:
    2-layer transformer encoder
    hidden_dim = 128
    num_heads = 4
    dropout = 0.1
    CLS token pooling

output:
    binary classifier for b
    MDN with K = 5 for [x, y, z, sin_r, cos_r]

training:
    AdamW
    learning_rate = 3e-4
    batch_size = 64
    epochs = 150
    early_stopping = 20 epochs
    gradient_clipping = 1.0

inference:
    sample 100 candidate poses
    run hard validator
    choose highest-likelihood valid candidate
    fall back to rule-based generator if no candidate passes
```

The main success criterion should not be low MSE. The main success criterion should be:

```text
How often can the model propose at least one valid next brick pose within the top K samples?
```

and, more importantly:

```text
How many bricks can the model place successfully during closed-loop rollout before failure?
```