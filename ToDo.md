# Architecture Improvement ToDo

## 1. Attention-Based Feature Fusion

**Problem**: Current DualPathwayModel uses simple concatenation for fusion, giving equal weight to CNN and radiomics features regardless of input. Adenocarcinoma-squamous confusion (59% of errors) requires dynamic feature weighting.

**Implementation Tasks**:
- [ ] Create `AttentionFusion` module in `src/ct_scan_mlops/model.py`
  - [ ] Project radiomics features to CNN dimension (128 -> 512)
  - [ ] Implement cross-attention (radiomics queries, CNN keys/values)
  - [ ] Add gating mechanism for adaptive fusion
  - [ ] Include LayerNorm for stability
- [ ] Add configuration options to `configs/model/dual_pathway.yaml`
  - [ ] `fusion_type`: "attention" or "concat"
  - [ ] `attention_heads`: number of attention heads (default: 4)
  - [ ] `attention_dropout`: dropout rate (default: 0.1)
- [ ] Update `DualPathwayModel` to support both fusion types
- [ ] Add unit tests for attention fusion module
- [ ] Run baseline comparison against concatenation fusion

**Expected Impact**: Better adenocarcinoma-squamous discrimination through dynamic feature emphasis based on input characteristics.

---

## 2. Hierarchical Classification

**Problem**: 4-way classification treats all confusions equally, but normal class has only 1.85% error rate while cancer subtypes have 10-14% error rates. A hierarchical approach can isolate the hard problem.

**Implementation Tasks**:
- [ ] Create `HierarchicalClassifier` module in `src/ct_scan_mlops/model.py`
  - [ ] Stage 1 head: Binary classification (cancer vs normal)
  - [ ] Stage 2 head: 3-way cancer subtype classification
  - [ ] Implement probability combination for 4-class output
  - [ ] Add `compute_hierarchical_loss()` method with configurable stage weights
- [ ] Add configuration options to `configs/model/`
  - [ ] Create `hierarchical.yaml` config
  - [ ] `stage1_weight`: weight for binary loss (default: 0.3)
  - [ ] `dropout`: classifier dropout rate
- [ ] Update training loop in `src/ct_scan_mlops/train.py`
  - [ ] Detect hierarchical classifier and use appropriate loss
- [ ] Add unit tests for hierarchical classifier
- [ ] Run experiments comparing flat vs hierarchical classification

**Expected Impact**: Isolate the hard cancer subtyping problem from easy cancer detection; mirrors clinical decision-making workflow.

---

## 3. Training Stability (Batch Normalization)

**Problem**: Training loss steps are oscillating significantly. The `DualPathwayModel` currently lacks normalization in its projection and fusion pathways, making training sensitive to initialization and learning rates.

**Implementation Tasks**:
- [x] Add `BatchNorm1d` to `DualPathwayModel` pathways in `src/ct_scan_mlops/model.py`
  - [x] Update `cnn_projection` to include `BatchNorm1d(cnn_feature_dim)`
  - [x] Update `radiomics_projection` to include `BatchNorm1d(radiomics_hidden)` for both linear layers
- [x] Verify model forward pass with batch size > 1
- [ ] Monitor training loss curves for smoother convergence in the next sweep

**Expected Impact**: Reduced oscillation in training loss, faster convergence, and potentially higher final accuracy due to better gradient flow.

---

## Priority Order

1. **Training Stability** - Immediate fix for training dynamics
2. **Attention-Based Fusion** - Lower complexity, direct improvement to existing architecture
3. **Hierarchical Classification** - Requires more changes but addresses root cause of confusion

## Notes

- Both improvements are based on error analysis showing adenocarcinoma-squamous confusion as primary challenge
- Shape features (sphericity, major_axis_length, perimeter) are most discriminative
- Consider combining both approaches for maximum impact
