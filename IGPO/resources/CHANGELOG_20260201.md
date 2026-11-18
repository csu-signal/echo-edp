# 📝 Changelog

All notable changes to this project will be documented in this file.

---

## [Feb 01, 2026]

### 🔧 Bug Fixes
- Introduced a simplified `tools_server` interface, resolving the previous issue where the `tools_server` module could not be publicly released due to company policies

### ✨ New Features

#### Vectorized Ground Truth LogP Computation
Accelerates the construction of info gain rewards. Enable via:
```bash
+algorithm.use_vectorized_gt_logprob=true
```

#### Flexible Info Gain Reward Calculation
Two computation methods available:

| Option | Formula | Config |
|--------|---------|--------|
| Log Probability Difference | $\bar{\log P_t} - \bar{\log P_{t-1}}$ | `+algorithm.info_gain_type=log_prob_diff` |
| Probability Difference | $e^{\bar{\log P_t}} - e^{\bar{\log P_{t-1}}}$ | `+algorithm.info_gain_type=prob_diff` |

#### Reward Normalization Modes
Two normalization strategies supported:

| Mode | Description | Config |
|------|-------------|--------|
| Joint | Normalizes all rewards together | `+algorithm.info_gain_norm_mode=joint` |
| Separate | Normalizes info gain reward and outcome reward independently | `+algorithm.info_gain_norm_mode=separate` |

#### Curriculum Learning
Gradually decays the weight of info gain reward as training progresses. Configure in `train.sh`:
```bash
+algorithm.use_curriculum=false \
+algorithm.curriculum_f1_init=0.5 \
+algorithm.curriculum_f1_final=1.0 \
+algorithm.curriculum_ig_init=1.0 \
+algorithm.curriculum_ig_final=0.5 \
```
---

> 🚀 **Stay tuned for more features coming soon!**
