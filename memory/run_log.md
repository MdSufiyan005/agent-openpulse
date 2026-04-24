
### Run: 2026-04-23 19:07
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
The function call to `merge_analysis_reports` with the provided arguments will generate a unified compression report based on the KL divergence analysis and other optional reports. The output will include the final quantization assignments, the corrected llama-quantize command string, flags for special handling, and a summary of the strategy.

---
### Run: 2026-04-23 19:15
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-23 19:32
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-23 19:40
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-23 19:51
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-23 19:57
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-23 20:00
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-23 20:05
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-23 20:12
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-23 20:15
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-24 05:20
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-24 05:34
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-24 05:37
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-24 05:37
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-24 05:52
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-24 06:00
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-24 06:03
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-24 06:04
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-24 06:07
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
KL divergence analysis is complete. Here is the summary for your review:

GLOBAL STATS:
  Total decoder layers scored: 168
  COMPRESS (≤Q5_K_M): 152
  KEEP (≥Q6_K): 16
  Quant distribution: {'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
  noise_scale used: 0.05

TOP 20 MOST SENSITIVE LAYERS (by KL bits):
  model.text_model.layers.7.mlp.down_proj: KL=0.025368b → ?
  model.text_model.layers.4.self_attn.v_proj: KL=0.018350b → ?
  model.text_model.layers.1.mlp.down_proj: KL=0.018117b → ?
  model.text_model.layers.0.mlp.down_proj: KL=0.007947b → ?
  model.text_model.layers.23.mlp.down_proj: KL=0.007022b...

---
### Run: 2026-04-24 14:58
**Layers:** 168
**Findings:**
- KL: safe=168 sensitive=0 noise=0.05 top3=['model.text_model.layers.7.mlp.down_proj=0.0254b', 'model.text_model.layers.4.self_attn.v_proj=0.0183b', 'model.text_model.layers.1.mlp.down_proj=0.0181b']
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
{"method": "kl_conductance_sparsity_merge", "signals_used": {"kl": true, "conductance": false, "sparsity": false}, "kl_report": "results/kl_divergence_report.json", "conductance_report": null, "sparsity_report": null, "total_layers": 168, "compress_count": 152, "keep_count": 16, "compress_layers": ["model.text_model.layers.2.self_attn.q_proj", "model.text_model.layers.2.self_attn.k_proj", "model.text_model.layers.2.self_attn.v_proj", "model.text_model.layers.2.self_attn.o_proj", "model.text_model.layers.2.mlp.gate_proj", "model.text_model.layers.2.mlp.up_proj", "model.text_model.layers.2.mlp.d...

---
### Run: 2026-04-24 19:50
**Layers:** 168
**Findings:**
- KL: safe=0 sensitive=0 noise=0.1 top3=[]
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
{  "final_quant_assignments": {
    "model.model.text_model.layers.0.self_attn.q_proj": "Q6_K",
    "model.model.text_model.layers.0.self_attn.k_proj": "Q6_K",
    "model.model.text_model.layers.0.self_attn.v_proj": "Q6_K",
    "model.model.text_model.layers.29.mlp.gate_proj": "Q6_K",
    "model.model.text_model.layers.29.mlp.up_proj": "Q6_K",
    "model.model.text_model.layers.29.mlp.down_proj": "Q6_K",
    "model.model.text_model.layers.15.mlp.down_proj": "Q4_K_M",
    "model.model.text_model.layers.12.mlp.gate_proj": "Q4_K_M",
    "model.model.text_model.layers.8.self_attn.v_proj": "Q4_K_M"...

---
### Run: 2026-04-24 21:05
**Layers:** 168
**Findings:**
- KL: safe=0 sensitive=0 noise=0.1 top3=[]
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
```json
{
  "final_quant_assignments": {
    "model.model.text_model.layers.0.self_attn.q_proj": "Q6_K",
    "model.model.text_model.layers.0.self_attn.k_proj": "Q6_K",
    "model.model.text_model.layers.0.self_attn.v_proj": "Q6_K",
    "model.model.text_model.layers.0.self_attn.out_proj": "Q6_K",
    "model.model.text_model.layers.0.mlp.gate_proj": "Q6_K",
    "model.model.text_model.layers.0.mlp.up_proj": "Q6_K",
    "model.model.text_model.layers.0.mlp.down_proj": "Q6_K",
    "model.model.text_model.layers.0.input_layernorm.weight": "Q6_K",
    "model.model.text_model.layers.0.input_layerno...

---
### Run: 2026-04-24 21:07
**Layers:** 168
**Findings:**
- KL: safe=0 sensitive=0 noise=0.1 top3=[]
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
```json
{
  "final_quant_assignments": {
    "model.model.text_model.layers.0.self_attn.q_proj": "Q6_K",
    "model.model.text_model.layers.0.self_attn.k_proj": "Q6_K",
    "model.model.text_model.layers.0.self_attn.v_proj": "Q6_K",
    "model.model.text_model.layers.0.self_attn.out_proj": "Q6_K",
    "model.model.text_model.layers.0.mlp.gate_proj": "Q6_K",
    "model.model.text_model.layers.0.mlp.down_proj": "Q6_K",
    "model.model.text_model.layers.0.mlp.up_proj": "Q6_K",

    "model.model.text_model.layers.1.self_attn.q_proj": "Q6_K",
    "model.model.text_model.layers.1.self_attn.k_proj":...

---
### Run: 2026-04-24 21:09
**Layers:** 168
**Findings:**
- KL: safe=0 sensitive=0 noise=0.1 top3=[]
- MERGED: compress=152 keep=16 dist={'Q6_K': 16, 'Q5_K_M': 25, 'Q4_K_M': 127}
**Agent:**
```json
{
  "final_quant_assignments": {
    "model.model.text_model.layers.0": "Q6_K",
    "model.model.text_model.layers.1": "Q6_K",
    "model.model.text_model.layers.2": "Q6_K",
    "model.model.text_model.layers.3": "Q4_K_M",
    "model.model.text_model.layers.4": "Q4_K_M",
    "model.model.text_model.layers.5": "Q6_K",
    "model.model.text_model.layers.6": "Q4_K_M",
    "model.model.text_model.layers.7": "Q4_K_M",
    "model.model.text_model.layers.8": "Q4_K_M",
    "model.model.text_model.layers.9": "Q4_K_M",
    "model.model.text_model.layers.10": "Q6_K",
    "model.model.text_model.l...

---