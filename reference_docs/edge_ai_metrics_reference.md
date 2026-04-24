# Edge AI Model Performance Reference Guide
### (For Models from Millions → Billions of Parameters)

This document provides **practical performance ranges** observed across:
- Hugging Face benchmarks
- llama.cpp / GGUF deployments
- Edge AI research (TinyML, MLPerf Tiny)
- Industry deployments (CPU + GPU inference)

⚠️ NOTE:
These are **guidelines, not strict rules**. Actual performance depends on:
- Hardware (CPU vs GPU vs NPU)
- Quantization level (FP16, Q8, Q6, Q4)
- Model architecture (transformer vs CNN vs hybrid)

---

# 1. MODEL SIZE CATEGORIES

| Category | Parameters |
|---------|-----------|
| Tiny | < 100M |
| Small | 100M – 1B |
| Medium | 1B – 7B |
| Large | 7B – 13B |
| Very Large | > 13B |

---

# 2. SPEED (TOKENS / SECOND)

### CPU (typical edge/server CPUs)

| Performance | Tokens/sec |
|------------|-----------|
| Very Slow | < 5 |
| Slow | 5 – 10 |
| Moderate | 10 – 20 |
| Fast | 20 – 40 |
| Very Fast | > 40 |

### GPU (consumer / datacenter)

| Performance | Tokens/sec |
|------------|-----------|
| Slow | < 20 |
| Moderate | 20 – 50 |
| Fast | 50 – 150 |
| Very Fast | > 150 |

📌 Notes:
- <10 tokens/sec → noticeable lag
- 10–20 → acceptable for many tasks
- >20 → smooth interactive experience

---

# 3. MEMORY USAGE (RAM / VRAM)

### General Guidelines

| Usage Level | RAM |
|------------|-----|
| Low | < 4 GB |
| Moderate | 4 – 8 GB |
| High | 8 – 16 GB |
| Very High | > 16 GB |

📌 Notes:
- GGUF Q4 models often reduce memory by ~60–75%
- Memory >10GB on CPU → strong candidate for quantization/pruning

---

# 4. CPU UTILIZATION

| Level | CPU % |
|------|------|
| Low | < 50% |
| Moderate | 50 – 75% |
| High | 75 – 90% |
| Saturated | > 90% |

📌 Interpretation:
- >95% → compute-bound system
- Combined with low speed → model too heavy

---

# 5. LATENCY

## Time to First Token (TTFT)

| Level | Time |
|------|------|
| Excellent | < 0.2s |
| Good | 0.2 – 0.5s |
| Moderate | 0.5 – 1s |
| High | > 1s |

## Load Time

| Level | Time |
|------|------|
| Fast | < 5s |
| Moderate | 5 – 15s |
| Slow | > 15s |

---

# 6. ACCURACY PROXIES (FOR COMPRESSION)

Since exact accuracy is task-dependent, proxies are used:

## KL Divergence (bits)

| Level | KL |
|------|----|
| Safe | < 0.05 |
| Acceptable | 0.05 – 0.1 |
| Risky | 0.1 – 0.2 |
| Harmful | > 0.2 |

## JS Distance

| Level | JS |
|------|----|
| Safe | < 0.1 |
| Moderate change | 0.1 – 0.2 |
| Significant change | > 0.2 |

---

# 7. INTERPRETATION RULES (IMPORTANT)

These are NOT strict thresholds but **patterns**:

### Rule 1 — Compute-bound system
- CPU > 90% AND tokens/sec < 10  
→ Model too large → reduce layers or quantize

---

### Rule 2 — Memory pressure
- RAM > 8–10 GB (CPU)
→ Strong candidate for quantization/pruning

---

### Rule 3 — Latency bottleneck
- TTFT > 0.5s
→ Optimize initialization or reduce model size

---

### Rule 4 — Safe compression zone
- JS < 0.1 OR KL < 0.05  
→ Layer can be aggressively compressed

---

### Rule 5 — Dangerous compression
- JS > 0.2 OR KL > 0.1  
→ Layer should be preserved

---

# 8. MODEL-SIZE-SPECIFIC EXPECTATIONS

## Small Models (<1B)
- Expected: 20–100 tokens/sec (CPU/GPU mix)
- Low memory footprint
- Aggressive quantization usually safe

---

## Medium Models (1B–7B)
- Expected CPU: 5–20 tokens/sec
- Memory: 6–16 GB
- Requires selective compression

---

## Large Models (7B–13B)
- Expected CPU: 2–10 tokens/sec
- Heavy memory usage
- Needs quantization + layer selection

---

# 9. RECOMMENDED ACTIONS (GUIDANCE FOR AGENT)

| Condition | Suggested Action |
|----------|----------------|
| Low speed | Quantization |
| High memory | Sparsity / pruning |
| High CPU | Reduce layers |
| High KL/JS | Avoid compression |
| Balanced metrics | Minor tuning |

---

# 10. SOURCES (REFERENCE BASIS)

This document is derived from:

- Hugging Face model benchmarks
- llama.cpp / GGUF performance reports
- MLPerf Tiny benchmarks
- Academic pruning & quantization research
- Real-world edge deployments

It is a **synthesized engineering reference**, not a single-source standard.

---