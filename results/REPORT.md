# GaLore vs LoRA vs full fine-tuning — results

Source: `runs.jsonl`, 31 runs.

Fairness check passed: within each phase all runs share model, data, seq_len, batch, steps, seed and dtype; across phases they share everything but step count.


## Memory benchmark (short runs, no eval)

| run | method | trainable % | peak GiB | exact? | predicted GiB | eval loss | eval ppl | tok/s | status |
|---|---|---|---|---|---|---|---|---|---|
| mem_full | full | 100.0 | 20.52 | yes | 16.39 | — | — | 2491 | completed |
| mem_galore_r128 | galore | 100.0 | 12.50 | yes | 10.02 | — | — | 1647 | completed |
| mem_galore_r128_layerwise | galore | 100.0 | 9.16 | yes | 6.17 | — | — | 1551 | completed |
# improvement
| mem_galore_r128_layerwise_embed | galore | 100.0 | 7.82 | yes | 5.28 | — | — | 1536 | completed |
| mem_lora_r128 | lora | 8.4 | 6.50 | yes | 5.60 | — | — | 2162 | completed |
| mem_lora_r16 | lora | 1.1 | 4.98 | yes | 4.29 | — | — | 2221 | completed |
# 

## Learning-rate sweep (short runs)

| run | method | trainable % | peak GiB | exact? | predicted GiB | eval loss | eval ppl | tok/s | status |
|---|---|---|---|---|---|---|---|---|---|
| sweep_full_lr1e-05 | full | 100.0 | 20.53 | yes | 16.39 | 1.1549 | 3.17 | 2414 | completed |
| sweep_full_lr2e-05 | full | 100.0 | 20.53 | yes | 16.39 | 1.1869 | 3.28 | 2417 | completed |
| sweep_full_lr5e-05 | full | 100.0 | 20.53 | yes | 16.39 | 1.3023 | 3.68 | 2417 | completed |
| sweep_full_lr5e-06 | full | 100.0 | 20.53 | yes | 16.39 | 1.1454 | 3.14 | 2390 | completed |
| sweep_galore_lr0.0001 | galore | 100.0 | 9.17 | yes | 6.17 | 1.1761 | 3.24 | 2191 | completed |
| sweep_galore_lr1e-05 | galore | 100.0 | 9.17 | yes | 6.17 | 1.1579 | 3.18 | 2205 | completed |
| sweep_galore_lr3e-05 | galore | 100.0 | 9.17 | yes | 6.17 | 1.1522 | 3.17 | 2223 | completed |
| sweep_lora_lr0.0001 | lora | 8.4 | 6.50 | yes | 5.60 | 1.1496 | 3.16 | 2118 | completed |
# | sweep_lora_lr0.0003 | lora | 8.4 | 6.51 | yes | 5.60 | 1.2057 | 3.34 | 2111 | completed |
| sweep_lora_lr0.001 | lora | 8.4 | 6.50 | yes | 5.60 | 7.3211 | 1511.93 | 2113 | completed |
| sweep_lora_lr3e-05 | lora | 8.4 | 6.50 | yes | 5.60 | 1.1484 | 3.15 | 2058 | completed |


# ## Full runs — the headline comparison

| run | method | trainable % | peak GiB | exact? | predicted GiB | eval loss | eval ppl | tok/s | status |
|---|---|---|---|---|---|---|---|---|---|
| final_full | full | 100.0 | 20.55 | yes | 16.39 | 1.1251 | 3.08 | 2380 | completed |
| final_galore | galore | 100.0 | 9.17 | yes | 6.17 | 1.1336 | 3.11 | 2142 | completed |
# | final_lora | lora | 8.4 | 6.50 | yes | 5.60 | 1.1277 | 3.09 | 1802 | completed |
| seed43_full | full | 100.0 | 20.55 | yes | 16.39 | 1.0876 | 2.97 | 2396 | completed |
| seed43_galore | galore | 100.0 | 9.17 | yes | 6.17 | 1.0962 | 2.99 | 2142 | completed |
| seed43_lora | lora | 8.4 | 6.50 | yes | 5.60 | 1.0890 | 2.97 | 1795 | completed |


## Layerwise × gradient accumulation (batch split is the variable)
# improvement

| run | method | trainable % | peak GiB | exact? | predicted GiB | eval loss | eval ppl | tok/s | status |
|---|---|---|---|---|---|---|---|---|---|
| accum_mb1_ga32_baseline | galore | 100.0 | 12.23 | yes | 10.02 | — | — | 1038 | completed |
| accum_mb1_ga32_layerwise | galore | 100.0 | 8.90 | yes | 6.17 | — | — | 1003 | completed |
| accum_mb2_ga16_baseline | galore | 100.0 | 12.50 | yes | 10.02 | — | — | 1405 | completed |
# improvement
| accum_mb2_ga16_layerwise | galore | 100.0 | 9.16 | yes | 6.17 | — | — | 1305 | completed |
| accum_mb4_ga8_baseline | galore | 100.0 | 13.00 | yes | 10.02 | — | — | 1480 | completed |
| accum_mb4_ga8_layerwise | galore | 100.0 | 9.66 | yes | 6.17 | — | — | 1380 | completed |
# | accum_mb8_ga4_baseline | galore | 100.0 | 14.00 | yes | 10.02 | — | — | 1489 | completed |
| accum_mb8_ga4_layerwise | galore | 100.0 | 10.67 | yes | 6.17 | — | — | 1424 | completed |


## Plots
# 
![train_loss.png](train_loss.png)
![eval_loss.png](eval_loss.png)
![memory.png](memory.png)
# Enhanced

# Enhanced

# Optimized
# 
# Refined

# Optimized

# Enhanced

# Enhanced

# Optimized
# improvement

# Optimized

# Optimized

# Refined

# Enhanced

# Optimized
# 
# Optimized

# Enhanced

# Enhanced

# Refined

# Optimized

# Optimized
