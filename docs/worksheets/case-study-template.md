# Case Study Worksheet

A fillable companion to **Appendix D.4 — Writing Your Own Case Study**. Copy this file
per engagement and complete it as you work. Anonymize freely ("a mid-size inference
team"); reviewers care about the method and the numbers, not the logo.

> Honesty rules (Chapter 14): change one thing at a time so each delta is attributable;
> report steady-state, not peak; state the hardware, model, and load; quote the speedup
> against a fair baseline. A modest, honest 1.9× beats an unreproducible 10×.

## Beat 1 — Situation
_One paragraph: the symptom exactly as the team first saw it, the wrong first conclusion,
and the SLA or budget at stake._

## Beat 2 — Baseline
_Capture before you touch anything._

| Metric | Baseline | Target / SLA |
|---|---|---|
| GPU utilization (%) | | |
| TTFT p50 / p99 (ms) | | |
| TPS (tokens/s) | | |
| Throughput (req/s) | | |
| Concurrency (in-flight req) | | |
| Cost per 1M tokens ($) | | |
| Monthly spend ($) | | |

## Beat 3 — Investigation
_Which tools, what each revealed, the single root cause and its evidence._

- nvidia-smi / DCGM (utilization, power):
- nsys (timeline / overlap):
- ncu (kernel counters / roofline):
- torch.profiler (operator attribution):
- flamegraph (CPU / data path):
- **Root cause:**

## Beat 4 — Fix
_Each change tied to the chapter that explains it._

- [ ] change — (chapter)
- [ ] change — (chapter)

## Beat 5 — Outcome and lesson

| Metric | Before | After | Change |
|---|---|---|---|
| GPU utilization (%) | | | |
| TTFT p99 (ms) | | | |
| TPS (tokens/s) | | | |
| Throughput (req/s) | | | |
| Cost per 1M tokens ($) | | | |

**Lesson (1–2 sentences, transferable):**
