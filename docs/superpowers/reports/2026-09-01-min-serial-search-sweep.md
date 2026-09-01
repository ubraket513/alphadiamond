# Min serial-search sweep — 2026-09-01

## full

| arm | completion | aborts max/deadline/other | moves p50/p90/p99 | revisit | repeat<=8 | max revisits | cycling | samples/h | eval/s | batch mean/p50/p90 | target H | norm H | full/legal KL | legal mass | top1 | boosted | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| b0-128 | 100.0% | 0/0/0 | 118/148/221 | 0.00724393 | 0.00658207 | 1.50781 | 0 | 1.34907e+06 | 47775.2 | 58.232/76/115 | 2.99628 | 0.787677 | 3.06168/0.821748 | 0.11216 | 0.0285645 | 0 | 83.436 |

### Comparisons


## smoke

| arm | completion | aborts max/deadline/other | moves p50/p90/p99 | revisit | repeat<=8 | max revisits | cycling | samples/h | eval/s | batch mean/p50/p90 | target H | norm H | full/legal KL | legal mass | top1 | boosted | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a0-128 | 0.0% | 32/0/0 | 800/800/800 | 0 | 0 | 1 | 0 | 0 | 14143.8 | 16.000/16/16 | 3.85109 | 0.973839 | 2.21024/0.103219 | 0.124412 | 0.0263672 | 0 | 233.549 |
| a0-256 | 0.0% | 0/32/0 | 618/623/623 | 0 | 0 | 0 | 0 | 0 | 14188.7 | 16.000/16/16 | 3.88317 | 0.978062 | 2.181/0.0871268 | 0.126187 | 0.0178223 | 0 | 360.049 |
| a0-400 | 0.0% | 0/32/0 | 387/390/390 | 0 | 0 | 0 | 0 | 0 | 13864.8 | 16.000/16/16 | 3.86337 | 0.979398 | 2.19659/0.0814674 | 0.123776 | 0.0209961 | 0 | 360.039 |
| a0-adaptive-256 | 0.0% | 32/0/0 | 800/800/800 | 0 | 0 | 1 | 0 | 0 | 14299.7 | 15.999/16/16 | 3.85109 | 0.973839 | 2.21024/0.103219 | 0.124412 | 0.0263672 | 0 | 231.002 |
| a0-adaptive-400 | 0.0% | 32/0/0 | 800/800/800 | 0 | 0 | 1 | 0 | 0 | 14171.2 | 15.999/16/16 | 3.85109 | 0.973839 | 2.21024/0.103219 | 0.124412 | 0.0263672 | 0 | 233.095 |
| b0-128 | 100.0% | 0/0/0 | 118/140/160 | 0.00398089 | 0.00398089 | 1.25 | 0 | 353404 | 12515.7 | 14.090/16/16 | 3.00436 | 0.786542 | 3.05943/0.831151 | 0.113314 | 0.0275489 | 0 | 39.565 |
| b0-256 | 100.0% | 0/0/0 | 115/142/150 | 0.00812047 | 0.00812047 | 1.5 | 0 | 161874 | 11386.1 | 13.265/16/16 | 3.04857 | 0.800473 | 3.01522/0.771761 | 0.112151 | 0.0149373 | 0 | 83.376 |
| b0-400 | 100.0% | 0/0/0 | 121/149/218 | 0.0122634 | 0.0111705 | 1.5625 | 0 | 91268.2 | 10009.5 | 11.576/16/16 | 3.0708 | 0.811083 | 2.99308/0.723746 | 0.109952 | 0.0205078 | 0 | 161.642 |

### Comparisons

- `a0-128` vs `b0-128`: completion -100.0% (denominator=32); wall +490.3% (denominator=39.565).
- `a0-256` vs `b0-128`: completion -100.0% (denominator=32); wall +810.0% (denominator=39.565).
- `a0-400` vs `b0-128`: completion -100.0% (denominator=32); wall +810.0% (denominator=39.565).
- `a0-adaptive-256` vs `b0-128`: completion -100.0% (denominator=32); wall +483.9% (denominator=39.565).
- `a0-adaptive-400` vs `b0-128`: completion -100.0% (denominator=32); wall +489.1% (denominator=39.565).
- `b0-256` vs `b0-128`: completion +0.0% (denominator=32); wall +110.7% (denominator=39.565).
- `b0-400` vs `b0-128`: completion +0.0% (denominator=32); wall +308.5% (denominator=39.565).
- `a0-256` vs `a0-128`: completion n/a (baseline=0); repeat<=8 n/a (baseline=0); wall +54.2% (denominator=233.549).
- `a0-400` vs `a0-128`: completion n/a (baseline=0); repeat<=8 n/a (baseline=0); wall +54.2% (denominator=233.549).
- `a0-adaptive-256` vs `a0-128`: completion n/a (baseline=0); repeat<=8 n/a (baseline=0); wall -1.1% (denominator=233.549).
- `a0-adaptive-400` vs `a0-128`: completion n/a (baseline=0); repeat<=8 n/a (baseline=0); wall -0.2% (denominator=233.549).

## Decision classification

`FULL_SWEEP_REQUIRED`

Smoke qualifiers (literal gate, including zero baselines): `a0-128, a0-256, a0-400, a0-adaptive-256, a0-adaptive-400`.
Full matrix complete: `false`.
Illegal-mass-dominated arms: `none`.
Deeper search helps: `false`. Adaptive search wins: `false`.
Selected ordered branch: `PENDING_FULL_SWEEP`.
Min remains B0 until the final acceptance gate passes.
The summarizer reports evidence only and does not alter code or configuration.
