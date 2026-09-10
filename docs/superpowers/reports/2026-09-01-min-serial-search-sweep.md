# Min serial-search sweep — 2026-09-01

## full

| arm | completion | aborts max/deadline/other | moves p50/p90/p99 | revisit | repeat<=8 | max revisits | cycling | samples/h | eval/s | batch mean/p50/p90 | target H | norm H | full/legal KL | legal mass | top1 | boosted | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a0-128 | 0.0% | 256/0/0 | 800/800/800 | 9.75343e-06 | 9.75343e-06 | 1.00781 | 0 | 0 | 84995.1 | 127.877/128/128 | 3.86003 | 0.973838 | 2.2008/0.104071 | 0.125646 | 0.026123 | 0 | 311.382 |
| a0-256 | 0.0% | 0/256/0 | 464/464/465 | 0 | 0 | 0 | 0 | 0 | 84905.2 | 127.805/128/128 | 3.86732 | 0.978161 | 2.19828/0.0864906 | 0.124227 | 0.0214844 | 0 | 360.363 |
| a0-400 | 0.0% | 0/256/0 | 297/298/298 | 0 | 0 | 0 | 0 | 0 | 84984 | 127.767/128/128 | 3.85551 | 0.979418 | 2.20889/0.0808324 | 0.122423 | 0.0209961 | 0 | 360.224 |
| a0-adaptive-256 | 0.0% | 256/0/0 | 800/800/800 | 9.75343e-06 | 9.75343e-06 | 1.00781 | 0 | 0 | 85429.5 | 127.250/128/128 | 3.86003 | 0.973838 | 2.2008/0.104071 | 0.125646 | 0.026123 | 9.76563e-06 | 309.823 |
| a0-adaptive-400 | 0.0% | 256/0/0 | 800/800/800 | 9.75343e-06 | 9.75343e-06 | 1.00781 | 0 | 0 | 84782.8 | 127.773/128/128 | 3.86003 | 0.973838 | 2.2008/0.104071 | 0.125646 | 0.026123 | 9.76563e-06 | 312.214 |
| b0-128 | 100.0% | 0/0/0 | 118/148/221 | 0.00724393 | 0.00658207 | 1.50781 | 0 | 1.34907e+06 | 47775.2 | 58.232/76/115 | 2.99628 | 0.787677 | 3.06168/0.821748 | 0.11216 | 0.0285645 | 0 | 83.436 |
| b0-256 | 100.0% | 0/0/0 | 116/154/199 | 0.00950787 | 0.00853413 | 1.61719 | 0 | 830764 | 58513.8 | 71.746/88/117 | 3.06585 | 0.806843 | 2.99712/0.737696 | 0.110582 | 0.0197754 | 0 | 136.960 |
| b0-400 | 100.0% | 0/0/0 | 120/169/223 | 0.0137728 | 0.0127786 | 1.69141 | 0 | 541505 | 59422.5 | 73.198/88/117 | 3.08179 | 0.81455 | 2.98594/0.712417 | 0.109663 | 0.019043 | 0 | 217.188 |

### Comparisons

- `a0-128` vs `b0-128`: completion -100.0% (denominator=256); wall +273.2% (denominator=83.4359).
- `a0-256` vs `b0-128`: completion -100.0% (denominator=256); wall +331.9% (denominator=83.4359).
- `a0-400` vs `b0-128`: completion -100.0% (denominator=256); wall +331.7% (denominator=83.4359).
- `a0-adaptive-256` vs `b0-128`: completion -100.0% (denominator=256); wall +271.3% (denominator=83.4359).
- `a0-adaptive-400` vs `b0-128`: completion -100.0% (denominator=256); wall +274.2% (denominator=83.4359).
- `b0-256` vs `b0-128`: completion +0.0% (denominator=256); wall +64.1% (denominator=83.4359).
- `b0-400` vs `b0-128`: completion +0.0% (denominator=256); wall +160.3% (denominator=83.4359).
- `a0-256` vs `a0-128`: completion n/a (baseline=0); repeat<=8 -100.0% (denominator=9.75343e-06); wall +15.7% (denominator=311.382).
- `a0-400` vs `a0-128`: completion n/a (baseline=0); repeat<=8 -100.0% (denominator=9.75343e-06); wall +15.7% (denominator=311.382).
- `a0-adaptive-256` vs `a0-128`: completion n/a (baseline=0); repeat<=8 +0.0% (denominator=9.75343e-06); wall -0.5% (denominator=311.382).
- `a0-adaptive-400` vs `a0-128`: completion n/a (baseline=0); repeat<=8 +0.0% (denominator=9.75343e-06); wall +0.3% (denominator=311.382).

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

`NO_DEEPER_SEARCH_BENEFIT`

Smoke qualifiers (literal gate, including zero baselines): `a0-128, a0-256, a0-400, a0-adaptive-256, a0-adaptive-400`.
Full matrix complete: `true`.
Illegal-mass-dominated arms: `a0-256, a0-400`.
Deeper search helps: `false`. Adaptive search wins: `false`.
Selected ordered branch: `D_AUTHORIZE_VACANCY_PRIOR_ANNEALING`.
Min remains B0 until the final acceptance gate passes.
The summarizer reports evidence only and does not alter code or configuration.
