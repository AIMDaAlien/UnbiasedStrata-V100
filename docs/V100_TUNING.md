# V100 tuning record (2026-10-18)

Machine: Tesla V100-PCIe-32GB (sm_70, PCIe Gen3 x16, 13.1 GB/s host->device), Ryzen 9 7950X, 64 GB
DDR5-6000, Linux 7.2.5 (Omarchy), NVIDIA 580.178.04, CUDA 12.9 + GCC 14 toolchain for the build.
Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S, context 196,608, INT8 KV with a 32K resident window,
MTP draft 4, 12 expert-pool workers. Production service: `unbiasedstrata.service` on port 8083.

## Constraints

- GPU power limit 100 W at boot (`gpu-nv-power.service`; card default 250 W). The PC lost power four
  times under sustained load (last at a 175 W sustained GPU limit), so all measurement runs go through
  `~/llm-trade-bench/tools/guarded-run.sh`, one heavy job at a time. Short probes up to 150 W were
  approved; 175 W and above were not.
- RAM 64 GB, service capped at 56 GiB; the IQ3_S pinned expert complement is 25.73 GiB plus a 2.32 GiB
  pinned KV copy. IQ3_S and the 196,608 context are kept.

## Harness

`local/v100_tune.py`: paired A/B of launch profiles in alternating order, each candidate in its own
temporary user service (56 GiB, no swap), warm-up request first, then N short coding requests
(~700-token prompts, medium reasoning, 400 generated tokens) and optional 80K/128K/180K fresh
fixtures from `~/Work/v100-20261013/long-context-fixtures`. Per request it records the engine's own
prompt/decode line, wall time, GPU power/SM clock/temperature at 2 Hz and CPU package energy (RAPL),
giving tokens/s and joules per generated token. Candidates may run root-helper commands before and
after (`/usr/local/sbin/v100-tune`: power limit 100..150 W, SM clock lock, CPU EPP, restore).

## Where the time and energy go (baseline, 100 W cap)

| phase | SM clock | GPU | result |
| --- | ---: | ---: | --- |
| decode, short prompts | ~1050 MHz (795..1357) | 98 W | 61-68 tok/s, 72% draft acceptance, 97% cache hits |
| decode, after 80K prompt | ~790 MHz | 100 W | 61 tok/s, 89% cache hits |
| prefill 80K | ~770 MHz | 100 W | 873 tok/s (1,432 tok/s at 250 W on 2026-10-13) |

- The 100 W cap binds in both phases. Prefill is compute-bound and loses ~40% against 250 W; decode
  loses less because it is memory- and PCIe-bound.
- Energy per generated token (short requests): GPU 1.9-2.1 J, **CPU package 2.4-2.6 J**. The CPU does
  2-3% of the expert work in decode yet costs more than the GPU. Idle package power is 86 W because
  the `performance` governor holds every core at a 3.0 GHz floor (cpu0 scaling_min_freq 3,012,481 kHz).

## Measured: prefill ring and chunk (2 rounds, 80K fresh fixture + 2 short each)

| candidate | chunk / ring | prefill tok/s | 80K wall s | decode tok/s |
| --- | --- | ---: | ---: | ---: |
| baseline | 8192 / 96 | 873 | 94.9 | 61.2 |
| `STRATA_PREFILL_AUTO_MAX=16384` + `STRATA_PREFILL_RING=192` | 16384 / 192 | 917 | 90.5 | 60.8 |
| same, ring 384 | 16384 / 384 | 918 | 90.5 | 60.1 |
| same, ring 512 | 16384 / 512 | 914 | 90.8 | 61.2 |

The gain (+5%, -4.4 s per 80K) is the chunk, not the ring: all three ring sizes measure the same, so at
this clock the GPU, not the PCIe stream, is the limiter. The baseline's 96-slot ring comes from the
pinned-share check running before the complement is pinned (`src/prefill/prefill.cpp`, `ring_slots`);
upstream fixed the classification in 0.1.41 (issue 1353). Short-prompt decode and energy unchanged.

## Measured: decode knobs (1 round, 4 short requests each; the second round was lost to the reserve-300 crash)

| candidate | decode tok/s | accept | GPU J/tok | CPU J/tok | note |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline (12 workers, draft 4, KV window 32K) | 67.3 | 0.76 | 1.91 | 2.34 | |
| 4 workers | 67.3 | 0.76 | 1.92 | **2.09** | same speed, -11% CPU energy |
| 8 workers | 67.4 | 0.76 | 1.91 | 2.24 | |
| 16 workers | 67.5 | 0.76 | 1.91 | 2.36 | |
| `STRATA_POOL_SPIN_US=2000` | 67.4 | 0.76 | 1.89 | 2.19 | workers sleep after 2 ms instead of 20 ms |
| draft 3 | 63.5 | 0.78 | 1.94 | 2.37 | slower |
| draft 6 | 63.7 | 0.64 | 1.96 | 2.37 | slower, acceptance drops |
| KV window 64K | 63.3 | 0.72 | 1.91 | 2.42 | fewer expert slots (12,812), slower |
| `--vram-reserve-mib 300` | - | - | - | - | 46 MiB free after load; MMQ cudaMalloc OOM on the first prompt. Dead end. |

With 97-98% of routed experts hitting the GPU cache, the worker count does not move decode speed at all;
it only moves CPU energy. Draft 4 and the 32K KV window stay. The CPU-side energy is dominated by the
governor floor (see above), so the EPP measurement decides the rest.

## Measured: GPU power limit and locked clocks (2 rounds, 80K fresh fixture + 2 short each)

| GPU limit / clock | decode tok/s | prefill tok/s | 80K wall s | GPU+CPU J/token (short) | sampled peak W |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 W boost (production) | 65.1 | 879 | 94.5 | 4.31 | 150-174 |
| 100 W locked 1000 MHz | 61.8 | 880 | 94.8 | 4.36 | 143 |
| 100 W locked 900 MHz | 56.7 | 877 | 95.7 | 4.56 | 144 |
| 100 W locked 800 MHz | 51.8 | 849 | 98.8 | 4.79 | 135 |
| **125 W boost** | **71.0** | **1122** | **74.7** | **4.28** | 161-169 |
| 150 W boost | 74.0 | 1264 | 66.6 | 4.48 | 201-208 |
| 150 W locked 1100 MHz | 65.0 | 1217 | 69.4 | 4.31 | 177 |

GPU joules per token are flat from 100 to 125 W (prefill ~0.115 J per prompt token at both); CPU joules per
token fall as the GPU finishes sooner, so the system total is lowest at 125 W. Locking the SM clock under a cap
never helped: the power capper already chooses the best clock, a lock only trims transient peaks by ~30 W. The
card already draws 150-174 W transients at the 100 W cap; the 2026-10-15 power loss followed a sustained 160-182 W
average, which 125 W stays well under. Decision (user, 2026-10-18): soak at 125 W for ~15 minutes of continuous
180K-prompt work under the watchdog, adopt 125 W as the boot cap if clean.

## Measured: CPU energy preference under load (2 rounds, 4 short each)

| EPP | decode tok/s | CPU J/token |
| --- | ---: | ---: |
| performance (governor performance) | 64.0 | 2.37 |
| balance_performance (governor powersave) | 63.8 | 2.38 |
| balance_power | 63.6 | 2.46 |

No effect under load: the expert pool keeps the cores busy whatever the preference. The EPP's value is at idle
(86 W package with the performance governor's 3.0 GHz floor); idle measurement pending.

## Engine 0.1.43.1 build

`build-v100-0143/unbiased-strata` (74 MB) built from the imported upstream 0.1.43.1 sources plus the fork's front
end, CUDA 12.9 / GCC 14, sm_70, `STRATA_EXPERIMENTAL_SM60=ON`, 4 jobs under the watchdog: 3 minutes, CPU Tctl peaked
at 86 C (the BIOS cap is 85 C: the CPU cooler is marginal for sustained all-core load). A/B against the tuned 0.1.39
binary pending (`local/v100/cands-engine.json`: defaults, `STRATA_SM70_TABLE=1`, `STRATA_ROUTE_TAIL_SKIP=0`,
`STRATA_KV_PREFETCH=1`, `STRATA_PREFILL_CPU_SHARE=0`).

## 125 W soak (passed) and adoption

Five back-to-back full 180K prefills (prompt cache defeated), 15.6 min of continuous load at the 125 W cap: GPU power
mean 123 W, p95 157 W, peak sample 192 W; GPU 56 C max; CPU Tctl 85.8 C max (the BIOS cap, 12 pool workers); no
reset, no watchdog stop. 180K prefill 939-947 tok/s at 125 W; decode after the 180K prompt rose from 60 to 78 tok/s
over the five requests as the adaptive cache's hit rate climbed from 88% to 98%. Decision: 125 W is the boot cap
(`adopt/gpu-nv-power-override.conf`, installed with `adopt/install-gpu-125w.sh`). The engine A/B runs at 125 W.

## Measured: engine 0.1.39 vs 0.1.43.1 at 125 W (2 rounds, 80K fresh fixture + 2 short each, 4 workers, chunk 16384)

| candidate | decode short | decode after 80K | prefill 80K tok/s | GPU J/tok (short) | CPU J/tok (short) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.1.39 (production binary) | 70.1 | 66.9 | 1183 | 2.19 | 1.93 |
| 0.1.43.1 defaults | 74.1 | 73.5 | 1176 | 2.08 | 1.80 |
| **0.1.43.1 + `STRATA_SM70_TABLE=1`** | **80.8** | **79.0** | 1177 | **1.93** | **1.77** |
| 0.1.43.1 + `STRATA_ROUTE_TAIL_SKIP=0` | 74.4 | 72.2 | 1177 | 1.99 | 1.86 |
| 0.1.43.1 + `STRATA_KV_PREFETCH=1` | 74.2 | 73.8 | 1177 | 2.12 | 1.81 |
| 0.1.43.1 + `STRATA_PREFILL_CPU_SHARE=0` | 73.9 | 73.4 | 1177 | 2.13 | 1.82 |

The Volta decode table (PR 1401's kernels, bitwise-identical output per the PR) is worth +15% decode over the old
binary and -12% GPU energy per token. Prefill is unchanged between engines at this cap. The route tail skip
(default on, changes answers slightly) and the CPU prompt share (default on below 1024-token chunks, changes answers
slightly) measured no gain here, so both are turned off for exact answers at no cost. KV prefetch: no gain.

Idle CPU package power: governor performance 85 W (Tctl 75 C, cores at 4.4 GHz idle) vs powersave +
balance_performance 60 W (Tctl 62 C). Under load the two are equal, so the balanced preference is adopted
(`adopt/cpu-epp.service`).

## Final configuration sweep at 125 W and production switch (2 rounds, 80K fixture with correctness gate + 2 short)

| candidate | decode short | decode after 80K | prefill 80K | CPU J/tok (short) | 80K answer has all 3 planted values |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.1.39, 12 workers (old production) | 71.9 | 65.3 | 1188 | 2.08 | 2/2 |
| final, 4 workers | 80.4 | 73.8 | 1182 | 1.74 | 2/2 |
| final + `STRATA_STAGER_SLEEP=1` | 80.4 | 73.6 | 1180 | 1.74 | 2/2 |
| **final, 12 workers** | **81.6** | **79.5** | 1180 | 1.94 | 2/2 |

"final" = engine 0.1.43.1 + `STRATA_SM70_TABLE=1` + `STRATA_ROUTE_TAIL_SKIP=0` + `STRATA_PREFILL_CPU_SHARE=0` +
`STRATA_PREFILL_AUTO_MAX=16384`. With a long context the GPU cache hit rate falls to ~87% and the CPU pool has real
work again, so 12 workers decode 8% faster after an 80K prompt than 4 workers at +0.2 J/token of CPU energy;
12 is kept for production (coding agents live in long contexts). The sleeping stager changes nothing here.

**Production since 2026-10-18 15:58** (`~/.config/unbiasedstrata/config.json`, backup
`config.json.before-v100-tuning-20261018-155758`): binary `local/bin/unbiased-strata-0.1.43.1-sm70`
(sha256 1b90f1b6174645feda18057d377eebb547121b2e18e731599075fcba93f9e066), 12 workers, the four variables above,
GPU boot cap 125 W. Verified live: engine 0.1.43.1, chunk 16384, 13,012 cache slots, 438 MiB VRAM free, health OK,
the variables present in the engine process. Rollback: restore the backup config (old binary, 100 W behaviour
needs `rm /etc/systemd/system/gpu-nv-power.service.d/override.conf`), restart the service.

**Against the 2026-10-18 morning state (100 W, 0.1.39, chunk 8192):** short-prompt decode 67 -> 81 tok/s (+21%),
decode after an 80K prompt 61 -> 79 tok/s (+30%), 80K prefill 873 -> 1180 tok/s (+35%), 80K request 95 -> 71 s,
system energy per generated token lower at every point measured, idle CPU package 85 -> 60 W once
`adopt/cpu-epp.service` is enabled.

## Task-suite wall clock (frozen 24-task DeepSWE2, seed 40, medium reasoning)

| | Oct 13: 0.1.39, 250 W | now: 0.1.43.1 + Volta table, 125 W, vision reserve on |
|---|---:|---:|
| median task wall | 37.2 s | 32.6 s |
| total, 24 tasks | 1017 s | 954 s |
| median TTFT (≈650-token prompts) | 1.5 s | 1.6 s |
| median decode | 77.3 tok/s | 84.5 tok/s |
| passes | 22/24 | 23/24 |

17 of 24 tasks faster, paired geometric-mean wall ratio 0.909; on the 14 tasks whose answers were within 15% of the same length, median wall 37.3 → 32.3 s. The 125 W configuration beats the card's 250 W run from five days ago on wall clock, with one more task passing (e21-index-invalidation). Short-prompt TTFT is unchanged: a 650-token prompt prefills in ~1.5 s either way.

## Vision cost and perception (125 W, 12 workers)

Encoder off vs on (`--vram-reserve-mib 2200`, Q8_0 encoder, 2048 image tokens; 2 rounds): decode short 83.2 -> 80.8
tok/s (-3%), after 80K 81.3 -> 76.7 (-6%, 12,237 vs 13,012 cache slots), prefill unchanged, correctness gate 2/2.
Reserve 1600 was an OOM at the first verify window (74 MiB free); production uses 2400 for margin. Screenshot
read-back (511x1376 terminal UI): title, selected entry and all four file names exact at both 1024 and 2048 image
tokens (the picture is ~720 tokens, under both caps); one of two IP ports missed. Eight synthetic end-to-end checks pass.

## Round 2 at 125 W (final text config, 12 workers; 2 rounds, 80K + 2 short)

| candidate | decode short | decode after 80K | prefill 80K | GPU J/tok | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| final (reference) | 80.9 | 82.0 | 1185 | 1.96 | 2/2 |
| `--adapt-async 1` | **85.5** | **89.5** | 1184 | **1.84** | 2/2 |
| `STRATA_PREFILL_AUTO_MAX=32768` | 82.5 | 83.6 | **1204** | 1.96 | 2/2 |
| `--adapt-every 2` | 84.8 | 78.1 | 1183 | 1.90 | 2/2 |
| `--adapt-swaps 192` | 81.1 | 77.2 | 1183 | 1.88 | 2/2 |
| `--ple-row-cache 4194304` | 80.6 | 81.3 | 1182 | 1.98 | 2/2 |

The async adaptive tier (opt-in in 0.1.43.1; upstream: answers can differ slightly from the blocking tier) is +6-9%
decode at lower energy; the 32K chunk ceiling +1.6% prefill. Both go to a round-3 acceptance on the production
vision configuration plus the 24-task suite before adoption. Swap/adaptation rates and the n-gram row cache: flat.

## Round 3 acceptance on the production vision configuration, and the switch (2026-10-18 evening)

| | production vision config | + `--adapt-async 1` + chunk ceiling 32768 |
| --- | ---: | ---: |
| decode short / after 80K | 78.5 / 82.7 tok/s | **86.4 / 89.6** tok/s |
| 80K prefill | 1182 tok/s | 1200 tok/s |
| GPU J/token (short) | 1.90 | 1.85 |
| correctness gate | 2/2 | 2/2 |
| 24-task suite: median wall / total / passes | 32.6 s / 954 s / 23 | **29.6 s / 874 s / 23** |

The task suite on the candidate: median task wall 29.6 s (Oct 13 at 250 W: 37.2 s), median decode 92.5 tok/s, 23/24
passes with one task flipping each way (e11 now passes, e13 now fails; both were "varies between repeats" tasks in
the Oct 13 three-seed campaign). Adopted. **Production since 18:3x:** the vision binary, Q8_0 encoder at 2048 image
tokens, `--vram-reserve-mib 2400 --adapt-async 1`, `STRATA_PREFILL_AUTO_MAX=32768`, plus the earlier variables;
backup `config.json.before-vision-async-*`. Live checks: health, eight image tests, variables and flags in the engine
process, 125 W cap.

**Against the morning (100 W, 0.1.39, chunk 8192):** short decode 67 -> 86 tok/s (+29%), decode after 80K 61 -> 90
(+47%), 80K prefill 873 -> 1200 (+37%), 80K request 95 -> 69 s, task-suite median 37.2 s (at 250 W!) -> 29.6 s,
with the vision encoder loaded and lower energy per token.

## Round 4: upstream PR 1401's Volta prompt kernels (branch `v100-pr1401`)

PR 1401 (rewin123, open, conflicts with upstream main) carries two halves: the Volta decode kernels (already in
0.1.43.1 as `STRATA_SM70_TABLE=1`, in production) and an FP16 tensor-core prompt-expert GEMM for sm_70
(`pf_wmma`, +9.6-10.3% prompt on the author's V100). A three-way merge against the PR's true base (e8ca9afd, an
ancestor of 0.1.43.1) applied cleanly except the six hunks that re-gate the decode half, which keep 0.1.43.1's
opt-in. Build `build-v100-pr1401/`; the PR's own parity test and a paired A/B against production follow.

PR 1401 result (production config, 2 rounds): parity test 0 failures on this card (IQ2_XXS/IQ2_S/IQ1_M/Q2_0 within
5e-4 of reference); 80K prefill 1200 -> 1214 tok/s (+1.2%), 80K wall 69.2 -> 68.0 s, correctness gate 2/2, decode
within noise. Far from the PR's +10%: at the 125 W cap the prompt path is power-bound (~960 MHz SM), so faster
tensor-core math buys little. Kept (free, parity-verified); `STRATA_PF_WMMA=0` restores the MMQ path.

PRs 2357 (`STRATA_ARGMAX_ROWS_SM70=1`) and 2358 (`STRATA_GR_NORM_FAST=1`), both sm_70 opt-ins claiming +0.5-0.7%
decode, merged cleanly (base 296b59d2) and go into round 5 with the model served from the unencrypted Samsung 980 PRO
(`/mnt/samsung/strata-models/`, the per-token n-gram table reads move off the LUKS-encrypted ORICO). PRs 2355/2356
(a second Volta prompt kernel, +2.5-4.3% prompt at 170-250 W) would conflict in 13 kernel hunks and are not taken:
at the 125 W cap the first Volta prompt kernel already gave only +1.2%.

## Round 5 at 125 W (production vision config; 2 rounds, 80K + 2 short; CPU Tctl sampled)

| candidate | decode short | decode after 80K | prefill 80K | 80K wall | Tctl max |
| --- | ---: | ---: | ---: | ---: | ---: |
| production (0.1.43.1 + Volta table + async cache) | 85.6 | 83.8 | 1196 | 69.4 s | 85.9 C |
| + PR 1401 prompt kernels + PRs 2357/2358 built in (`pr-all`) | 86.6 | 88.1 | **1212** | **68.3 s** | 85.9 C |
| `pr-all` + `STRATA_ARGMAX_ROWS_SM70=1 STRATA_GR_NORM_FAST=1` | 86.5 | 91.5 | 1211 | 68.3 s | 85.9 C |
| `pr-all`, model shards served from the Samsung 980 PRO | 86.1 | 90.4 | 1212 | 68.2 s | 85.8 C |

Decode after 80K swings +-4 tok/s between identical runs (the adaptive cache's state), so the opt-ins and the SSD
move are within noise; the prompt kernels' +1.3% prefill repeats. Start-up is 38-44 s from either drive. Adopted:
the `pr-all` binary (`local/bin/unbiased-strata-0.1.43.1-sm70-pr1401`); the two opt-ins stay off (no measured gain,
exact kernels kept); the model stays on the ORICO drive with the Samsung copy kept as a ready fallback. The CPU sits
at its 85 C cap in every candidate: cooling, not software, decides that number.

## Round 6 at 125 W: CPU boost cap, 135 W probe, resident routing (production vision config, 2026-10-18 night)

Window housekeeping first: the two queued chains for this round had deadlocked for two hours. Each polled
`pgrep -f` for the previous job's label, and that label sat in the chain's own command line, so each matched itself
and slept forever. Rule from now on: one script per window, launched once; never gate on a pattern that appears in
the waiting shell.

### CPU boost cap (`scaling_max_freq`, 2 rounds, 80K + 2 short)

| cap | decode short | decode after 80K | prefill 80K | 80K wall | CPU J/tok (80K) | Tctl max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uncapped (5883 MHz boost) | 87.4 | 88.1 | 1202 | 68.9 s | 28.5 | 85.9 C |
| **4500 MHz** | 87.2 | 88.4 | 1195 | 69.2 s | **19.7** | **64.4 C** |
| 4000 MHz | 87.1 | 86.2 | 1194 | 69.4 s | 17.6 | 57.6 C |
| 3500 MHz | 86.8 | 83.3 | 1195 | 69.4 s | 16.4 | 56.8 C |

The engine does not use the boost: at 4.5 GHz decode is unchanged, 80K prefill is -0.6% (inside the run-to-run
spread), the 80K wall clock is +0.3 s, while the CPU spends 31% less energy per generated token and stops at 64 C
instead of pinning the 85 C BIOS limit. 4.0 GHz starts to cost decode after long prompts (-2%), 3.5 GHz costs 5%.
Adopted: 4500 MHz (`v100-tune maxfreq 4500` now; `adopt/cpu-maxfreq-4500.conf` as `/etc/tmpfiles.d` drop-in for
boot, installer `adopt/install-cpu-maxfreq.sh`). This is the measured answer to the cooling question: the heat came
from single-core boost residency during prefill, which the engine never cashes in. It also makes the BIOS boost
override / Curve Optimizer items low-stakes: with the cap, the chip never visits the voltages where -20 CO is
risky. Note for other workloads on this PC: single-thread peaks lose up to 20% (5.7 -> 4.5 GHz); undo with
`v100-tune maxfreq reset`.

### 135 W probe (1 round, 80K + 2 short; GPU locked at neither)

| cap | decode short | decode after 80K | prefill 80K | 80K wall | GPU W (80K) | GPU J/tok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 125 W | 87.4 | 89.1 | 1203 | 68.9 s | 120 | 35.5 |
| 135 W | 89.6 | 90.7 | **1284** | **64.7 s** | 131 | 36.3 |

Ten more watts buy +6.7% prefill and -6% on an 80K prompt's wall clock (+2% decode) at +2% GPU energy per token:
roughly energy-neutral per unit of work, not a free lunch. Held back pending a soak (peaks ran 192 W at the 125 W cap).

**135 W soak (2026-10-18 22:09-22:24, user-approved after the CPU cap landed): passed.** Five back-to-back 180K
prefills (prompt cache defeated) on the production pr-all binary, 13.3 min under load: secrets 5/5, no watchdog
stop, no kernel/NVRM messages; GPU power mean 133 W, p95 177 W, peak sample 219 W (125 W soak: 123 / 157 / 192);
GPU 59 C max; CPU Tctl 66 C max (the 4.5 GHz cap at work; the 125 W soak ran at 86 C). 180K prefill 1086-1094
tok/s vs 939-947 at 125 W (+15%: the longer the prompt, the more power-bound), 180K request wall 168 s vs 194 s;
decode after the 180K prompt 76 -> 92 tok/s over the five requests (hit rate climbing to 97%). **Adopted: 135 W
is the cap** (`v100-tune pl 135` live; boot override `adopt/gpu-nv-power-override-135.conf`, installer
`adopt/install-gpu-135w.sh`). 150 W remains the approved probe ceiling and was not tried: the peak-transient
margin (219 W samples on a PC that has lost power under GPU load) is the reason to stop here.

### `STRATA_ROUTE_RESIDENT=0.5` (2 rounds, 80K + 2 short)

| candidate | decode short | decode after 80K | GPU hit % (80K) | GPU J/tok (80K) | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| production | 87.1 | 87.9 | 80.5 | 38.5 | 2/2 |
| route-resident 0.5 | 88.4 | **94.4** | **92.3** | 36.7 | 2/2 |

Biasing the router toward experts already resident on the GPU lifts the cache hit rate from 80% to 92% after a
long prompt and decode by 7%, at lower GPU energy. It changes which experts answer, so the 80K secret gate is not
enough; the frozen 24-task suite decided against it: 22/24 passes (production 23/24; e5 and e24 flipped to fail, e13
to pass), median task wall 32.7 s vs 29.6 s, because the answers ran longer (median 2886 vs 2570 generated tokens)
while task-level decode was +0.7%. The per-token gain is real, the per-task result is worse. **Rejected.**
Result files: `bench/results/v100-routeres-0.5.json` vs `v100-after-tuning-async32k-flash-40.json`;
`local/compare_tasks.py` prints the comparison.

### Production after round 6

Binary `local/bin/unbiased-strata-0.1.43.1-sm70-pr1401` (pr-all, switched 21:31, health + eight image tests pass),
135 W (after the soak), CPU boost capped at 4500 MHz, everything else as in round 3. Window script: `local/v100/window6.sh`.

## Round 7 at 135 W: what is left before the BIOS (production pr-all binary, CPU capped 4.5 GHz)

Levers checked and closed without a run: transparent huge pages are already `always` (37 of the engine's 46 GB
resident as huge pages); `--suffix-draft` (prompt-lookup drafting) is the engine default; `--kv k8v4` cannot stream,
so it would put the whole 196K KV back into VRAM and cost expert slots; PCIe is at the Gen3 x16 limit, HBM2 is
fixed-clock, EPP was within 0.5 tok/s. Driver note: stay on 580 / CUDA 12 (CUDA 13 dropped Volta).

### GPU clock ceiling (`-lgc 135,MAX`, aimed at the power transients, 1 round)

| ceiling | decode short | decode after 80K | prefill 80K | GPU peak sample | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| none (1380) | 87.9 | 90.4 | 1298 | 210 W | 197 W |
| 1200 MHz | 84.7 | 86.0 | 1299 | 211 W | 202 W |
| 1100 MHz | 79.9 | 81.1 | 1294 | 216 W | 204 W |

The ceiling costs decode (short prompts run at 1300-1340 MHz when unlocked) and does not clip the transients: the
peaks are the load burst itself, not the clock. Dead end on both counts; clocks stay unlocked.

### CCD affinity, 16 workers, draft length 5 (2 rounds, 80K + 2 short, 135 W)

| candidate | decode short | decode after 80K | prefill 80K | 80K wall | CPU J/tok (80K) | GPU hit % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| production (12 workers, draft 4, unpinned) | 87.9 | 90.7 | 1293 | 63.9 s | 19.7 | 80.8 |
| pinned to CCD0 (cores 0-7 + SMT 16-23) | 86.6 | **83.9** | 1294 | 64.1 s | 19.6 | 81.5 |
| 16 workers | 85.3 | 91.2 | 1292 | 64.3 s | 16.9 | 82.8 |
| draft length 5 | 81.6 | 88.2 | 1292 | 64.2 s | 18.4 | 80.6 |

Pinning the engine to one chiplet costs 7% decode after a long prompt: the expert streaming path wants both L3s.
(First attempt note: `AllowedCPUs=` on a user unit is silently inert here, the session has no cpuset controller
delegated; the harness now pins with a `taskset` prefix, profile key `cpu_affinity`, and verifies through
`/proc/PID/status`.) 16 workers trades 3% short-prompt decode for 14% less CPU energy after long prompts, a wash;
draft 5 lowers acceptance (0.65 vs 0.70) and loses on both. Nothing adopted; the shortlist before the BIOS is
exhausted. Production stays: pr-all binary, 135 W, 12 workers, draft 4, unpinned, CPU capped 4.5 GHz.

## Round 8 at 150 W: the second PCIe connector (2026-10-25)

Since 2026-10-23 the V100 is fed by both PCIe 8-pin connectors, so the power cap was probed again (production pr-all
config, 2 rounds, 80K + 2 short, guarded; `local/v100/pl150.sh`):

| cap | 80K prefill | 80K wall | decode short | decode after 80K | peak sample | p99 | GPU max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 135 W | 1298 | 64.0 s | 86.9 | 91.5 | 231 W | 206 W | 58 C |
| **150 W** | **1393** | **59.8 s** | 89.3 | 92.2 | 251 W | 224 W | 60 C |
| 165 W | 1464 | 57.2 s | 89.9 | 91.3 | 282 W | 266 W | 62 C |

Secrets 2/2 everywhere. 165 W adds 4-5% on long prompts but its transients pass the card's 250 W rating, on a PC whose
2026-10-24 power loss is still unexplained; not adopted. **150 W soak passed** (`soak150.sh`: five back-to-back salted
180K prefills, 13 min): prefill 1173-1182 tok/s (135 W: 1086-1094), decode after 180K 93 tok/s, secrets 5/5, mean
144 W, peak sample 259 W, p99 234 W, GPU 64 C, Tctl 73 C. Production runs at 150 W (runtime, `v100-tune pl 150`);
the boot drop-in still says 135 W until changed with root.

Knobs re-checked at 150 W with long contexts (2 rounds, 1 short + fresh 80K + 128K; `round8.sh`):

| candidate | prefill 80K | prefill 128K | decode 80K | decode 128K | 128K wall | slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| production (96-slot ring) | 1391 | 1283 | 88.7 | 92.9 | 101.5 s | 12,135 |
| ring 192 | 1391 | 1282 | 92.1 | 93.5 | 101.6 s | 12,135 |
| ring 384 | 1386 | 1279 | 89.9 | 93.6 | 101.8 s | 12,135 |
| KV window 64K | 1392 | 1286 | 90.2 | 94.0 | 101.2 s | 11,931 |
| 16 workers | 1391 | 1288 | 91.0 | 91.6 | 101.1 s | 12,135 |
| vision off (reference only) | 1393 | 1292 | 92.7 | 95.2 | 100.8 s | 13,012 |

Prefill is identical across ring sizes even at 150 W: the 96-slot ring is not what limits it, and the stream is not the
bottleneck. Decode spread follows draft acceptance (production's own two rounds: 86.3 and 91.0 after 80K) and is inside
noise for every knob. The KV window and 16 workers start slower (60 s / 54 s vs 44 s) and gain nothing. Vision costs
877 expert slots and about 3-8% short-prompt decode and prefill, 2-3% after long prompts; it stays on (the user's
choice; the reserve is already near its floor: 1600 MiB was an OOM, 2200 the smallest that worked). Nothing adopted
besides the cap.

## Round 9: upstream Strata 0.2.6 (engine 0.1.50) against production, 150 W (2026-10-25)

Upstream now carries every Volta piece this fork added (SM70 table, PR 1401's prompt attention, #2357/#2358) plus new
opt-ins. Set up from source in `~/opt/strata-upstream/0.2.6` (its own data folder; the CUDA 12 engine package
sha-verified against the release's SHA256SUMS), run through its Python server (`serve/server.py`, configs in `src/ab/`)
under the same harness via the profile key `argv`. **Stock defaults do not fit at 196K:** without `--resident-experts`
the 46.8 GiB of experts go to anonymous RAM and less than 4 GB stayed free (guard stop; upstream's own setup estimates
~77 GB for IQ3_S at 192K). The candidates below therefore run 0.1.50 with production's engine arguments and switches
(our pack and expert profile, `--resident-experts --kv-resident 32768 --adapt-async 1`, 12 workers, Q8_0 encoder at
2048 image tokens, `STRATA_PF_WMMA=1`, `STRATA_ENGINE_NICE=0`; 2 rounds, 1 short + fresh 80K + 128K):

| candidate | slots | prefill 80K | prefill 128K | 80K wall | 128K wall | decode short | decode 80K | decode 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| production 0.1.43.1 + PR 1401 | 12,194 | 1395 | 1288 | 59.5 s | 101.1 s | 89.1 | 88.4 | 94.1 |
| 0.1.50, reserve 2400 | 11,477 | 1523 | 1400 | 55.0 s | 93.2 s | 87.1 | 86.0 | 88.4 |
| 0.1.50, reserve 1000 | 12,135 | 1522 | 1399 | 55.0 s | 93.2 s | 87.9 | 84.6 | 91.6 |
| **0.1.50, reserve 1000 + `STRATA_GDN_REC_WARP=2`** | 12,135 | **1526** | **1407** | **54.7 s** | **92.7 s** | **91.3** | 87.3 | 91.9 |

Secrets 4/4 for every candidate; power identical (peak samples 248-252 W). The new engine reads long prompts **9%
faster** (80K and 128K requests 8% shorter); decode is the same within round-to-round noise (production's own two
rounds after 80K: 90.3 and 86.4). 0.1.50's picture reader leaves 2.1 GB of VRAM idle at the 2400 MiB reserve, so the
reserve-1000 runs hold the same slot count as production. `STRATA_GDN_REC_WARP=2` changes the FP32 summation order of
the DeltaNet recurrence (answers can differ slightly) and measured best on every row. Not adopted yet: it needs a
180K-prompt-plus-two-images fit check at the 1000 MiB reserve and the 24-task suite (answer quality) first.

**Adoption gates (same day, 150 W).** Vision gate (`local/vision_gate.py`: `test_vision.py`, seven scored questions on two
2560x1440 synthetic screenshots from `local/vision_assets.py`, each at the 2048-token picture cap, and the worst case:
both pictures plus the 180K fixture in one request):

| | production | 0.1.50, reserve 1000 + GDN 2 |
| --- | --- | --- |
| functional checks | pass | pass |
| perception | 7/7 | 7/7 |
| worst case (183,704 prompt tokens) | 160.1 s, secrets 3/3, still answers after | 149.0 s, secrets 3/3, still answers after |
| least VRAM free during the gate | 906 MiB | 782 MiB |

24-task DeepSWE2 suite (seed 40, medium reasoning): production 23/24, median task 28.3 s, suite 878 s, median decode
97.3 tok/s; 0.1.50 24/24 (e4 flips to a pass), median task 32.2 s, suite 913 s, median decode 95.0 tok/s. Time per
generated token is 2% higher on 0.1.50 (0.0111 vs 0.0109 s; decode lower on 18 of 24 tasks); the rest of the wall
difference is answer length, which moves both ways by up to 2x per task (e3 2,207 -> 4,856 tokens, e14 7,142 -> 4,921),
median +2%. Short agent prompts gain nothing from the faster prompt read. **Verdict: a wash for this workload** (+9%
on fresh long prompts, -2% decode, quality and vision equal on one seed); production stays on 0.1.43.1 + PR 1401. The
0.1.50 setup is kept ready in `~/opt/strata-upstream/0.2.6` (profile `local/v100/up-best-18081.json`).

## Still open

- Rebooted 2026-10-18 16:27: `iommu=pt` active (all 27 IOMMU groups identity), NMI watchdog off, PCIe probe unchanged at
  13.1 GB/s (already at the Gen3 x16 practical limit; the passthrough removes IOTLB work, not bandwidth). The NVIDIA PAT
  option is inert: driver 580 exposes no module parameters under /sys/module/nvidia (the drop-in can be removed).
  `cpu-epp.service` ran but power-profiles-daemon (Omarchy) re-applied its `performance` profile; `powerprofilesctl set
  balanced` is the persistent fix: governor powersave, EPP balance_performance, idle package 59 W / Tctl 63 C with the
  model loaded (was 84 W / 76 C).
- Vision: encoder helper build, front-end image support, VRAM reserve sizing (approved, after this).
- CPU cooling: resolved by measurement in round 6. The 85 C was boost residency the engine never used; the 4.5 GHz
  cap holds Tctl at 64 C with identical engine numbers. Boot persistence needs the tmpfiles drop-in installed
  (`adopt/install-cpu-maxfreq.sh`). BIOS items left: Restore AC Power Loss = Power On (the one that matters for a PC
  that drops power), AI Overclocking off and Curve Optimizer -20 -> -10/0 as stability hygiene.


## Vision (approved 2026-10-18, after the tuning lands)

Goal: the model's own encoder (`mmproj-Qwen3.8-Flash-Next-BF16.gguf`, 27-layer ViT + projector) on the GPU, image
token budget 2048 per picture (upstream default 1024), no disk staging. The engine already takes images (`--vision`,
GENI requests with an embeddings file and M-RoPE positions); upstream's `tools/vision/strata_vision.cpp` builds the
encoder helper from llama.cpp's mtmd. The fork's front end refuses `--vision` today. Work: build the helper from the
submodule, accept OpenAI `image_url` parts (data: URLs and files) in the front end, run the helper, pass the
embeddings to the engine; size `--vram-reserve-mib` from a measured start (the 300 MiB experiment showed 46 MiB free
is an OOM), then A/B text speed with the reserve on vs off and verify a 180K prompt with two images fits.
Expected cost: ~700 fewer expert slots (5%), 2-6% decode.

Encoder files (`/home/aim/opt/Strata-data/models/mmproj/`): `mmproj-Qwen3.8-Flash-Next-BF16.gguf` 907,543,008 B,
sha256 b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0 (matches the Hub). A Q8_0 requantization
(`tools/vision/mmproj_q8.py`, llama.cpp's own rules: 2-D weights Q8_0, biases/norms/position embeddings F32, the
patch-embedding convolution and the 27 `ffn_down` weights (inner dim 4304, not a multiple of 32) F16) is
614,933,856 B, sha256 182b8b0643e90a4480f6686170d6ca7055cc52f006bf0a1780caad62d873b126. The Q8_0 file is the one to
load: ~290 MB less VRAM for the encoder weights at a negligible accuracy cost.
