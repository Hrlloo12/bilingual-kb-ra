# Concurrency report

**Hardware:** CPU Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz, 24 usable vCPUs, 257,863 MiB RAM; GPU NVIDIA L4 (23,034 MiB, driver 595.71.05); Linux-6.8.0-139-generic-x86_64-with-glibc2.35.

Closed-loop load: each simulated user sends its next request as soon as the previous one returns. Every level draws from the same shuffled question order (seed 2026). Interactive users run real two-turn conversations, each in its own session, so half of the interactive requests are follow-ups that go through the rewrite. GPU and host figures are sampled every second during the level.

## Quick Search

| Users | Requests | Failures | QPS | avg ms | p50 | p95 | max | GPU util avg % | GPU mem max MiB | CPU avg % | Host RAM max MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 48 | 0 (0.0%) | 78.44 | 12.6 | 12.6 | 13.7 | 15.2 | – | – | – | – |
| 2 | 48 | 0 (0.0%) | 163.34 | 12.0 | 12.2 | 16.5 | 21.5 | – | – | – | – |
| 4 | 64 | 0 (0.0%) | 232.87 | 16.8 | 15.7 | 23.9 | 34.1 | – | – | – | – |
| 8 | 128 | 0 (0.0%) | 218.52 | 36.1 | 35.7 | 50.7 | 57.4 | – | – | – | – |
| 16 | 256 | 0 (0.0%) | 240.61 | 65.6 | 63.4 | 89.8 | 116 | 0 | 16,284 | 18 | 13,653 |

On the GPU host, Quick Search levels with 1–8 users finished in under one second, before the first one-second resource sample, so only the 16-user level has resource figures. Quick Search does not use the GPU; the GPU memory shown is the idle serving stack. The CPU-only run below was measured after the sampler was changed to also sample at the start and end of each level.

## Smart AI Search

| Users | Requests | Failures | QPS | avg ms | p50 | p95 | max | GPU util avg % | GPU mem max MiB | CPU avg % | Host RAM max MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 48 | 0 (0.0%) | 1.30 | 769 | 796 | 1,258 | 1,346 | 90 | 16,300 | 5 | 13,783 |
| 2 | 48 | 0 (0.0%) | 2.22 | 889 | 924 | 1,413 | 1,487 | 96 | 16,300 | 7 | 13,775 |
| 4 | 64 | 0 (0.0%) | 3.66 | 1,077 | 1,072 | 1,698 | 1,803 | 97 | 16,308 | 11 | 13,806 |
| 8 | 128 | 0 (0.0%) | 5.28 | 1,481 | 1,558 | 2,297 | 2,806 | 99 | 16,340 | 11 | 13,894 |
| 16 | 256 | 0 (0.0%) | 5.59 | 2,820 | 2,900 | 3,642 | 4,235 | 99 | 16,406 | 12 | 13,930 |

## Interactive AI Search

| Users | Requests | Failures | QPS | avg ms | p50 | p95 | max | GPU util avg % | GPU mem max MiB | CPU avg % | Host RAM max MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 48 | 0 (0.0%) | 0.77 | 1,291 | 1,237 | 1,787 | 2,229 | 87 | 16,406 | 5 | 13,901 |
| 2 | 48 | 0 (0.0%) | 1.39 | 1,416 | 1,373 | 1,908 | 2,401 | 94 | 16,406 | 7 | 13,850 |
| 4 | 64 | 0 (0.0%) | 2.34 | 1,614 | 1,539 | 2,299 | 2,963 | 96 | 16,406 | 8 | 13,871 |
| 8 | 128 | 0 (0.0%) | 3.73 | 2,024 | 1,953 | 2,846 | 3,642 | 97 | 16,406 | 10 | 13,912 |
| 16 | 256 | 0 (0.0%) | 4.40 | 3,483 | 3,430 | 4,825 | 6,535 | 96 | 16,406 | 10 | 13,928 |

Follow-up turns only:

| Users | avg ms | p95 ms |
|---|---|---|
| 1 | 1,442 | 2,160 |
| 2 | 1,573 | 2,152 |
| 4 | 1,770 | 2,607 |
| 8 | 2,176 | 3,252 |
| 16 | 3,568 | 4,831 |

## Quick Search on a CPU-only host

**Hardware:** CPU AMD EPYC 7763 64-Core Processor, 4 usable vCPUs, 15,994 MiB RAM; GPU none; Linux-6.8.0-1064-azure-x86_64-with-glibc2.36.

| Users | Requests | Failures | QPS | avg ms | p50 | p95 | max | GPU util avg % | GPU mem max MiB | CPU avg % | Host RAM max MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 48 | 0 (0.0%) | 127.57 | 7.8 | 7.5 | 9.6 | 10.7 | – | – | 66 | 2,989 |
| 2 | 48 | 0 (0.0%) | 183.93 | 10.7 | 10.1 | 15.7 | 17.4 | – | – | 79 | 2,990 |
| 4 | 64 | 0 (0.0%) | 230.23 | 16.9 | 16.8 | 24.2 | 33.4 | – | – | 84 | 2,993 |
| 8 | 128 | 0 (0.0%) | 227.15 | 34.3 | 32.3 | 57.2 | 78.8 | – | – | 95 | 2,997 |
| 16 | 256 | 0 (0.0%) | 233.24 | 66.9 | 61.6 | 112 | 205 | – | – | 49 | 3,001 |
