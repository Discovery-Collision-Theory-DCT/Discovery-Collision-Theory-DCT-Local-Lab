# Example DCT Experiment Report

Run: `quickstart_20260307_000000`  
Config: `config/quickstart.yaml`  
Model backend: local OpenAI-compatible endpoint

## Benchmark Families
- symbolic
- dynamical
- compression

## Methods
- baseline_single_a
- baseline_single_b
- baseline_merged_naive
- full_dct

## Aggregate Metrics (illustrative example)
| Method | Validity Rate | Held-out Accuracy | Exact Match | Compression Score | Novelty | Time-to-Valid | Cumulative Improvement |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_single_a | 0.42 | 0.58 | 0.19 | 0.041 | 0.77 | 1.7 | 0.23 |
| baseline_single_b | 0.39 | 0.55 | 0.17 | 0.039 | 0.79 | 1.9 | 0.20 |
| baseline_merged_naive | 0.45 | 0.60 | 0.20 | 0.038 | 0.73 | 1.5 | 0.26 |
| full_dct | 0.56 | 0.69 | 0.29 | 0.044 | 0.81 | 1.1 | 0.35 |

## Uplift of full_dct over baselines
- vs `baseline_single_a`
  - validity rate: `+0.14`
  - held-out accuracy: `+0.11`
  - exact match rate: `+0.10`
  - cumulative improvement: `+0.12`
- vs `baseline_single_b`
  - validity rate: `+0.17`
  - held-out accuracy: `+0.14`
  - exact match rate: `+0.12`
  - cumulative improvement: `+0.15`
- vs `baseline_merged_naive`
  - validity rate: `+0.11`
  - held-out accuracy: `+0.09`
  - exact match rate: `+0.09`
  - cumulative improvement: `+0.09`

## Observed Pattern
- Collision hypotheses had higher acceptance than naive merged candidates in symbolic and dynamical tasks.
- Memory write-back improved later-round novelty and reduced time-to-valid-discovery.
- Compression benchmark benefited modestly; best gains were in symbolic and dynamical families.

## Scientific Honesty
These results validate DCT behavior only in controlled synthetic environments. They do not establish autonomous real-world scientific law discovery.
