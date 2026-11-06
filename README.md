# When Greedy Is Sorting

Code for the paper **When Greedy Is Sorting: Exact and Certified Ideal Rankings for Novelty Metrics**
(Kehan Liu, Zhaoyun Chen, Xianquan Yan), WSDM 2027 Findings.

## Setup

Use Python 3.12 or later. All experiments run on a CPU; no GPU is required.
Run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Exact certificates require **SoPlex 8.0.3 with GMP rational support**, with
`soplex` on `PATH`. Obtain it from the [official SoPlex project](https://soplex.zib.de/).
The implementation requires an optimal rational solution with zero primal
and dual violation and checks its primal constraints exactly. Numerical LP
timing uses SciPy/HiGHS and is separate from the rational certificate decisions.
PyMuPDF is used to read the official result appendices in the leaderboard stage.

## Prepare the external inputs

Download the four official TREC Web diversity judgment files:

```bash
python prepare_trec.py
```

The downloader writes to the ignored `data/trec_diversity/` directory and
keeps any files already present. Data are fetched from NIST rather than bundled.

For the MIND-small runtime benchmark, obtain `MINDsmall_train.zip` and
`MINDsmall_dev.zip` from the [official MIND release](https://msnews.github.io/)
under its data terms:

The legacy Azure download endpoint currently rejects public requests. The
[Recommenders mirror](https://huggingface.co/datasets/Recommenders/MIND)
provides the archives at this fixed revision:

- <https://huggingface.co/datasets/Recommenders/MIND/resolve/2115068f457b577f93a268fffe22373e45727d89/MINDsmall_train.zip?download=true>
- <https://huggingface.co/datasets/Recommenders/MIND/resolve/2115068f457b577f93a268fffe22373e45727d89/MINDsmall_dev.zip?download=true>

The same upstream data terms apply.

```bash
python prepare_mind.py \
  --train-zip /path/to/MINDsmall_train.zip \
  --dev-zip /path/to/MINDsmall_dev.zip
```

This extracts the three files actually used: training news,
training behaviors and development news. Development behaviors are not used.

## Reproduce the paper experiments

| Stage | Reported quantities reproduced |
|---|---|
| `python reproduce.py synthetic` | Exhaustive optimum comparisons for support-one alpha-DCG and ERR-IA |
| `python reproduce.py numeric` | TREC query/document structure, numerical full-query LP timings, MIND-small alpha-DCG timings and MIND-sized ERR-IA timings |
| `python reproduce.py exact --workers 8` | Exact sampled-pool and full-query LP decisions, greedy optimality gaps and solver-free projection bounds |
| `python reproduce.py erria` | Objective agreement and runtime on support-one TREC ERR-IA queries |
| `python reproduce.py leaderboard` | Score intervals and preservation of published leaderboard comparisons |

The `numeric` stage requires both TREC and MIND inputs. The `exact` and
`erria` stages require TREC inputs. Run `leaderboard` after `exact`; it reads
the newly computed exact intervals and downloads the official NIST result
appendices. To run every stage in this dependency order:

```bash
python reproduce.py all --workers 8
```

The seeds, cutoffs, sampling and repetition counts retain the paper's settings.
Exact decisions treat the evaluator's stored binary64 discounts as exact
rational numbers, not decimal approximations. Overlap certificates cover
alpha-DCG; the exact support-one algorithms cover both alpha-DCG and ERR-IA.

Newly computed measurements and summaries are written under `outputs/`.
Input files, intermediate files, precomputed results, figures and separate
software tests are not distributed. Wall-clock times vary with hardware and
system load; numerical comparisons retain their objective-equality checks.

## Implementation

- `src/chainmerge.py`: exact ChainMerge paths, metric computation and the
  repeated marginal-scan comparators used in the experiments.
- `src/exact_certificate.py`: exact-rational LP and projection certificates.
- `experiments/`: numerical LP, exhaustive and runtime experiments.
- `formal/`: exact TREC recomputation and leaderboard interval calculations.
