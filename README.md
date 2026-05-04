# htmrl


![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)

**Research tooling for HTM + reinforcement learning (HTMRL) pipelines** — encoders, HTM-style spatial pooling, Gymnasium environments, and RL agents (including [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)) so you can prototype, measure, and compare end-to-end pipelines in one codebase.

This repository began as a **PSU Capstone** project. It packages layers you can compose (input → encoding → agent / “brain” → environment) and ships tests and demos that characterize encoder and spatial pooler behavior.

The [Numenta HTM](https://numenta.com/resources-hierarchical-temporal-memory/) family of ideas informs the design. An `htm` dependency is pulled from Git as configured in `pyproject.toml` ([`tool.uv.sources`](pyproject.toml)); the figures at the end include a **comparison** against **HTM core Python bindings** where applicable.

## Table of contents

- [htmrl](#htmrl)
  - [Table of contents](#table-of-contents)
  - [About](#about)
  - [Quick start](#quick-start)
  - [Project layout](#project-layout)
  - [Dependencies (summary)](#dependencies-summary)
  - [Development setup](#development-setup)
  - [Makefile targets](#makefile-targets)
  - [Troubleshooting](#troubleshooting)
  - [Validation figures (this implementation)](#validation-figures-this-implementation)
    - [Encoder overlap vs distance](#encoder-overlap-vs-distance)
    - [Spatial pooler: overlap vs distance (active columns)](#spatial-pooler-overlap-vs-distance-active-columns)
    - [Spatial pooler: activation frequency distributions](#spatial-pooler-activation-frequency-distributions)
    - [Noise robustness and continual learning](#noise-robustness-and-continual-learning)
  - [Comparison: HTM core Python bindings](#comparison-htm-core-python-bindings)
  - [Citation](#citation)
  - [Acknowledgments](#acknowledgments)
  - [License](#license)

## About

**Who this is for:** researchers and engineers who want a **Python-first** workspace to experiment with SDR encoders, spatial pooler dynamics, and RL training without rewriting glue code each time.

**What you can do here:**

- Build pipelines that combine **encoders** (scalar, RDSE, date, category, …), **HTM-oriented agent layers**, and **Gymnasium** environments.
- Run the **automated research demo** (`scripts/research_pipeline_demo.py`) to exercise encoder checks, a small experiment matrix (brain vs PPO vs tabular baselines), and plotted summaries under `reports/research_demo/` by default.
- Use **`src/demo_driver.py`** for interactive exploration of input → encoder → brain flows (some demos expect files under `data/`).

For full install steps, Make on each OS, venv recreation, and git habits, see **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## Quick start

From the repository root, after [uv](https://github.com/astral-sh/uv) is installed:

```bash
make install    # or: uv sync --all-groups (see CONTRIBUTING.md)
make test       # pytest suite + coverage (see Makefile for ARGS)
```

**End-to-end research demo** (writes to `reports/research_demo/` unless `--out-dir` is set):

```bash
uv run python scripts/research_pipeline_demo.py --episodes 2 --max-steps 5
```

Use `uv run python scripts/research_pipeline_demo.py --help` for options (environments, PPO pretrain steps, skipping the encoder stage, log level, etc.).

**Encoder showcase** (also invoked from the research demo):

```bash
uv run python scripts/encoder_types.py --help
```

## Project layout

| Path | Role |
|------|------|
| [`src/htmrl/`](src/htmrl/) | Main package: `encoder_layer`, `agent_layer`, `environment`, `input_layer`, `grapher`, logging utilities |
| [`scripts/`](scripts/) | `research_pipeline_demo.py`, encoder demos, plotting helpers |
| [`tests/`](tests/) | Pytest unit and integration tests |
| [`data/`](data/) | Sample datasets referenced by demos (`easyData.xlsx`, etc.) |
| [`src/legacy/`](src/legacy/) | Older HTM / encoder paths kept for reference |
| [`user/`](user/) | User-specific experiments (e.g. custom pipelines) |

## Dependencies (summary)

Declared in [`pyproject.toml`](pyproject.toml): includes **SymPy**, **pandas**, **matplotlib**, **scikit-learn**, **SciPy**, **Gymnasium**, **Stable-Baselines3**, **PyTorch**, and related stack; optional **`rl`** extra is documented there. Dev and test groups add **pytest**, **pre-commit**, linters, and formatters.

**Platform note:** PyTorch wheels are not published for every OS and CPU pair. If `uv sync` fails on **Intel macOS** or another unsupported combo, relax or override the `torch` constraint for your platform (see uv’s hints when resolution fails).

## Development setup

Short path:

1. Install **uv** and **Make** — see [CONTRIBUTING.md](CONTRIBUTING.md) (environment and Make sections).
2. From the repo root: `make install` (creates/refreshes `.venv`, syncs groups, installs pre-commit when in a git repo).

## Makefile targets

```bash
make <target>
```

| Target | Description |
|--------|-------------|
| `make help` | List all commands |
| `make install` | Create/refresh env, sync groups, install pre-commit hooks |
| `make setup-dev` | Dev dependencies sync |
| `make format` | Format with isort and black |
| `make lint` | Run flake8 |
| `make test` | Run tests with coverage (`make test ARGS="-v tests/test_file.py"` for a subset) |
| `make clean` | Remove common build/test artifacts |
| `make update` | `uv lock --upgrade` |
| `make pre-commit` | Run all pre-commit hooks |
| `make recreate-venv` | Force rebuild `.venv` |

## Troubleshooting

| Issue | Likely cause | Fix |
|-------|----------------|-----|
| `uv` not found | uv not installed | [Install uv](CONTRIBUTING.md) |
| Wrong Python version | Env not pinned | `uv python install 3.12 && uv python pin 3.12` |
| Dependencies outdated | Stale lockfile | `uv lock --upgrade && uv sync --all-groups` |
| Pre-commit fails | Missing / stale env | `make install` |
| Wrong or broken env | Stale `.venv` | `make recreate-venv` |

---

## Validation figures (this implementation)

These plots support qualitative validation of **overlap vs distance** for encoders, **active-column overlap** after the spatial pooler, **column activation frequency** under different inputs and training epochs, **noise robustness**, and **synapse formation** under dataset shift. Axes and titles on the images carry the precise experimental settings.

### Encoder overlap vs distance

![Scalar encoder overlap vs distance (non-periodic): nearby values overlap; distant values do not.](test_images/scalar_encoder_overlap_vs_distance_not_periodic.png)

![Scalar encoder overlap vs distance (periodic): similarity repeats with period.](test_images/scalar_encoder_overlap_vs_distance_periodic.png)

![RDSE overlap vs distance: hashing introduces similarity noise across distances.](test_images/rdse_overlap_vs_distance.png)

### Spatial pooler: overlap vs distance (active columns)

![SP with scalar input field: overlap on active columns vs distance (some distant similarity noise).](test_images/spatial_pooler_active_col_overlap_vs_distance_with_scalar_input_field.png)

![SP with RDSE input field: stronger noise; some peaks exceed a rough 50% overlap guideline.](test_images/spatial_pooler_active_col_overlap_vs_distance_with_rdse_input_field.png)

### Spatial pooler: activation frequency distributions

![Activation frequency at epoch 0, random data, encoder bypassed: many columns participate.](test_images/Activation_Frequency_Distribution_with_random_data_zero_epoch_excluding_encoder.png)

![Activation frequency at epoch 49, random cells, encoder bypassed: emerging dominant columns (~10–15% activity).](test_images/Activation_Frequency_Distribution_with_random_cells_excluding_encoder_spatial_pooler.png)

![Activation frequency at epoch 49, random values with RDSE vs the scalar random run above: more “dead” columns; fewer columns ever active.](test_images/Activation_Frequency_Distribution_with_random_date_including_encoder_spatial_pooler.png)

![Activation frequency at epoch 49, sine wave + scalar (non-periodic): broad participation with dominant bands.](test_images/Activation_Frequency_Distribution_with_sine_wave_with_scalar_encoder_periodic_false_spatial_pooler.png)

![Activation frequency at epoch 49, sine wave + RDSE: many inactive columns and very dominant winners.](test_images/Activation_Frequency_Distribution_with_sine_wave_with_rdse_encoder_spatial_pooler.png)

### Noise robustness and continual learning

![SP noise robustness vs training epoch: tolerance to injected input noise improves with training.](test_images/spatial_pooler_noise_robustness.png)

![Synapse formation on two datasets: burst of new synapses on a disjoint dataset, then slower formation as the SP adapts.](test_images/synapse_formation_two_datasets.png)

---

## Comparison: HTM core Python bindings

Parallel runs against **HTM core Python bindings** for overlap, activation distributions (with and without boosting), and related settings. Use these alongside the section above to compare behavior, not as a performance benchmark.

![RDSE overlap vs distance (htm core bindings).](test_images/htm_core_python_bindings/rdse_overlap_vs_distance.png)

![Scalar encoder overlap vs distance, periodic (htm core bindings).](test_images/htm_core_python_bindings/Scalar_encoder_overlap_vs_distance_periodic.png)

![SP active-column overlap vs distance with RDSE (htm core bindings).](test_images/htm_core_python_bindings/sp_overlap_vs_distance_with_rdse.png)

![Activation frequency, random cells, encoder excluded (htm core bindings).](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_cells_excluding_encoder.png)

![Activation frequency, random cells, encoder excluded, high boosting (htm core bindings).](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_cells_excluding_encoder_with_100_boost.png)

![Activation frequency, random data, epoch 0, encoder excluded (htm core bindings).](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_data_zero_epoch_excluding_encoder.png)

![Activation frequency, random data, epoch 0, scalar encoder (htm core bindings).](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_data_zero_epoch_scalar_encoder.png)

![Activation frequency, random data, epoch 0, RDSE (htm core bindings).](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_data_zero_epoch_rdse.png)

![Activation frequency, random data, scalar encoder (htm core bindings).](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_data_once_with_scalar_encoder.png)

![Activation frequency, random data, scalar encoder + boosting (htm core bindings).](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_data_once_with_scalar_encoder_and_boost.png)

![Activation frequency, random data, RDSE (htm core bindings).](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_data_once_with_rdse.png)

![Activation frequency, random data, RDSE + boosting (htm core bindings).](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_data_once_with_rdse_and_boost.png)

![Activation frequency, sine wave, scalar non-periodic (htm core bindings).](test_images/htm_core_python_bindings/activation_frequency_distribution_with_sin_wave_with_scalar_encoder_periodic_false.png)

![Activation frequency, sine wave, scalar non-periodic + boosting (htm core bindings).](test_images/htm_core_python_bindings/activation_frequency_distribution_with_sin_wave_with_scalar_encoder_periodic_false_and_boosting.png)

![Activation frequency, sine wave, RDSE (htm core bindings).](test_images/htm_core_python_bindings/activation_frequency_distribution_with_sin_wave_with_rdse.png)

![Activation frequency, sine wave, RDSE + boosting (htm core bindings).](test_images/htm_core_python_bindings/activation_frequency_distribution_with_sin_wave_with_rdse_and_boosting.png)

## Citation

If you use this software in academic work, cite the **PSU Capstone** report or any public artifact your team publishes, and cite **Numenta / HTM** sources appropriate to the theory you rely on. Add concrete BibTeX or DOI links here when available.

## Acknowledgments

Developed as a **Pennsylvania State University (PSU) Capstone** project; contributors are listed in [`pyproject.toml`](pyproject.toml) authors metadata.

## License

No `LICENSE` file is present in this repository yet. Add one at the project root (for example MIT, Apache-2.0, or your institution’s preferred terms) before redistributing or publishing packages.
