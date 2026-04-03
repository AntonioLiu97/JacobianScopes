# Jacobian Scopes

**Jacobian Scopes** are a suite of tools for **token-level causal attribution** in large language models. 
They quantify how much each input token influences the model’s behavior at a chosen prediction.
This repository provides implementations and demos so you can run scopes on your own prompts.

![Temperature Scope on time-series data](./assets/fig2.png)

![Fisher translations](./assets/Fisher_translations.png)

## Try it

**Option A — Online (no installation).** Run an interactive demo in the browser on **[Hugging Face Spaces — Jacobian Scopes Demo](https://huggingface.co/spaces/Typony/JacobianScopes)**.

**Option B — Local notebooks.** After installation (read on for instructions), open and run these demonstration notebooks from the repository root:

- **[`Jacobian_Scopes_natural_language.ipynb`](./Jacobian_Scopes_natural_language.ipynb)** — attribution on natural-language prompts.
- **[`Jacobian_Scopes_time_series.ipynb`](./Jacobian_Scopes_time_series.ipynb)** — attribution on numerical time-series treated as tokens.

## Installation

Clone the repository and install the package in editable mode (from the repository root, where `pyproject.toml` lives):

```bash
git clone https://github.com/AntonioLiu97/JacobianScopes.git
cd JacobianScopes
pip install -e .
```

<!-- `pip` and PyPI normalize the **distribution** name to `jacobian-scopes` (lowercase, hyphenated) for packaging; the **Python import** is always `JacobianScopes`, matching the `src/JacobianScopes/` package directory. -->

In notebooks and scripts, import the library as:

```python
import JacobianScopes
```

## Paper

For method details, theory, and experiments, see:

**[Jacobian Scopes: token-level causal attributions in LLMs](https://arxiv.org/abs/2601.16407)** (arXiv:2601.16407).

## Repository structure

```
JacobianScopes/
│
├── src/JacobianScopes/              # Installable Python package
│   ├── __init__.py                  # Public API
│   ├── JacobianScopes.py            # Fisher, Temperature, and Semantic scope algorithms
│   ├── JacobianScopes_utils.py      # Forward pass construction and model utilities
│   └── plotter_utils.py             # Visualization utilities
│
├── Jacobian_Scopes_natural_language.ipynb   # Demo: natural language
├── Jacobian_Scopes_time_series.ipynb        # Demo: time series
│
├── assets/                          # Images for this README
│
├── paper/                           # Reproducibility for the paper
│   ├── benchmarks/                  # Large-scale evaluation scripts
│   ├── figures/                     # Notebooks that generate figures
│   ├── data/                        # Datasets and data processing
│   └── results/                     # Benchmark outputs (JSON)
│
├── pyproject.toml                   # Package metadata and dependencies
└── README.md
```

## Authors

- [Toni J.B. Liu](https://antonioliu97.github.io/About_Me.html), jl3499@cornell.edu
- [Baran Zadeoğlu](https://math.cornell.edu/baran-zadeoglu), bz333@cornell.edu
- [Raphaël Sarfati](https://raphaelsarfatixyz.wordpress.com/), raphael.sarfati@cornell.edu
- [Nicolas Boullé](https://nboulle.github.io/), nb690@cam.ac.uk
- [Christopher J. Earls](https://earls.cee.cornell.edu/people/), earls@cornell.edu

## BibTeX

```bibtex
@misc{liu2026jacobianscopestokenlevelcausal,
    title={Jacobian Scopes: token-level causal attributions in LLMs},
    author={Toni J. B. Liu and Baran Zadeoğlu and Nicolas Boullé and Raphaël Sarfati and Christopher J. Earls},
    year={2026},
    eprint={2601.16407},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2601.16407},
}
```
