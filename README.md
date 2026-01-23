# Jacobian Scopes
![Temperature Scope on time-series data](./figures/fig2.png)

<!-- ![Sesmantic Scope liberal_conservative](./figures/liberal_conservative.png) -->

![Fisher translations](./figures/Fisher_translations.png)

## Overview
This repository contains interactive demonstrations for the paper:

    Jacobian Scopes: token-level causal attributions in LLMs

## Notebooks
- The time-series generator is adapted from [LLMs learn governing principles of dynamical systems, revealing an in-context neural scaling law](https://aclanthology.org/2024.emnlp-main.842/).
  You can generate all time series used in the paper (e.g., Brownian motion and Lorenz systems) using [`time_series_genrator/series_generator.ipynb`](./time_series_genrator/series_generator.ipynb).
- Temperature Scope + Semantic Scope notebooks (same core ideas, with different visualization pipeline adapted to linguistic/numerical inputs):
  - Natural language: [`Temperature_Semantic_Scopes_natural_language.ipynb`](./Temperature_Semantic_Scopes_natural_language.ipynb)
  - Time series: [`Temperature_Semantic_Scopes_time_series.ipynb`](./Temperature_Semantic_Scopes_time_series.ipynb)
- Path-integrated Semantic Scope, inspired by [Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a.html): [`Path_Integrated_Semantic_Scope.ipynb`](./Path_Integrated_Semantic_Scope.ipynb)
- Fisher + Temperature (for side-by-side comparison): [`Fisher_Temperature_Scopes.ipynb`](./Fisher_Temperature_Scopes.ipynb)

## Authors

- [Toni J.B. Liu](https://antonioliu97.github.io/About_Me.html), jl3499@cornell.edu
- [Baran Zadeoğlu](https://math.cornell.edu/baran-zadeoglu),bz333@cornell.edu
- [Raphaël Sarfati](https://raphaelsarfatixyz.wordpress.com/), raphael.sarfati@cornell.edu
- [Nicolas Boullé](https://nboulle.github.io/), nb690@cam.ac.uk
- [Christopher J. Earls](https://earls.cee.cornell.edu/people/), earls@cornell.edu
