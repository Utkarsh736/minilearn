# Minilearn

A small machine learning library built from scratch to understand how core ML models work under the hood.

`minilearn` is inspired by the simplicity of the scikit-learn API, but the goal here is clarity over completeness. The project focuses on implementing classical machine learning algorithms in a clean, readable way so the math, data flow, and design decisions stay visible.

## Why this project exists

Most of the time, libraries like scikit-learn should be the practical choice. But when learning machine learning, there is a big difference between **using** a model and **understanding** how it is trained, how predictions are made, and what assumptions are built into the implementation.

This project exists to make that understanding concrete by:

- implementing models from scratch with NumPy
- keeping the API simple and familiar
- building things incrementally with tests
- treating the codebase as both a usable package and a study tool

## Project goals

The long-term aim is to grow `minilearn` into a lightweight, installable Python package that includes a small but meaningful set of classical ML tools.

Planned areas include:

- linear models
- preprocessing utilities
- evaluation metrics
- nearest neighbors methods
- naive Bayes models
- tree-based models
- clustering algorithms

The emphasis is on correctness, readability, and intuition rather than performance tuning or production-scale coverage.

## Installation

### From GitHub

```bash
pip install git+https://github.com/Utkarsh736/minilearn.git
```

### Local development

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
git clone https://github.com/Utkarsh736/minilearn.git
cd minilearn
uv sync
```

## Quick example

```python
import numpy as np
from minilearn import LinearRegression

X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([3.0, 5.0, 7.0, 9.0])

model = LinearRegression(lr=0.01, epochs=1000)
model.fit(X, y)

prediction = model.predict([[5.0]])
print(prediction)
```

## Development workflow

A simple workflow for working on the project:

```bash
uv sync
uv run pytest
uv build
```

## Project philosophy

A few principles guide the project:

- simple code is preferred over clever code
- explicit math is preferred over hidden abstraction
- tests are added early
- packaging matters from the start
- the library should remain easy to read end to end


## Notes

`minilearn` is best thought of as a compact ML codebase for building strong intuition while still being structured like a real Python package.

If the project becomes useful to others along the way, even better.
