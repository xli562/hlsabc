# HLSABC

## Overview

Estimating high-level metrics for high-level synthesis (HLS) with large language models (LLMs).

## Features

- **Benchmarks**: Polybench and ABC benchmarks for HLS evaluation.
- **Utilities**: Tools for logging, performance feedback, and estimations.
- **Tests**: Comprehensive test suite for validation.

## Repository Structure

- `benchmarks/`: Contains benchmark programs for HLS evaluation.
- `input/`: Input files for HLS designs.
- `logs/`: Log files generated during execution.
- `scratch/`: Temporary files and scripts.
- `scripts/`: Helper scripts for testing and execution.
- `tests/`: Test cases for utilities and benchmarks.
- `utils/`: Utility modules for logging, benchmarking, and estimations.
- `vivado/`: Vivado project files and reports.

## Getting Started

- Clone the repository:

```bash
git clone https://github.com/xli562/hlsllm.git
```

- Install Allo

- Install dependencies:

```bash
conda env create -n hlsabc -f environment.yml
pip install -r requirements.txt
```

- Run tests:

```bash
./scripts/pytest --parallel
```

## Testing

Some tests require certain files in `tests/resource`.

### Running tests

```bash
./scripts/pytest --parallel
```

If `--parallel` is specified, all tests not marked by `@pytest.mark.serial` will run in parallel. Then, all tests marked as serial will run in serial.

If `--parallel` is not specified, all tests will run in serial.

### Configuring tests

In `pytest.ini`,

- `testpaths` specifies the directory / directories to collect tests from.
- `markers` defines pytest markers such as `serial` or `slow`.
- `addopts` can specify which specific files or functions should be run.

For example,

```pytest
testpaths = 
    tests

markers =
    serial: tests that need to be run serially
    slow: slow tests (runs csynth or LLM)

addopts = 
    --color=yes
    tests/test_llm.py::test_simple
```

runs `test_simple` in `tests/test_llm.py`.
