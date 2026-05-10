# CNN for CIFAR-10

This project includes a ResNet18 implementation for image classification on the CIFAR-10 dataset, along with training scripts, evaluation metrics, and a structured project layout. The `REPORT.md` file contains detailed results and analysis of different training runs, while this `README.md` provides an overview of the project structure and quickstart instructions.

## Architecture

- ResNet18 for image classification
- CIFAR-10 dataset
- Training scripts and evaluation metrics

## Project structure

```
.
├── config.yaml
├── data
│   ├── processed
│   └── raw
├── Makefile
├── outputs
│   ├── figures
│   ├── models
│   └── reports
├── pyproject.toml
├── README.md
├── REPORT.md
├── requirements.txt
├── scripts
│   ├── 00_explore_data.py
│   ├── 01_train.py
│   ├── 02_compare_runs.py
│   └── __init__.py
└── src
    ├── cnn_cifar_10
    │   ├── config
    │   │   ├── __init__.py
    │   │   ├── paths.py
    │   │   └── read_yaml.py
    │   ├── data
    │   │   ├── augmentation.py
    │   │   ├── __init__.py
    │   │   ├── load_data.py
    │   │   └── preprocessing.py
    │   ├── evaluation
    │   │   ├── artifacts.py
    │   │   ├── compare.py
    │   │   ├── __init__.py
    │   │   └── plots.py
    │   ├── __init__.py
    │   ├── io
    │   │   ├── __init__.py
    │   │   └── run_id.py
    │   ├── models
    │   │   ├── __init__.py
    │   │   └── resnet.py
    │   └── training
    │       ├── evaluate.py
    │       ├── __init__.py
    │       └── trainer.py
```

## Quickstart

**1. Install dependencies**

```bash
make setup
```

**2. Configure training**

Edit `config.yaml` to adjust hyperparameters:

```yaml
training:
  batch_size: 128
  num_epochs: 15
  learning_rate: 0.001
model:
  architecture: resnet18
  pretrained: true
  freeze_backbone: true
  input_size: 224
```

**3. Train**

```bash
make train
```

The dataset is downloaded automatically on first run. Artifacts are saved in `outputs/`.

**4. Run tests**

```bash
make test
```

## All Makefile commands

| Command        | Description                                               |
| -------------- | --------------------------------------------------------- |
| `make help`    | Show all Makefile commands                                |
| `make setup`   | Create `.venv` and install all dependencies               |
| `make explore` | Run the data exploration script                           |
| `make train`   | Run the training script                                   |
| `make compare` | Compare different training runs using all metrics         |
| `make lint`    | Check and format code with ruff                           |
| `make test`    | Run the test suite with pytest                            |
| `make clean`   | Remove `__pycache__`, pytest/ruff caches, and all outputs |

## Outputs

Each run generates a unique ID (timestamp-based) and saves:

| File                                            | Description              |
| ----------------------------------------------- | ------------------------ |
| `outputs/models/model_<run_id>.pth`             | Model weights            |
| `outputs/models/metrics_<run_id>.json`          | All metrics              |
| `outputs/figures/training_curves_<run_id>.png`  | Loss and accuracy curves |
| `outputs/reports/comparison_report_<run_id>.md` | Comparison report        |
