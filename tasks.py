import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "signal_classification"
PYTHON_VERSION = "3.12"

# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )

@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)

# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py", echo=True, pty=not WINDOWS)

@task
def LinearMLP(ctx: Context) -> None:
    """Look at the LinearMLP model."""
    ctx.run(f"python src/{PROJECT_NAME}/model.py --config configs/models/modelv0_param1.yaml", echo=True, pty=not WINDOWS)

@task
def NonLinearMLP(ctx: Context) -> None:
    """Look at the NonLinearMLP model."""
    ctx.run(f"python src/{PROJECT_NAME}/model.py --config configs/models/modelv1_param1.yaml", echo=True, pty=not WINDOWS)

@task
def CNN(ctx: Context) -> None:
    """Look at the CNN model."""
    ctx.run(f"python src/{PROJECT_NAME}/model.py --config configs/models/modelv2_param1.yaml", echo=True, pty=not WINDOWS)

@task
def train_LinearMLP(ctx: Context) -> None:
    """Train model LinearMLP model using config."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py --config configs/models/modelv0_param1.yaml", echo=True, pty=not WINDOWS)

@task
def train_NonLinearMLP(ctx: Context) -> None:
    """Train model NonLinearMLP model using config."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py --config configs/models/modelv1_param1.yaml", echo=True, pty=not WINDOWS)

@task
def train_CNN(ctx: Context) -> None:
    """Train model CNN model using config."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py --config configs/models/modelv2_param1.yaml", echo=True, pty=not WINDOWS)

@task
def evaluate_LinearMLP(ctx: Context) -> None:
    """Evaluate LinearMLP model using config."""
    ctx.run(f"python src/{PROJECT_NAME}/evaluate.py --config configs/models/modelv0_param1.yaml", echo=True, pty=not WINDOWS)
@task
def evaluate_NonLinearMLP(ctx: Context) -> None:
    """Evaluate NonLinearMLP model using config."""
    ctx.run(f"python src/{PROJECT_NAME}/evaluate.py --config configs/models/modelv1_param1.yaml", echo=True, pty=not WINDOWS)

@task
def evaluate_CNN(ctx: Context) -> None:
    """Evaluate CNN model using config."""
    ctx.run(f"python src/{PROJECT_NAME}/evaluate.py --config configs/models/modelv2_param1.yaml", echo=True, pty=not WINDOWS)

@task
def show_predictions_LinearMLP(ctx: Context) -> None:
    """Show predictions for LinearMLP model using config."""
    ctx.run(f"python src/{PROJECT_NAME}/visualize.py --config configs/models/modelv0_param1.yaml", echo=True, pty=not WINDOWS)

@task
def show_predictions_NonLinearMLP(ctx: Context) -> None:
    """Show predictions for NonLinearMLP model using config."""
    ctx.run(f"python src/{PROJECT_NAME}/visualize.py --config configs/models/modelv1_param1.yaml", echo=True, pty=not WINDOWS)

@task
def show_predictions_CNN(ctx: Context) -> None:
    """Show predictions for CNN model using config."""
    ctx.run(f"python src/{PROJECT_NAME}/visualize.py --config configs/models/modelv2_param1.yaml", echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
