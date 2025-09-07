from __future__ import annotations

from pathlib import Path

import typer

from .dataset import generate_synthetic_dataset, save_dataset_csv
from .evaluate import main as eval_main
from .train import main as train_main
from .explain import main as explain_main


app = typer.Typer(add_completion=False, help="Perovskite stability ML CLI")


@app.command()
def make_data(n: int = 2500, seed: int = 42, out: str = "data/synthetic/abx3_synthetic.csv") -> None:
    df = generate_synthetic_dataset(n=n, seed=seed)
    save_dataset_csv(df, Path(out))
    typer.echo(f"Wrote {len(df)} rows to {out}")


@app.command()
def train(config: str):
    import sys

    sys.argv = ["perostab.train", "--config", config]
    train_main()


@app.command()
def evaluate(model_path: str, data_csv: str = "data/synthetic/abx3_synthetic.csv"):
    import sys

    sys.argv = ["perostab.evaluate", "--model-path", model_path, "--data-csv", data_csv]
    eval_main()


@app.command()
def explain(model_path: str, data_csv: str = "data/synthetic/abx3_synthetic.csv", n_samples: int = 500):
    import sys

    sys.argv = [
        "perostab.explain",
        "--model-path",
        model_path,
        "--data-csv",
        data_csv,
        "--n-samples",
        str(n_samples),
    ]
    explain_main()


def main():
    app()


if __name__ == "__main__":
    main()

