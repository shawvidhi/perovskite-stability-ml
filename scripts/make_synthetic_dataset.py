#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import typer

from perostab.dataset import generate_synthetic_dataset, save_dataset_csv


def main(n: int = 2500, seed: int = 42, out: str = "data/synthetic/abx3_synthetic.csv"):
    df = generate_synthetic_dataset(n=n, seed=seed)
    save_dataset_csv(df, Path(out))
    typer.echo(f"Wrote dataset with {len(df)} rows to {out}")


if __name__ == "__main__":
    typer.run(main)
