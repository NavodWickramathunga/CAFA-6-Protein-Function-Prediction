"""Kaggle competition downloader for CAFA-6.

Usage:
    python -m src.data.download --dest data/raw

Prereqs:
    - Kaggle CLI credentials at %USERPROFILE%\.kaggle\kaggle.json
    - Competition rules accepted on Kaggle.
"""
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import typer

COMPETITION = "cafa-6-protein-function-prediction"
app = typer.Typer()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@app.command()
def main(dest: str = typer.Option("data/raw", help="Download directory")) -> None:
    dest_path = Path(dest)
    ensure_dir(dest_path)

    api = KaggleApi()
    api.authenticate()

    typer.echo(f"Downloading {COMPETITION} to {dest_path}...")
    api.competition_download_files(COMPETITION, path=str(dest_path), quiet=False)
    typer.echo("Done. If you see a .zip, unzip it to the data folder.")


if __name__ == "__main__":
    app()
