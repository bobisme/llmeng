from pathlib import Path
from datetime import datetime as dt

import typer
from loguru import logger

from pipelines.digital_data_etl import digital_data_etl
from pipelines.feature_engineering import feature_engineering

app = typer.Typer()

root_dir = Path(__file__).resolve().parent.parent


@app.command()
def run_etl(etl_config_filename: str):
    logger.info("Running ETL")
    run_args_etl = {}
    config_path = str(root_dir / "configs" / etl_config_filename)
    run_name = f"digital_data_etl_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    digital_data_etl.with_options(config_path=config_path, run_name=run_name)(
        **run_args_etl
    )
    logger.info("Done")


@app.command()
def run_feature_engineering(
    no_cache: bool = False,
    config_path: Path = root_dir / "configs" / "feature_engineering.yaml",
    run_name: str | None = None,
):
    run_args_fe = {}
    pipeline_args = {
        "enable_cache": not no_cache,
        "config_path": config_path,
        "run_name": (
            run_name
            or f"feature_engineering_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        ),
    }
    feature_engineering.with_options(**pipeline_args)(**run_args_fe)


if __name__ == "__main__":
    app()
