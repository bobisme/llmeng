from pathlib import Path
from datetime import datetime as dt

import typer
from loguru import logger

from pipelines.digital_data_etl import digital_data_etl

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


if __name__ == "__main__":
    app()
