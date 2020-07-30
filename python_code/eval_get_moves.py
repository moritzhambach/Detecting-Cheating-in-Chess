import pandas as pd
import click
import logging

logging.basicConfig(level=logging.INFO,)

LOGGER = logging.getLogger()


@click.command()
@click.option("--input-path", help="filename of single input file", required=True)
@click.option("--output-path", help="where to save result (as parquet)", required=True)
def main(input_path, output_path):
    df = pd.read_json(input_path)
    df["opponentIsComp"] = -1  # column is expected by next stage
    df[["moves", "opponentIsComp"]].to_parquet(output_path)


if __name__ == "__main__":
    main()
