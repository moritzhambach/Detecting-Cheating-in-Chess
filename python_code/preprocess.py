import pandas as pd
import click
import logging

logging.basicConfig(level=logging.INFO,)

LOGGER = logging.getLogger()


def loadData(file_list):
    df = pd.DataFrame()
    for file in file_list:
        data = pd.read_json(file)
        df = df.append(data, ignore_index=True)
    return df


@click.command()
@click.option(
    "--input-paths", help="filenames of input files, separated by comma", required=True
)
@click.option("--output-path", help="where to save result (as parquet)", required=True)
def main(input_paths, output_path):
    file_list = input_paths.split(",")
    n_files = len(file_list)
    LOGGER.info("found {} files".format(n_files))
    df = loadData(file_list)
    df.to_parquet(output_path)


if __name__ == "__main__":
    main()
