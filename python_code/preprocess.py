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


def prefilterGames(df, plymin, plymax):
    df.loc[df["WhiteIsComp"].isnull(), "WhiteIsComp"] = 0.0
    df.loc[df["WhiteIsComp"] == "Yes", "WhiteIsComp"] = 1.0
    df.loc[df["BlackIsComp"].isnull(), "BlackIsComp"] = 0.0
    df.loc[df["BlackIsComp"] == "Yes", "BlackIsComp"] = 1.0

    df = df[(df.PlyCount > plymin) & (df.PlyCount < plymax)]  # restrict game lengths
    df = df[(df.Result == "1-0") | (df.Result == "0-1")]  # no draws
    df = df[
        (df.TimeControl == "300+0")
        | (df.TimeControl == "600+0")
        | (df.TimeControl == "900+0")
    ]  # choose timecontrol in sec
    df = df[df["WhiteIsComp"] == 0.0]  # only take human players as white,for now
    return df


def balanceWinrates(df):
    # balance black player label and win rate (so the algorithm doesn't just detect game winners)
    dfBlackcomputer_winning = df[(df.BlackIsComp == 1.0) & (df.Result == "0-1")]
    dfBlackcomputer_losing = df[(df.BlackIsComp == 1.0) & (df.Result == "1-0")]
    dfBlackhuman_winning = df[(df.BlackIsComp == 0.0) & (df.Result == "0-1")]
    dfBlackhuman_losing = df[(df.BlackIsComp == 0.0) & (df.Result == "1-0")]
    n_min = min(
        [
            len(dfBlackcomputer_winning),
            len(dfBlackcomputer_losing),
            len(dfBlackhuman_winning),
            len(dfBlackhuman_losing),
        ]
    )
    df = pd.concat(
        [
            dfBlackcomputer_winning[:n_min],
            dfBlackcomputer_losing[:n_min],
            dfBlackhuman_winning[:n_min],
            dfBlackhuman_losing[:n_min],
        ]
    )
    df = df.sample(frac=1)  # reshuffle
    df = df.reset_index(drop=True)

    return df


@click.command()
@click.option(
    "--input-paths", help="filenames of input files, separated by comma", required=True
)
@click.option("--output-path", help="where to save result (as parquet)", required=True)
@click.option("--plymin", help="min number of half moves", required=True, type=int)
@click.option("--plymax", help="max number of half moves", required=True, type=int)
def main(input_paths, output_path, plymin, plymax):
    file_list = input_paths.split(",")
    n_files = len(file_list)
    LOGGER.info("found {} files".format(n_files))
    df = loadData(file_list)
    df = prefilterGames(df, plymin, plymax)
    df = df.sample(frac=1)  # shuffle data
    df = balanceWinrates(df)

    df = df[["BlackIsComp", "WhiteIsComp", "moves"]]
    LOGGER.info("number of games after preprocessing: {}".format(df.shape[0]))
    df.to_parquet(output_path)


if __name__ == "__main__":
    main()
