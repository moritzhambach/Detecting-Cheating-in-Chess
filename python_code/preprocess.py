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


def balanceEngineRatio(df):
    df_comp = df[df.opponentIsComp == 1.0]
    df_human = df[df.opponentIsComp == 0.0]
    n_min = min(len(df_comp), len(df_human))
    df = pd.concat([df_comp[:n_min], df_human[:n_min]])
    df = df.sample(frac=1)  # reshuffle
    df = df.reset_index(drop=True)
    return df


def prefilterGames(df, plymin, plymax, human_color):
    """we want games where the human player (with color human_color) lost, and the opponent is 50/50 human or computer"""
    df = df[
        (df.PlyCount > plymin) & (df.PlyCount < plymax)
    ]  # restrict game lengths. First moves are irrelevant as they can be memorized.
    df = df[
        (df.TimeControl == "300+0")
        | (df.TimeControl == "600+0")
        | (df.TimeControl == "900+0")
        | (df.TimeControl == "900+5")
        | (df.TimeControl == "900+10")
        | (df.TimeControl == "1200+0")
    ]  # choose timecontrol in sec. Very short games are weird (and hard to use engines on due to computation time)

    df.loc[
        df["WhiteIsComp"].isnull(), "WhiteIsComp"
    ] = 0.0  # field is null for human vs human games
    df.loc[df["WhiteIsComp"] == "Yes", "WhiteIsComp"] = 1.0
    df.loc[df["BlackIsComp"].isnull(), "BlackIsComp"] = 0.0
    df.loc[df["BlackIsComp"] == "Yes", "BlackIsComp"] = 1.0

    if human_color == "White":
        df = df[df["WhiteIsComp"] == 0.0]
        df = df[
            df.Result == "0-1"
        ]  # only interested in games where the human player lost
        df = df.rename(columns={"BlackIsComp": "opponentIsComp"})
    elif human_color == "Black":
        df = df[df["BlackIsComp"] == 0.0]
        df = df[df.Result == "1-0"]
        df = df.rename(columns={"WhiteIsComp": "opponentIsComp"})

    df = df[["opponentIsComp", "moves"]]
    df = balanceEngineRatio(df)
    return df


@click.command()
@click.option(
    "--input-paths", help="filenames of input files, separated by comma", required=True
)
@click.option("--output-path", help="where to save result (as parquet)", required=True)
@click.option(
    "--human-color", help="Black or White, what was the human playing", required=True
)
@click.option("--plymin", help="min number of half moves", required=True, type=int)
@click.option("--plymax", help="max number of half moves", required=True, type=int)
def main(input_paths, output_path, plymin, plymax, human_color):
    file_list = input_paths.split(",")
    n_files = len(file_list)
    LOGGER.info("found {} files".format(n_files))
    df = loadData(file_list)
    df = prefilterGames(df, plymin, plymax, human_color)

    LOGGER.info("number of games after preprocessing: {}".format(df.shape[0]))
    df.to_parquet(output_path)


if __name__ == "__main__":
    main()
