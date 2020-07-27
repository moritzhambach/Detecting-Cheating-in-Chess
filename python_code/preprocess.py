import pandas as pd
import click
import logging
import json

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


def prefilterGames(df, params, human_color):
    """we want games where the human player (with color human_color) lost, and the opponent is 50/50 human or computer"""
    min_game_length = params["plymax"] # need at least as many moves as will be used in algorithm
    max_game_length = params["max_game_length"]
    df = df[
        (df.PlyCount > min_game_length) & (df.PlyCount < max_game_length)
    ]  # restrict game lengths. First moves are irrelevant as they can be memorized.
    
    # choose timecontrol. Very short games are weird (and hard to use engines on due to computation time)
    df = df[df["TimeControl"].isin(params["timecontrols"])]

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
@click.option("--params-path", help="path to config file", required=True)
@click.option("--output-path", help="where to save result (as parquet)", required=True)
@click.option(
    "--human-color", help="Black or White, what was the human playing", required=True
)
def main(input_paths, output_path, params_path, human_color):
    with open(params_path) as f:
        params = json.load(f)
    file_list = input_paths.split(",")
    n_files = len(file_list)
    LOGGER.info("found {} files".format(n_files))
    df = loadData(file_list)
    df = prefilterGames(df, params, human_color)

    LOGGER.info("number of games after preprocessing: {}".format(df.shape[0]))
    df.to_parquet(output_path)


if __name__ == "__main__":
    main()
