import pandas as pd
import click
import logging
import json

logging.basicConfig(level=logging.INFO,)

LOGGER = logging.getLogger()


def checkParameters(df, human_color, params):
    if not len(df) == 1:
        LOGGER.info("evaluating single games only, check your input!")
        raise ValueError
    else:
        LOGGER.info("check 1 passed")
    if not (
        (human_color == "White" and df.Result[0] == "0-1")
        or (human_color == "Black" and df.Result[0] == "1-0")
    ):
        LOGGER.info("you did not lose the game, why do you care if engine was used?")
        LOGGER.info("model was not trained on won games, might be incorrect")
    else:
        LOGGER.info("check 2 passed")
    if not (len(df.moves[0]) > params["plymin"]):
        LOGGER.info(
            "game is too short, can not distinguish engine use from opening knowledge"
        )
        raise ValueError
    else:
        LOGGER.info("check 3 passed")
    if not (df.TimeControl[0] in params["timecontrols"]):
        LOGGER.info(
            "game has a different time control than the model knows about, results might be incorrect"
        )
    else:
        LOGGER.info("check 4 passed")


@click.command()
@click.option("--input-path", help="filename of single input file", required=True)
@click.option("--output-path", help="where to save result (as parquet)", required=True)
@click.option("--params-path", help="preprocessing parameters", required=True)
@click.option(
    "--human-color", help="which color were you playing? Black or White?", required=True
)
def main(input_path, output_path, human_color, params_path):
    df = pd.read_json(input_path)
    LOGGER.info(df.columns)
    with open(params_path) as f:
        params = json.load(f)
    LOGGER.info(params)
    checkParameters(df, human_color, params)
    df["opponentIsComp"] = -1  # column is expected by next stage
    df[["moves", "opponentIsComp"]].to_parquet(output_path)


if __name__ == "__main__":
    main()
