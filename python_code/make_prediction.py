import click
import logging
import numpy as np
import tensorflow as tf
import h5py
import json
import pandas as pd


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
@click.option(
    "--input-path-attacks",
    help="input array of training data (attacked squares)",
    required=True,
    default="data/preprocessed/fen_eval_attacks.npy.npz",
)
@click.option(
    "--input-path",
    help="input array of training data (positions)",
    required=True,
    default="data/preprocessed/fen_eval.npy.npz",
)
@click.option(
    "--input-path-model-black",
    help="path to load best model, if human played black",
    required=True,
    default="data/models/best_model_black_human.h5",
)
@click.option(
    "--input-path-model-white",
    help="path to load best model, if human played white",
    required=True,
    default="data/models/best_model_white_human.h5",
)
@click.option(
    "--params-path", required=True, default="data/configs/preprocess_params.json",
)
@click.option(
    "--path-json-data", required=True, default="data/raw_data/json/evaluation/eval.json"
)
@click.option(
    "--human-player-color",
    help="Black or White, what did the human play",
    required=True,
)
def main(
    input_path,
    path_json_data,
    input_path_attacks,
    input_path_model_black,
    input_path_model_white,
    human_player_color,
    params_path,
):
    with open(params_path) as f:
        params = json.load(f)
    df = pd.read_json(path_json_data)
    checkParameters(df, human_player_color, params)

    data_positions = np.load(input_path)["arr_0"]
    data_attacks = np.load(input_path_attacks)["arr_0"]
    data = np.concatenate((data_positions, data_attacks), axis=2)
    if human_player_color == "White":
        opponent = "Black"
        model_path = input_path_model_white
    elif human_player_color == "Black":
        opponent = "White"
        model_path = input_path_model_black
    else:
        raise ValueError("please specify the color you played, Black or White")
    model = tf.keras.models.load_model(model_path)
    try:
        res = model.predict(data)[0][1]
    except:
        data = data.reshape(
            data.shape[0], data.shape[1], -1
        )  # flatten for use with non-CNN model
        res = model.predict(data)[0][1]

    LOGGER.info(
        f" Probability that your opponent ({opponent} Player) is using an engine: {res}"
    )


if __name__ == "__main__":
    main()
