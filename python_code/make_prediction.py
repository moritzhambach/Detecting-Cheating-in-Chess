import click
import logging
import numpy as np
import tensorflow as tf
import h5py


logging.basicConfig(level=logging.INFO,)

LOGGER = logging.getLogger()


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
    "--human-player-color",
    help="Black or White, what did the human play",
    required=True,
)
def main(
    input_path,
    input_path_attacks,
    input_path_model_black,
    input_path_model_white,
    human_player_color,
):
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
