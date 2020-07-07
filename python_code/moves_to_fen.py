import pandas as pd
import click
import logging
import numpy as np
import pgn_to_fen
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,)

LOGGER = logging.getLogger()


def moves_to_fen(moves):
    """turn pgn notation (describing the moves) into a version of the fen notation (describing the board),
    with zeros for empty squares instead of the standard notation (e.g. "00000000" instead of "8" for eight free squares)"""
    fenlist = []
    pgnConverter = pgn_to_fen.PgnToFen()
    pgnConverter.resetBoard()

    try:
        for move in moves:
            pgnConverter.move(str(move))
            fen = pgnConverter.getFullFen()
            fen = fen.split(" ")[0]
            fen = (
                fen.replace("1", "0")
                .replace("2", "00")
                .replace("3", "000")
                .replace("4", "0000")
            )
            fen = (
                fen.replace("5", "00000")
                .replace("6", "000000")
                .replace("7", "0000000")
                .replace("8", "00000000")
            )
            fen = list(fen.replace("/", ""))
            fen = np.array(fen)
            fen = np.reshape(fen, [8, 8])
            fenlist.append(fen)

        fenArray = np.stack(fenlist, axis=0)
        # only return positions of the middle game!
        return fenArray
    except Exception:
        LOGGER.info("can not create fen")


def getFenPerChannel(input_array, min_ply_to_consider, max_ply_to_consider):
    """ takes a fen (with strings describing the pieces on the field) and expands it in an additional dimension, basically one-hot-encoding the pieces"""
    pieceList = ("P", "R", "N", "B", "Q", "K", "p", "r", "n", "b", "q", "k")
    res = np.zeros(
        (max_ply_to_consider - min_ply_to_consider, 12, 8, 8)
    )  # time, channel, row, column

    if input_array is None:
        return res
    input_array = input_array[min_ply_to_consider:max_ply_to_consider, :, :]
    for k, piece in enumerate(pieceList):
        mask = input_array == piece
        res[:, k, :, :] = mask
    return res.astype(int)


@click.command()
@click.option(
    "--input-path", help="expects parquet file", required=True, type=click.Path()
)
@click.option(
    "--max-ply-to-consider",
    type=int,
    help="will drop everything after this many half moves (to keep a consistent length)",
)
@click.option(
    "--min-ply-to-consider",
    type=int,
    help="will drop everything before this many half moves (because it could be memorized)",
)
@click.option("--output-path", help="where to save result", required=True)
@click.option("--output-path-labels", help="where to save labels", required=True)
def main(
    input_path,
    output_path,
    output_path_labels,
    min_ply_to_consider,
    max_ply_to_consider,
):
    df = pd.read_parquet(input_path)
    resList = []
    labelList = []
    for move, label in tqdm(zip(df["moves"], df["opponentIsComp"])):
        fen = moves_to_fen(move)
        fen_per_channel = getFenPerChannel(
            fen, min_ply_to_consider, max_ply_to_consider
        )
        if np.count_nonzero(fen_per_channel) > 0:
            resList.append(fen_per_channel)
            labelList.append(label)
    res = np.stack(resList, axis=0).astype(int)
    labels = np.array(labelList).astype(int)

    LOGGER.info(f"output shape: {res.shape}, labels shape: {labels.shape}")
    np.savez_compressed(output_path, res)
    np.savez_compressed(output_path_labels, labels)


if __name__ == "__main__":
    main()
