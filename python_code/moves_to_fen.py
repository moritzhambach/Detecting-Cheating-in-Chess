import pandas as pd
import click
import logging
import numpy as np
import pgn_to_fen
import chess
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO,)

LOGGER = logging.getLogger()


def getFenArray(fen):
    fen = fen.split(" ")[0]
    fen = (
        fen.replace("1", "0")
        .replace("2", "00")
        .replace("3", "000")
        .replace("4", "0000")
        .replace("5", "00000")
        .replace("6", "000000")
        .replace("7", "0000000")
        .replace("8", "00000000")
    )
    fen = list(fen.replace("/", ""))
    fenArray = np.array(fen)
    fenArray = np.reshape(fenArray, [8, 8])
    return fenArray


def movesToFenList(movesList):
    """given a list of moves, return a list of standard fen notations"""
    fenlist = []
    pgnConverter = pgn_to_fen.PgnToFen()
    pgnConverter.resetBoard()
    try:
        for move in movesList:
            pgnConverter.move(str(move))
            fen = pgnConverter.getFullFen()
            fenlist.append(fen.split(" ")[0])
        return fenlist
    except Exception:
        LOGGER.info("can not create fen")


def fenList_to_fenArray(fenList):
    """turn standard fen notation into a fen version with zeros for empty squares 
    (e.g. "00000000" instead of "8" for eight free squares)"""
    fenArrayList = []
    for fen in fenList:
        fenArray = getFenArray(fen)
        fenArrayList.append(fenArray)

    fenArrayOverTime = np.stack(fenArrayList, axis=0)
    return fenArrayOverTime


def getAttacksPerPiece(fen):
    baseBoard = chess.BaseBoard(fen)
    attacks_list = []
    for k in range(64):  # 0 = A1, 1 = A2, etc
        piece = baseBoard.piece_at(k)
        if piece:
            attacks = baseBoard.attacks(
                k
            ).tolist()  # 64 bools, true if square is attacked from piece in square k
            attacks_sparse = [j for j, x in enumerate(attacks) if x]
            attacks_list.append((str(piece), attacks_sparse))

    return attacks_list


def getAttacksByPiecetype(fen):
    attacks_by_piece = getAttacksPerPiece(fen)
    pieceList = ("P", "R", "N", "B", "Q", "K", "p", "r", "n", "b", "q", "k")
    attacks_by_pieceType = {}
    for piece in pieceList:
        attacked_squares = [
            tup[1] for tup in attacks_by_piece if tup[0] == piece
        ]  # list of all squares attacked by all pieces of this type (for example all white pawns)
        attacked_squares = sum(attacked_squares, [])  # flatten
        attacks_by_pieceType[piece] = set(attacked_squares)  # ignoring double attacks
    return attacks_by_pieceType


def getAttackTensor(attacks_by_pieceType):
    pieceList = ("P", "R", "N", "B", "Q", "K", "p", "r", "n", "b", "q", "k")
    output_array = np.zeros((12, 8, 8))
    for channel, piece in enumerate(pieceList):
        attacksList = attacks_by_pieceType[piece]
        for pos in attacksList:
            pos_x = pos % 8
            pos_y = 7 - pos // 8
            output_array[channel, pos_y, pos_x] = 1
    return output_array


def getAttacksTensorOverTime(fenList):
    """returns tensor of shape (time, channel, row, col) describing a single game,
    where the channels represent the 12 piece types and a value of 1 means this piece
    attacks this square at this time"""
    res = np.zeros((len(fenList), 12, 8, 8))
    for j, fen in enumerate(fenList):
        attacks_by_pieceType = getAttacksByPiecetype(fen)
        res[j] = getAttackTensor(attacks_by_pieceType)
    return res.astype(int)


def getFenPerChannel(input_array):
    """ takes a fen (with strings describing the pieces on the field) and expands it
    in an additional dimension, basically one-hot-encoding the pieces"""
    pieceList = ("P", "R", "N", "B", "Q", "K", "p", "r", "n", "b", "q", "k")
    res = np.zeros((input_array.shape[0], 12, 8, 8))  # time, channel, row, column

    if input_array is None:
        return res
    for k, piece in enumerate(pieceList):
        mask = input_array == piece
        res[:, k, :, :] = mask
    return res.astype(int)


@click.command()
@click.option(
    "--input-path", help="expects parquet file", required=True, type=click.Path()
)
@click.option(
    "--params-path", help="configuration params",
)
@click.option("--output-path", help="where to save result", required=True)
@click.option("--output-path-labels", help="where to save labels", required=True)
@click.option(
    "--output-path-attacks", help="where to save attack tensors", required=True
)
def main(
    input_path, output_path, output_path_labels, output_path_attacks, params_path,
):
    df = pd.read_parquet(input_path)
    with open(params_path) as f:
        params = json.load(f)
    min_ply_to_consider = params["plymin"]
    max_ply_to_consider = params["plymax"]

    resList = []
    labelList = []
    attacksList = []
    for moveList, label in tqdm(zip(df["moves"], df["opponentIsComp"])):
        # loop over games, TODO: optimize
        fenList = movesToFenList(moveList)
        failCounter = 0
        if not fenList:
            failCounter += 1
            continue
        fenList = fenList[
            min_ply_to_consider:max_ply_to_consider
        ]  # only keep positions of the middle game!
        fenArray = fenList_to_fenArray(fenList)
        fen_per_channel = getFenPerChannel(fenArray)
        attacksTensor = getAttacksTensorOverTime(fenList)

        if np.count_nonzero(fen_per_channel) > 0:
            resList.append(fen_per_channel)
            labelList.append(label)
            attacksList.append(attacksTensor)
    LOGGER.info(f"failed games: {failCounter}")
    res = np.stack(resList, axis=0).astype(int)
    labels = np.array(labelList).astype(int)
    attacks = np.stack(attacksList, axis=0).astype(int)

    LOGGER.info(
        f"output shape: {res.shape}, labels shape: {labels.shape}, attacks shape: {attacks.shape}"
    )
    np.savez_compressed(output_path, res)
    np.savez_compressed(output_path_labels, labels)
    np.savez_compressed(output_path_attacks, attacks)


if __name__ == "__main__":
    main()
