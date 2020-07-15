##########################################################################################
#               not my own work! taken from                                              #
#  https://github.com/JonathanCauchi/PGN-to-JSON-Parser/blob/master/pgn_to_json.py       #
##########################################################################################

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import chess.pgn
import re
import sys
import os.path
from tqdm import tqdm
import pathlib
import logging
from datetime import datetime
import sys, traceback

log = logging.getLogger().error

for i in [1, 2]:
    dir_ = sys.argv[i]
    if not os.path.exists(dir_):
        raise Exception(dir_ + " not found")

max_games = int(sys.argv[3])

is_join = False
if len(sys.argv) == 5:
    if sys.argv[4] == "join":
        is_join = True


inp_dir = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])


def get_file_list(local_path):
    tree = os.walk(str(local_path))
    file_list = []
    out = []
    test = r".+pgn$"
    for i in tree:
        file_list = i[2]

    for name in file_list:
        if len(re.findall(test, name)):
            out.append(str(local_path / name))
    return out


def get_data(pgn_file, max_games):
    node = chess.pgn.read_game(pgn_file)
    error_counter = 0
    game_counter = 0
    while node is not None and game_counter <= max_games:
        game_counter += 1
        try:
            data = node.headers

            data["moves"] = []

            while node.variations:
                next_node = node.variation(0)
                data["moves"].append(
                    re.sub("\{.*?\}", "", node.board().san(next_node.move))
                )
                node = next_node

            out_dict = {}

            for key in data.keys():
                out_dict[key] = data.get(key)

            # log(data.get('Event'))
            node = chess.pgn.read_game(pgn_file)
            yield out_dict
        except:
            error_counter = error_counter + 1
            print("skipping {}".format(error_counter))
            node = chess.pgn.read_game(pgn_file)
            continue


def convert_file(file_path, max_games):
    file_name = file_path.name.replace(file_path.suffix, "") + ".json"
    log("convert file " + file_path.name)
    out_list = []
    try:
        json_file = open(str(out_dir / file_name), "w")
        pgn_file = open(str(file_path), encoding="utf-8-sig")  # changed encoding

        for count_d, data in tqdm(enumerate(get_data(pgn_file, max_games), start=0)):
            # log(file_path.name + " " + str(count_d))
            out_list.append(data)

        log(" save " + file_path.name)
        json.dump(out_list, json_file)
        json_file.close()
        log("done")
    except Exception as e:
        log(traceback.format_exc(10))
        log("ERROR file " + file_name + " not converted")


def create_join_file(file_list, max_games):
    log(" create_join_file ")
    name = str(out_dir / "join_data.json")
    open(name, "w").close()
    json_file = open(str(out_dir / "join_data.json"), "a")
    json_file.write("[")
    for count_f, file in enumerate(file_list, start=0):
        pgn_file = open(file, encoding="ISO-8859-1")
        for count_d, data in tqdm(enumerate(get_data(pgn_file, max_games), start=0)):
            # log(str(count_f) + " " + str(count_d))
            if count_f or count_d:
                json_file.write(",")
            data_str = json.dumps(data)
            json_file.write(data_str)
        log(pathlib.Path(file).name)
    json_file.write("]")
    json_file.close()


file_list = get_file_list(inp_dir)

start_time = datetime.now()
if not is_join:
    for file in file_list:
        convert_file(pathlib.Path(file), max_games)
else:
    create_join_file(file_list, max_games)

end_time = datetime.now()
log("time " + str(end_time - start_time))

