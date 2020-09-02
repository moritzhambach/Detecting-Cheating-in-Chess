# Detecting-Cheating-in-Chess
Can chess engine use be detected just from the moves on the board (but no chess engine)? Let's try it out, using a CNN - LSTM architecture and other architectures.
- [Usage](#Usage)
- [Why is cheating in chess hard to detect](#Why-is-cheating-in-chess-hard-to-detect)
- [Getting the data](#Getting-the-data)
- [Preprocessing](#Preprocessing)
- [Visualisation as heatmaps](#Visualisation-as-heatmaps)
- [CNN LSTM](#CNN-LSTM)




## Usage
### installation
clone repo, then from root of project: `conda create -f environment.yml` , `activate chess-classifier`
### classify a game
* get the pgn notation, save it under data/raw_data/evaluation/eval.pgn
* run `dvc repro dvc-stages/eval_moves_to_fen.dvc && python python_code/make_prediction.py --human-player-color White` (or Black if you were playing black pieces)
### train with your own data
* get pgn data, put it into folders data/raw_data/<year>, make sure the preprocess stages have the correct input path variable, then run `dvc repro dvc-stages/train_CNN_LSTM_black_human.dvc`. Might take a while for a lot of data. If happy with the results, put trained model into models/best_model_black_human.h5 (or white), and start predicting.

### reproducibility
* I use data versioning control (https://dvc.org/) for a reproducible pipeline from data ingestion to preprocessing and training.

## Why is cheating in chess hard to detect
Online chess suffers from the problem that the opponent could easily enter the moves into a chess engine on their smartphone and win easily.
Chess websites try to detect this by running an engine theirselves and comparing the moves played to the suggestions. Sophisticated cheaters could circumvent this by randomly choosing moves that are further down the list of sugestion list, or play a bad move once in a while, since they will win anyways. Also, with new chess engines based on neural networks (like Deepminds "Alpha Zero" or open source "Leela Chess Zero"), the comparison of chess moves might need to be done for several engines. Another try to catch cheaters currently is to analyze timing between moves, but future engines could not require much time to calculate and the "natural" waiting times can be added to fool the detection tool.

Let's approach the problem from another side. Experienced chess players can often detect a cheater when he plays "unintuitive" moves,
the kind that a human wouldn't ever naturally come up with. This hints that there might be hidden patterns in human chess, which could be 
distinguished from the way a computer plays (non-principled, just play whatever works out best 20-30 moves into the future).

Enter machine learning, specifically Deep Learning. Since the problem is both spatial (8x8 board) and temporal (board changes each move),
my intuitive approach is using convolutional neural networks (CNN) which feed their results into recurrent neural networks (RNN), specifically
a type called Long-Short-Term Memory (LSTM). But let's start from the beginning:

## Getting the data

millions of chess games are easily available online, and are usually labeled whether a computer or human played each side. I dowloaded close to 1 million games from https://www.ficsgames.org/ . The data comes in pgn format, which I first convert into JSON (using https://github.com/Assios/pgn-to-json ) and then read into pandas dataframe for preprocessing. The moves themselves are converted into board states (with help of https://github.com/niklasf/python-chess), meaning (channel x 8 x 8) arrays of 1s and 0s, where a 1 means a piece exists at this position, and the channel determines the piece type (white knight, black Queen, etc). As there are 6 pieces per color (pawns, knights, bishops, rooks, Queen, King), we have 12 channels. The board state per move is then stacked onto each other to create a tensor of shape (time, channel, 8, 8 ) for each game. I also get another 12 channels with the fields that each piece can currently attack, although this does not seem to help too much.

## Preprocessing

1.) only take games where the human player lost (who cares about engines if you at least draw)
2.) opponent is 50/50 human or computer
3.) select games lengths between 20 and 100 ply, and only train and evaluate model with from 20 to 40 ply. The first 10 moves could be memorized, so detecting engines here does not make sense.

Currently left with about 90k games for each case (human plays black or white).

## Visualisation as heatmaps
Below we see heatmaps (average square occupation over the game), averaged over a thousand games each (here including the whole game, also the opening). There seems to be slight differences in black human vs black computer heatmaps, but in first baseline attempts wasn't able to get more than 60% test accuracy when using only heatmaps as training data for classification. More heatmaps can be seen in the data exploration notebook.

black pawns, black is computer:

![alt text](https://user-images.githubusercontent.com/33765868/43685360-05665d3e-98b2-11e8-80d3-7586e53cdc1e.png)

black pawns, black is human:

![alt text](https://user-images.githubusercontent.com/33765868/43685394-8e200774-98b2-11e8-88b6-e95bfd5b7ade.png)


## CNN LSTM
(work in progress)
Using the TimeDistributed wrapper on Conv2D layers allows easy setup of my network. The (channelx8x8) maps of each time step fist undergo 3 convolutional layers of kernel size 3x3 without padding, reducing the size to (filter x 2 x 2), are then flattened and fed into LSTM neurons, followed by a Dense (fully connected layer). The currently best result is 80% accuracy, see below. It is still overfitting, although dropout is applied. Will add more data soon.

![alt text](https://user-images.githubusercontent.com/33765868/43685326-382504ce-98b1-11e8-8564-a89dd4d4c57a.png)
