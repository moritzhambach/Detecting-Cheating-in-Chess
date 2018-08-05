# Detecting-Cheating-in-Chess
Can chess engine use be detected just from the moves on the board? Let's try, using a CNN - LSTM architecture (and no chess engine).

Online chess suffers from the problem that the opponent could easily enter the moves into a chess engine on their smartphone and win easily.
Chess websites try to detect this by running an engine theirselves and comparing the moves played to the suggestions. Sophisticated cheaters could circumvent this by randomly choosing moves that are further down the list of sugestion list, or play a bad move once in a while, since they will win anyways. Also, with new chess engines based on neural networks (like Deepminds "Alpha Zero" or open source "Leela Chess Zero"), the comparison of chess moves might need to be done for several engines.

Let's approach the problem from another side. Experienced chess players can often detect a cheater when he plays "unintuitive" moves,
the kind that a human wouldn't ever naturally come up with. This hints that there might be hidden patterns in human chess, which could be 
distinguished from the way a computer plays (non-principled, just play whatever works out best 20-30 moves into the future).

Enter machine learning, specifically Deep Learning. Since the problem is both spatial (8x8 board) and temporal (board changes each move),
my intuitive approach is using convolutional neural networks (CNN) which feed their results into recurrent neural networks (RNN), specifically
a type called Long-Short-Term Memory (LSTM). But let's start from the beginning:

## Getting the data

millions of chess games are easily available online, and are usually labeled whether a computer or human played each side. I dowloaded close to 1 million games from https://www.ficsgames.org/ . The data comes in pgn format, which I first convert into JSON (using https://github.com/Assios/pgn-to-json ) and then read into pandas dataframe for preprocessing. The moves themselves are converted into board states (with help of https://github.com/niklasf/python-chess), meaning (channel x 8 x 8) arrays of 1s and 0s, where a 1 means a piece exists at this position, and the channel determines the piece type (white knight, black Queen, etc). As there are 6 pieces per color (pawns, knights, bishops, rooks, Queen, King), we have 12 channels. The board state per move is then stacked onto each other to create a tensor of shape (time, channel, 8, 8 ) for each game.

## Preprocessing, visualisation as heatmaps
Computers win more often than humans, so if we want to detect computers we should make sure that the algorithm isn't just a "win detector". For this reason I balance the data so that
1.) white is always human (for now, can extend this later)
2.) black is 50/50 human or computer
3.) everyone (white human, black human, black computer) has 50% win probability
Also, I select games with game lenght of 15-50 moves (30-100 half moves/ply) and time control of 5, 10 or 15 min per player (bullet chess is weird, and cheaters struggle there anyways, since the chess engine is often too slow). I also leave out draws. We're now left with 70 000 games.

Below we see heatmaps (average square occupation over the game), averaged over a thousand games each. There seems to be slight differences in black human vs black computer heatmaps, but in first baseline attempts wasn't able to get more than 60% test accuracy when using only heatmaps as training data for classification. More heatmaps can be seen in the data exploration notebook.

![alt text](https://user-images.githubusercontent.com/33765868/43685326-382504ce-98b1-11e8-8564-a89dd4d4c57a.png)
![alt text](https://user-images.githubusercontent.com/33765868/43685326-382504ce-98b1-11e8-8564-a89dd4d4c57a.png)
![alt text](https://user-images.githubusercontent.com/33765868/43685326-382504ce-98b1-11e8-8564-a89dd4d4c57a.png)


## CNN LSTM (work in progress)
Using the TimeDistributed wrapper on Conv2D layers allows easy setup of my network. The (channelx8x8) maps of each time step fist undergo 3 convolutional layers of kernel size 3x3 without padding, reducing the size to (filter x 2 x 2), are then flattened and fed into LSTM neurons, followed by a Dense (fully connected layer). The currently best result is 80% accuracy, see below. It is still overfitting, although dropout is applied. Will add more data soon (currently 60k games).

![alt text](https://user-images.githubusercontent.com/33765868/43685326-382504ce-98b1-11e8-8564-a89dd4d4c57a.png)
