# Detecting-Cheating-in-Chess
Can chess engine use be detected just from the moves on the board? Let's try, using a CNN - LSTM architecture (and no chess engine).

Online chess suffers from the problem that the opponent could easily enter the moves into a chess engine on their smartphone and win easily.
Chess websites try to detect this by running an engine teirselves and comparing the moves played to the suggestions.
Sophisticated cheaters could circumvent this by randomly choosing moves that are further down the list of sugestion list,
or play a bad move once in a while, since they will win anyways. Also, with new chess engines based on neural networks
(like Deepminds "Alpha Zero" or open source "Leela Chess Zero"), the comparison of chess moves might need to be done for several engines.

Let's approach the problem from the other side. Experienced chess players can often detect a cheater when he plays "unintuitive" moves,
the kind that a human wouldn't ever naturally come up with. This hints that there might be hidden patterns in human chess, which could be 
distinguished from the way a computer plays (non-principled, just play whatever works out best 20-30 moves into the future).

Enter machine learning, specifically Deep Learning. Since the problem is both spatial (8x8 board) and temporal (board changes each move),
my intuitive approach is using convolutional neural networks (CNN) which feed their results into recurrent neural networks (RNN), specifically
a type called Long-Short-Term Memory (LSTM). But let's start from the beginning:

## Getting the data

millions of chess games are easily available online, and are usually labeled whether a computer or human played each side. 
