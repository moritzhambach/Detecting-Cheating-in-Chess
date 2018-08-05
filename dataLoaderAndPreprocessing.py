import pandas as pd

def loadData():
    file1 = r'''C:\Users\mhamb\jupyter_notebooks\chess\data1\ficsgamesdb_2015_CvH_nomovetimes_1529540.json'''
    file2 = r'''C:\Users\mhamb\jupyter_notebooks\chess\data1\ficsgamesdb_2016_chess_nomovetimes_1531440.json'''
    file3 = r'''C:\Users\mhamb\jupyter_notebooks\chess\data1\ficsgamesdb_201708_chess_nomovetimes_1531489.json'''
    file4 = r'''C:\Users\mhamb\jupyter_notebooks\chess\data1\ficsgamesdb_2016_CvH_nomovetimes_1529438.json'''
    file5 = r'''C:\Users\mhamb\jupyter_notebooks\chess\data1\ficsgamesdb_2017_CvH_nomovetimes_1529542.json'''

    df = pd.read_json(file1, lines=True)

    for file in [file2, file3, file4, file5]:
        df = pd.concat([df, pd.read_json(file, lines=True)])

    df = df.drop(['Black', 'BlackElo', 'BlackClock', 'Time', 'BlackRD', 'Date', 'ECO', 'Event', 'FICSGamesDBGameNo',
              'Round', 'Site', 'WhiteRD', 'White', 'WhiteElo', 'WhiteClock'], axis=1)

    return df

def filterAndBalanceData(df, plymin, plymax):
    df = df=df[df.PlyCount>plymin]                             # restrict game lengths
    df = df[df.PlyCount<plymax]
    df = df[(df.Result=="1-0") |(df.Result=="0-1")]            # no draws
    df = df[(df.TimeControl=="300+0") |(df.TimeControl=="600+0") | (df.TimeControl=="900+0")]    # choose timecontrol
    df = df[df.WhiteIsComp.isnull()==True]                     # white always human player
    df = df.sample(frac=1)                                     # shuffle data
    # now balance black player label and win rate (so the algorithm doesn't just detect game winners)
    dfBlackcomputer_winning =  df[(df.BlackIsComp == "Yes") & (df.Result=="0-1")]
    dfBlackcomputer_losing =  df[(df.BlackIsComp == "Yes") & (df.Result=="1-0")]
    dfBlackhuman_winning = df[(df.BlackIsComp.isnull() == True) & (df.Result=="0-1")]
    dfBlackhuman_losing = df[(df.BlackIsComp.isnull() == True) & (df.Result=="1-0")]
    n_min=min([len(dfBlackcomputer_winning), len(dfBlackcomputer_losing),
              len(dfBlackhuman_winning), len(dfBlackhuman_losing)])
    df = pd.concat([dfBlackcomputer_winning[:n_min], dfBlackcomputer_losing[:n_min],
                 dfBlackhuman_winning[:n_min], dfBlackhuman_losing[:n_min]])
    df = df.sample(frac=1)                                        #reshuffle
    df = df.reset_index(drop=True)
    df=df.fillna(value=0)                             #for some reasons "no" is coded as NaN in the chess package...
    df=df.replace(to_replace="Yes", value=1)
    return df

