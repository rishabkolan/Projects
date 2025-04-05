import numpy as np
import pandas as pd


ipl = pd.read_csv('data/ipl_data.csv')
ipl = ipl[~ipl.eq('No stats').any(axis=1)]
ipl = ipl.drop(columns = ['Highest_Score', 'Best_Bowling_Match', 'Year'])
ipl = ipl.set_index('Player_Name')
ipl = ipl.astype(float)
ipl = ipl.reset_index()
ipl = ipl.groupby('Player_Name').sum()
ipl = ipl.assign(Batting_Strike_Rate = (ipl.get('Runs_Scored')/ipl.get('Balls_Faced'))*100,
                 Batting_Average = ipl.get('Runs_Scored')/ipl.get('Matches_Batted'),
                 Economy_Rate = ipl.get('Runs_Conceded')/(ipl.get('Balls_Bowled')/6)
                  )
auction = pd.read_csv('data/ipl_2025_auction_players.csv')
auction = auction.set_index('Players')
ipl = ipl.merge(auction, left_index = True, right_index = True).drop(columns = ['Team', 'Base', 'Sold', 'Type'])
ipl = ipl.fillna(0.0)

def get_stats(df, name, stats):
    s = np.array([])
    for i in range(0, len(stats)):
        s = np.append(s, df.get(stats[i]).loc[name])
    return s
def calculate_similarity(stats1, stats2):

    total = 0
    if len(stats1) == len(stats2):


        stats_subtract = stats1 - stats2
        for i in np.arange(0, len(stats_subtract)):

            total = total + stats_subtract[i]**2

        return total**0.5
    else:
        return 10000


def calculate_similarity_for_all(name, stats):
    s = np.array([])
    for i in np.arange(0, len(ipl.index)):
        s = np.append(s, calculate_similarity(get_stats(ipl, name, stats), get_stats(ipl, ipl.index[i], stats)))
    return s

def get_player_recommendations(n, df, name, stats):

    s = ipl.assign(similarity = calculate_similarity_for_all(name,stats)).sort_values(by = 'similarity', ascending = True).take(np.arange(1, n+1))
    return s


print(get_player_recommendations(5, ipl, 'Kane Williamson', ['Batting_Strike_Rate', 'Batting_Average', 'Balls_Faced']))























