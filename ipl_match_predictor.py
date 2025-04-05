import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

ipl_matches = pd.read_csv('data/Cricket_data.csv')
ipl_matches = ipl_matches[(ipl_matches.get('season') == 2023) | (ipl_matches.get('season') == 2022)]
ipl_teams = ipl_matches.get('home_team').unique()


def concat(df, arr):
    new_df2 = pd.DataFrame()
    for i in arr:
        df1 = df[df.get('home_team') == i]
        df1 = df1.assign(opponent = df1.get('away_team'))
        df2 = df[df.get('away_team') == i]
        df2 = df2.assign(opponent = df2.get('home_team'))
        new_df = pd.concat([df1, df2])
        s  = np.array([])
        r = np.array([])
        home = np.array(new_df.get('home_team'))
        away = np.array(new_df.get('away_team'))
        winner = np.array(new_df.get('winner'))
        home_stats = ['home_overs', 'home_runs', 'home_wickets', 'home_boundaries']
        away_stats = ['away_overs', 'away_runs', 'away_wickets', 'away_boundaries']

        for k in np.arange(len(home_stats)):
            a = df2[home_stats[k]]
            df2[home_stats[k]] = df2[away_stats[k]]
            df2[away_stats[k]] = a


        for j in np.arange(0, len(home)):
            if home[j] == i:
                s = np.append(s, 'Home')
            else:
                s = np.append(s, 'Away')
            if winner[j] == i:
                r = np.append(r, 'W')
            else:
                r = np.append(r, 'L')





        new_df = new_df.assign(venue = s, result = r, team = i)
        new_df2 = pd.concat([new_df2, new_df], ignore_index= True)
    return new_df2

ipl_matches = concat(ipl_matches, ipl_teams)
ipl_matches['venue_code'] = ipl_matches['venue'].astype('category').cat.codes
ipl_matches['opp_code'] =  ipl_matches['opponent'].astype('category').cat.codes
ipl_matches['toss_code'] = ipl_matches['decision'].astype('category').cat.codes
ipl_matches['tosswinner_code'] = ipl_matches['toss_won'].astype('category').cat.codes
ipl_matches['target'] = (ipl_matches['result'] == 'W').astype('int')

rf = RandomForestClassifier(n_estimators= 50, min_samples_split = 10, random_state = 1)
train = ipl_matches[ipl_matches['season'] == 2022]
test = ipl_matches[ipl_matches['season'] == 2023]
predictors = ['venue_code', 'opp_code', 'toss_code']

rf.fit(train[predictors], train['target'])
preds = rf.predict(test[predictors])
acc = accuracy_score(test['target'], preds)
combined = pd.DataFrame(dict(actual = test['target'], prediction = preds))
print(pd.crosstab(index = combined['actual'], columns = combined['prediction']))
print(precision_score(test['target'], preds))

def rolling_averages(group, cols, new_cols):
    group = group.sort_values('id')
    rolling_stats = group[cols].rolling(3, closed = 'left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset = new_cols)
    return group


cols = ['home_overs', 'home_runs', 'home_wickets', 'home_boundaries', 'away_overs', 'away_runs', 'away_wickets', 'away_boundaries']
new_cols = [f"{c}_rolling" for c in cols]

grouped_matches = ipl_matches.groupby('team')
group = grouped_matches.get_group('SRH')

iplmatches_rolling = ipl_matches.groupby('team').apply(lambda x: rolling_averages(x, cols, new_cols))
iplmatches_rolling = iplmatches_rolling.droplevel('team').reset_index()
print(iplmatches_rolling)

def make_predictions(data, predictors):
    train = data[data['season'] == 2022.0]
    test = data[data['season'] == 2023.0]
    rf.fit(train[predictors], train['target'])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual = test['target'], predicted = preds), index = test.index)
    precision = precision_score(test['target'], preds)
    return combined, precision
combined, precision = make_predictions(iplmatches_rolling, predictors + new_cols)
print(precision)

#comparing ipl_matches and iplmatches_rolling, ipl_matches is a better model and adding more predictors made the model less accurate.





