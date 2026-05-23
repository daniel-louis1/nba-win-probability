import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import playbyplayv3


def process_game(game_id):

    #get play by play data
    pbp = playbyplayv3.PlayByPlayV3(game_id)
    df = pbp.get_data_frames()[0]

    #1st feature
    df['score_diff'] = (pd.to_numeric(df['scoreHome']) - pd.to_numeric(df['scoreAway'])).ffill()

    #2nd feature
    df['seconds_remaining'] = df.apply(lambda row: parse_clock(row['clock'], row['period']), axis = 1)

    #3rd feature
    df['possession'] = df['location'].map({"h" : 1, "v" : 0}).ffill()

    #4th feature
    df['home_fouls'] = ((df['actionType'] == 'Foul') & (df['location'] == 'h')).astype(int).cumsum()
    df['away_fouls'] = ((df['actionType'] == 'Foul') & (df['location'] == 'v')).astype(int).cumsum()


    return df


def parse_clock(clock_str, period):
    min = clock_str.split("M")[0].split("T")[1]
    seconds = clock_str.split("M")[1].split("S")[0]

    min = float(min)
    seconds = float(seconds)

    seconds_left_in_quarter = (min * 60) + seconds
    seconds_remaining =  (4 - period) * 720 + seconds_left_in_quarter
    return seconds_remaining

result = process_game("0042500312")
print(result[['score_diff', 'seconds_remaining', 'possession', 'home_fouls', 'away_fouls']])


