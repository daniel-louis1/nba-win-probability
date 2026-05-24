import pandas as pd
import os
import time
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import playbyplayv3


def parse_clock(clock_str, period):
    min = clock_str.split("M")[0].split("T")[1]
    seconds = clock_str.split("M")[1].split("S")[0]

    min = float(min)
    seconds = float(seconds)

    seconds_left_in_quarter = (min * 60) + seconds
    seconds_remaining = (4 - period) * 720 + seconds_left_in_quarter
    return seconds_remaining


def process_game(game_id, home_team_won):
    # get play by play data
    pbp = playbyplayv3.PlayByPlayV3(game_id)
    df = pbp.get_data_frames()[0]

    # 1st feature - score differential
    df['score_diff'] = (pd.to_numeric(df['scoreHome']) - pd.to_numeric(df['scoreAway'])).ffill()

    # 2nd feature - seconds remaining in game
    df['seconds_remaining'] = df.apply(lambda row: parse_clock(row['clock'], row['period']), axis=1)

    # 3rd feature - possession (1 = home, 0 = away)
    df['possession'] = df['location'].map({"h": 1, "v": 0}).ffill()

    # 4th feature - foul counts
    df['home_fouls'] = ((df['actionType'] == 'Foul') & (df['location'] == 'h')).astype(int).cumsum()
    df['away_fouls'] = ((df['actionType'] == 'Foul') & (df['location'] == 'v')).astype(int).cumsum()

    # label
    df['home_team_won'] = home_team_won

    return df


def build_dataset(season):
    save_path = f"data/{season}_dataset.csv"

    gameIDsList = leaguegamefinder.LeagueGameFinder(player_or_team_abbreviation="T", season_nullable=season)
    gamesDF = gameIDsList.get_data_frames()[0]
    uniqueGames = gamesDF['GAME_ID'].unique()

    # resume from where we left off if interrupted
    if os.path.exists(save_path):
        existing = pd.read_csv(save_path)
        done_ids = existing['gameId'].unique()
        uniqueGames = [g for g in uniqueGames if g not in done_ids]
        print(f"Resuming — {len(uniqueGames)} games remaining")
    else:
        print(f"Starting fresh — {len(uniqueGames)} games to process")

    for i, game_id in enumerate(uniqueGames):
        try:
            home_row = gamesDF[(gamesDF['GAME_ID'] == game_id) & (gamesDF['MATCHUP'].str.contains('vs.'))]
            W_L = home_row['WL'].values[0]
            home_team_won = 1 if W_L == 'W' else 0
            game_df = process_game(game_id, home_team_won)
            # save after every game so we don't lose progress
            game_df.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)
            print(f"[{i+1}/{len(uniqueGames)}] Saved {game_id}")
            time.sleep(1)
        except Exception as e:
            print(f"Skipping {game_id}: {e}")
            continue

    return pd.read_csv(save_path)


dataset = build_dataset("2022-23")
print(f"\nDataset shape: {dataset.shape}")
print(dataset[['score_diff', 'seconds_remaining', 'possession', 'home_fouls', 'away_fouls', 'home_team_won']].head())
