import pandas as pd
import os
import time
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import playbyplayv3

#convert to total seconds remaining in the quarter, then add remaining full quarters
def parse_clock(clock_str, period):
    min = clock_str.split("M")[0].split("T")[1]
    seconds = clock_str.split("M")[1].split("S")[0]
    min = float(min)
    seconds = float(seconds)
    seconds_left_in_quarter = (min * 60) + seconds
    seconds_remaining = (4 - period) * 720 + seconds_left_in_quarter
    return seconds_remaining

#fetch play-by-play data for a single game from NBA API and get features
def process_game(game_id, home_team_won):
    pbp = playbyplayv3.PlayByPlayV3(game_id)
    df = pbp.get_data_frames()[0]
    df['score_diff'] = (pd.to_numeric(df['scoreHome']) - pd.to_numeric(df['scoreAway'])).ffill()
    df['seconds_remaining'] = df.apply(lambda row: parse_clock(row['clock'], row['period']), axis=1)
    df['possession'] = df['location'].map({"h": 1, "v": 0}).ffill()
    df['home_fouls'] = ((df['actionType'] == 'Foul') & (df['location'] == 'h')).astype(int).cumsum()
    df['away_fouls'] = ((df['actionType'] == 'Foul') & (df['location'] == 'v')).astype(int).cumsum()
    #label: did home team win (1 = yes, 0 = no)
    df['home_team_won'] = home_team_won
    return df


def refetch_missing(season):
    save_path = f"data/{season}_dataset.csv"

    print(f"\n=== Refetching missing games for {season} ===")

    # get the full list of games for this season from the NBA API
    gameIDsList = leaguegamefinder.LeagueGameFinder(player_or_team_abbreviation="T", season_nullable=season)
    gamesDF = gameIDsList.get_data_frames()[0]
    all_ids = set(gamesDF['GAME_ID'].unique())

    # compare against what we already have saved to find gaps
    existing = pd.read_csv(save_path)
    saved_ids = set(existing['gameId'].unique())

    # skip preseason games (0012xxxxx), we don't want those in training data
    missing = [g for g in all_ids - saved_ids if not g.startswith('0012')]

    print(f"Found {len(missing)} missing non-preseason games: {missing}")

    for game_id in missing:
        try:
            # try to find the home team (team whose matchup shows 'vs.')
            home_row = gamesDF[(gamesDF['GAME_ID'] == game_id) & (gamesDF['MATCHUP'].str.contains('vs.'))]
            # fallback for neutral site games (Play-In, NBA Cup) where neither team shows 'vs.'
            if len(home_row) == 0:
                home_row = gamesDF[gamesDF['GAME_ID'] == game_id].iloc[[0]]
            W_L = home_row['WL'].values[0]
            home_team_won = 1 if W_L == 'W' else 0
            game_df = process_game(game_id, home_team_won)
            # append to existing CSV without rewriting the header
            game_df.to_csv(save_path, mode='a', header=False, index=False)
            print(f"  Saved {game_id}")
            time.sleep(1)  # rate limit, avoid getting blocked by NBA API
        except Exception as e:
            print(f"  Skipping {game_id}: {e}")


refetch_missing("2024-25")
refetch_missing("2025-26")
print("\nDone! All missing games fetched.")
