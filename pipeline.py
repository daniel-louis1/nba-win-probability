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

    # overtime periods are 5 minutes, not 12, and nothing is scheduled after them.
    # without this branch (4 - period) * 720 goes negative from period 5 on
    if period > 4:
        seconds_remaining = seconds_left_in_quarter
    else:
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


# NBA game ids encode the game type in the third character:
# 1 = preseason, 2 = regular season, 3 = all-star, 4 = playoffs, 5 = play-in, 6 = NBA Cup final
# preseason is exhibition basketball, rosters are experimental and starters sit, so the
# result says nothing about who actually wins games. all-star is the same problem, worse.
# both get dropped here at collection so they never reach the CSVs.
EXCLUDED_GAME_PREFIXES = ('001', '003')


def is_useful_game(game_id):
    return not str(game_id).startswith(EXCLUDED_GAME_PREFIXES)


def build_dataset(season):
    save_path = f"data/{season}_dataset.csv"

    gameIDsList = leaguegamefinder.LeagueGameFinder(player_or_team_abbreviation="T", season_nullable=season)
    gamesDF = gameIDsList.get_data_frames()[0]
    uniqueGames = gamesDF['GAME_ID'].unique()

    # drop preseason and all-star before we spend an API call on them
    before = len(uniqueGames)
    uniqueGames = [g for g in uniqueGames if is_useful_game(g)]
    print(f"Excluded {before - len(uniqueGames)} preseason/all-star games")

    # resume from where we left off if interrupted
    if os.path.exists(save_path):
        # gameId has to be read as a string. pandas otherwise turns '0022500001' into
        # the int 22500001, which never matches the string ids from the API, so the
        # resume check silently failed and every game got downloaded and appended again
        existing = pd.read_csv(save_path, dtype={'gameId': str})
        done_ids = set(existing['gameId'].unique())
        uniqueGames = [g for g in uniqueGames if g not in done_ids]
        print(f"Resuming — {len(uniqueGames)} games remaining")
    else:
        print(f"Starting fresh — {len(uniqueGames)} games to process")

    for i, game_id in enumerate(uniqueGames):
        try:
            home_row = gamesDF[(gamesDF['GAME_ID'] == game_id) & (gamesDF['MATCHUP'].str.contains('vs.'))]
            # fallback for neutral site games (Play-In, NBA Cup) where neither team shows 'vs.'
            if len(home_row) == 0:
                home_row = gamesDF[gamesDF['GAME_ID'] == game_id].iloc[[0]]
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


# guarded so that importing pipeline (for parse_clock or is_useful_game) doesn't kick
# off a 30-60 minute scrape as a side effect
if __name__ == "__main__":
    dataset = build_dataset("2025-26")
    print(f"\nDataset shape: {dataset.shape}")
    print(dataset[['score_diff', 'seconds_remaining', 'possession', 'home_fouls', 'away_fouls', 'home_team_won']].head())
