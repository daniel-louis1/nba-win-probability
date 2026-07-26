#cleans the raw play-by-play CSVs before training. two things were quietly inflating
#the dataset:
#  1. duplicate rows. pipeline.py used to read gameId back as an integer, so the
#     resume check never matched the string ids from the API and every re-run
#     appended the same games again. 2024-25 ended up 51% duplicate, which silently
#     weights those games several times during training.
#  2. preseason and all-star games. starters rest and rotations are experimental, so
#     the result says nothing about who wins real games.
#writes new *_clean.csv files and never touches the originals, so the raw downloads
#stay intact if anything here turns out to be wrong

import os
import pandas as pd

from pipeline import EXCLUDED_GAME_PREFIXES

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

# actionNumber is unique within a game, so the pair identifies one play exactly
PLAY_KEY = ["gameId", "actionNumber"]


def clean_season(season):
    source = f"data/{season}_dataset.csv"
    destination = f"data/{season}_clean.csv"

    if not os.path.exists(source):
        print(f"{season}: {source} not found, skipping")
        return None

    df = pd.read_csv(source, dtype={"gameId": str}, low_memory=False)
    started = len(df)

    df = df.drop_duplicates(subset=PLAY_KEY, keep="first")
    after_dedupe = len(df)

    keep = ~df["gameId"].str.startswith(EXCLUDED_GAME_PREFIXES)
    excluded_games = df.loc[~keep, "gameId"].nunique()
    df = df[keep]

    df.to_csv(destination, index=False)

    print(f"{season}:")
    print(f"  {started:>9,} rows in")
    print(f"  {started - after_dedupe:>9,} duplicate rows removed "
          f"({100 * (started - after_dedupe) / started:.1f}%)")
    print(f"  {after_dedupe - len(df):>9,} rows dropped from {excluded_games} "
          f"preseason/all-star games")
    print(f"  {len(df):>9,} rows out across {df['gameId'].nunique()} games -> {destination}")
    return df


if __name__ == "__main__":
    total = 0
    games = 0
    for season in SEASONS:
        cleaned = clean_season(season)
        if cleaned is not None:
            total += len(cleaned)
            games += cleaned["gameId"].nunique()
    print(f"\n{total:,} clean rows across {games} games")
    print("point train.py at the *_clean.csv files to retrain honestly")
