#grades the in-game model by phase of game instead of one number for everything.
#a single accuracy figure is misleading here: calling the winner with 40 seconds left
#and a 15 point lead is nearly free, calling it in the opening minutes is close to a
#coin flip, and averaging them describes neither situation.
#reproduces the exact test split from train.py (same seed, same grouping by gameId)
#so these are the same games the model was graded on

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupShuffleSplit

from model import WinProbabilityModel
from train import load_data

FEATURES = ['score_diff', 'seconds_remaining', 'possession', 'home_fouls', 'away_fouls']


#which part of the game a play belongs to
def phase_label(row):
    period = row['period']
    if period > 4:
        return "OT"
    return f"Q{int(period)}"


#coarse bands of game time remaining, tighter at the end where it matters
def time_bucket(seconds):
    if seconds > 2160:
        return "1. 48-36 min left"
    if seconds > 1440:
        return "2. 36-24 min left"
    if seconds > 720:
        return "3. 24-12 min left"
    if seconds > 300:
        return "4. 12-5 min left"
    if seconds > 60:
        return "5. 5-1 min left"
    return "6. final minute"


def summarise(frame, label_column):
    rows = []
    for label, group in frame.groupby(label_column, observed=True):
        truth = group['home_team_won'].values
        probs = group['probability'].values
        predicted = (probs >= 0.5).astype(int)

        # the bar to clear in this phase: always pick the home team
        baseline = max(truth.mean(), 1 - truth.mean())

        rows.append({
            label_column: label,
            "plays": len(group),
            "accuracy": (predicted == truth).mean(),
            "baseline": baseline,
            "lift": (predicted == truth).mean() - baseline,
            # log loss needs both classes present to be meaningful
            "log_loss": log_loss(truth, probs) if len(np.unique(truth)) > 1 else np.nan,
            "brier": brier_score_loss(truth, probs) if len(np.unique(truth)) > 1 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(label_column)


def main():
    df = load_data()
    df = df.dropna(subset=FEATURES + ['home_team_won', 'period'])

    # identical split to train.py so we are scoring genuinely held-out games
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(splitter.split(df[FEATURES], df['home_team_won'], groups=df['gameId']))
    test = df.iloc[test_idx].copy()

    scaler = joblib.load("scaler.pkl")
    model = WinProbabilityModel()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    scaled = scaler.transform(test[FEATURES])
    with torch.no_grad():
        test['probability'] = model(torch.tensor(scaled, dtype=torch.float32)).squeeze().numpy()

    test['phase'] = test.apply(phase_label, axis=1)
    test['window'] = test['seconds_remaining'].apply(time_bucket)

    print(f"test set: {len(test):,} plays across {test['gameId'].nunique()} games\n")

    print("=" * 74)
    print("IN-GAME MODEL BY QUARTER")
    print("=" * 74)
    print(summarise(test, 'phase').to_string(index=False, float_format="%.4f"))

    print("\n" + "=" * 74)
    print("IN-GAME MODEL BY TIME REMAINING")
    print("=" * 74)
    print(summarise(test, 'window').to_string(index=False, float_format="%.4f"))

    print("\nthe pattern to expect: accuracy climbs as the game runs out of time,")
    print("because score and clock become decisive. early-game plays are where the")
    print("pregame model carries the information the in-game model does not have.")


if __name__ == "__main__":
    main()
