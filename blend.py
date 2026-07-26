#combines the pregame and in-game models into one win probability.
#the two fail in opposite places. evaluate.py shows the in-game net is only about 62%
#accurate in Q1, because a 4-point lead with 40 minutes left barely constrains the
#outcome. the pregame model is about 68% at that same moment because it knows which
#roster is better, which the in-game net was never told.
#so at tip-off the pregame model should be doing the talking, and by Q4 the live score
#should have taken over completely. this learns where that handover happens instead of
#guessing: for each band of time remaining it searches for the mix that minimises log
#loss. both halves use 2025-26, which neither model trained on, and the games are split
#so the weights are fitted and tested on different games

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit

import pregame_service
from evaluate import FEATURES, time_bucket
from model import WinProbabilityModel
from train import load_data

SEASON = "2025-26"
WEIGHTS_PATH = "blend_weights.pkl"

# probabilities get clipped before going through a logarithm so a confident miss
# can't produce an infinite loss
EPSILON = 1e-6


def _ingame_probabilities(df):
    scaler = joblib.load("scaler.pkl")
    net = WinProbabilityModel()
    net.load_state_dict(torch.load("model.pth"))
    net.eval()
    scaled = scaler.transform(df[FEATURES])
    with torch.no_grad():
        return net(torch.tensor(scaled, dtype=torch.float32)).squeeze().numpy()


#work out home and away tricodes per game from the play location column. the feed
#never names the two teams directly, but every play is tagged h or v, so the team
#appearing on the h plays is the home team
def _game_teams(df):
    located = df.dropna(subset=["location", "teamTricode"])
    home = (located[located["location"] == "h"]
            .groupby("gameId")["teamTricode"].agg(lambda s: s.mode().iloc[0]))
    away = (located[located["location"] == "v"]
            .groupby("gameId")["teamTricode"].agg(lambda s: s.mode().iloc[0]))
    return pd.DataFrame({"home": home, "away": away}).dropna()


#number repeat matchups so the two datasets can be lined up. two teams meet several
#times a season so a home/away pair alone is ambiguous, but both sources are in
#chronological order, so the nth BOS vs LAL in one is the nth BOS vs LAL in the other
def _with_occurrence(frame, order_column):
    frame = frame.sort_values(order_column).copy()
    frame["occurrence"] = frame.groupby(["home", "away"]).cumcount()
    return frame


#score every 2025-26 game with the pregame model, out of sample
def _pregame_probabilities():
    models = pregame_service._load_models()
    weights = models["weights"]
    features = weights["features"]

    games = pd.read_csv("data/pregame_dataset.csv", parse_dates=["date"])
    games = games[games["season"] == SEASON].copy()

    X = games[features].values
    probability = (
        weights["lr"] * models["logistic"].predict_proba(models["scaler"].transform(X))[:, 1]
        + weights["rf"] * models["forest"].predict_proba(X)[:, 1]
        + weights["gbm"] * models["gbm"].predict_proba(X)[:, 1]
    )
    games["pregame_probability"] = probability
    return _with_occurrence(games, "date")[["home", "away", "occurrence",
                                            "pregame_probability"]]


#one row per play, carrying both models' opinions of that game
def build_frame():
    df = load_data()
    df = df.dropna(subset=FEATURES + ["home_team_won", "period"])

    # only the games the in-game network was tested on, so it isn't scored on plays
    # it already fitted
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(splitter.split(df[FEATURES], df["home_team_won"], groups=df["gameId"]))
    df = df.iloc[test_idx]

    # the pregame model only covers this season out of sample.
    # a game id looks like 0022500001: chars 0-2 are the game type (002 regular,
    # 004 playoff) and chars 3-4 are the season, so 25 means 2025-26
    season_code = SEASON.split("-")[0][-2:]
    df = df[df["gameId"].str[:3].isin(["002", "004"])
            & (df["gameId"].str[3:5] == season_code)]
    if df.empty:
        raise RuntimeError("no 2025-26 games in the in-game test split")

    teams = _game_teams(df)
    teams = _with_occurrence(teams.reset_index(), "gameId")

    pregame = _pregame_probabilities()
    matched = teams.merge(pregame, on=["home", "away", "occurrence"], how="inner")
    print(f"matched {len(matched)} of {len(teams)} test games to a pregame prediction")

    df = df.merge(matched[["gameId", "pregame_probability"]], on="gameId", how="inner")
    df["ingame_probability"] = _ingame_probabilities(df)
    df["window"] = df["seconds_remaining"].apply(time_bucket)
    return df


#per time band, find how much to trust the live model versus the pregame prior.
#weight 0 ignores the live score entirely, weight 1 ignores the pregame prior entirely
def fit_weights(frame):
    weights = {}
    for window, group in frame.groupby("window"):
        truth = group["home_team_won"].values
        live = np.clip(group["ingame_probability"].values, EPSILON, 1 - EPSILON)
        prior = np.clip(group["pregame_probability"].values, EPSILON, 1 - EPSILON)

        best_w, best_loss = 1.0, np.inf
        for w in np.arange(0, 1.001, 0.02):
            loss = log_loss(truth, np.clip(w * live + (1 - w) * prior, EPSILON, 1 - EPSILON))
            if loss < best_loss:
                best_w, best_loss = w, loss
        weights[window] = float(best_w)
    return weights


def apply_weights(frame, weights):
    live = np.clip(frame["ingame_probability"].values, EPSILON, 1 - EPSILON)
    prior = np.clip(frame["pregame_probability"].values, EPSILON, 1 - EPSILON)
    w = frame["window"].map(weights).fillna(1.0).values
    return np.clip(w * live + (1 - w) * prior, EPSILON, 1 - EPSILON)


_serving_weights = None


#the number the dashboard should actually show. early in a game this leans on the
#pregame prior, late it is pure live score. if the pregame model isn't built or the
#teams aren't recognised it falls back to the live probability rather than failing
def blended_probability(ingame_probability, seconds_remaining, home=None, away=None):
    global _serving_weights

    if _serving_weights is None:
        try:
            _serving_weights = joblib.load(WEIGHTS_PATH)
        except FileNotFoundError:
            _serving_weights = {}

    weight = _serving_weights.get(time_bucket(seconds_remaining), 1.0)
    if weight >= 1.0 or not home or not away:
        return float(ingame_probability), None

    try:
        prior = pregame_service.predict(home, away)["home_win_probability"]
    except Exception:
        return float(ingame_probability), None

    return float(weight * ingame_probability + (1 - weight) * prior), float(prior)


def main():
    frame = build_frame()

    # split by game, not by play, so a game is never in both halves
    games = np.sort(frame["gameId"].unique())
    fit_games = set(games[::2])
    fit_frame = frame[frame["gameId"].isin(fit_games)]
    test_frame = frame[~frame["gameId"].isin(fit_games)]
    print(f"fitting on {fit_frame['gameId'].nunique()} games, "
          f"evaluating on {test_frame['gameId'].nunique()} games\n")

    weights = fit_weights(fit_frame)

    rows = []
    for window in sorted(test_frame["window"].unique()):
        group = test_frame[test_frame["window"] == window]
        truth = group["home_team_won"].values
        live = np.clip(group["ingame_probability"].values, EPSILON, 1 - EPSILON)
        prior = np.clip(group["pregame_probability"].values, EPSILON, 1 - EPSILON)
        blended = apply_weights(group, weights)

        rows.append({
            "window": window,
            "plays": len(group),
            "live_weight": weights.get(window, 1.0),
            "live_acc": ((live >= 0.5) == truth).mean(),
            "blend_acc": ((blended >= 0.5) == truth).mean(),
            "live_logloss": log_loss(truth, live),
            "blend_logloss": log_loss(truth, blended),
        })

    results = pd.DataFrame(rows)
    results["logloss_gain"] = results["live_logloss"] - results["blend_logloss"]

    print("=" * 92)
    print("BLENDED vs LIVE-ONLY, on held-out 2025-26 games")
    print("=" * 92)
    print(results.to_string(index=False, float_format="%.4f"))

    truth = test_frame["home_team_won"].values
    live_all = np.clip(test_frame["ingame_probability"].values, EPSILON, 1 - EPSILON)
    blend_all = apply_weights(test_frame, weights)
    print(f"\noverall accuracy  live {((live_all >= 0.5) == truth).mean():.4f} "
          f"-> blended {((blend_all >= 0.5) == truth).mean():.4f}")
    print(f"overall log loss  live {log_loss(truth, live_all):.4f} "
          f"-> blended {log_loss(truth, blend_all):.4f}")

    joblib.dump(weights, WEIGHTS_PATH)
    print(f"\nsaved {WEIGHTS_PATH}: {weights}")


if __name__ == "__main__":
    main()
