#serves the pregame model, turning two team codes into a win probability.
#train_pregame.py leaves behind fitted models but no way to ask them about a matchup
#that hasn't happened yet. this rebuilds a feature row the same way pregame_pipeline.py
#does, using each team's most recent state from the dataset plus last season's
#basketball-reference ratings, so a hypothetical BOS vs LAL gets scored with exactly
#the features the model was trained on.
#everything is loaded once at import and cached, so a request costs no network calls

import joblib
import numpy as np
import pandas as pd

import bref
from pregame_pipeline import BLEND_GAMES, FEATURE_COLUMNS, SEASONS, _rest_flags

DATASET_PATH = "data/pregame_dataset.csv"
LATEST_SEASON = SEASONS[-1]

_models = None
_snapshots = None
_reference = None


def _load_models():
    global _models
    if _models is None:
        weights = joblib.load("pregame_weights.pkl")
        _models = {
            "logistic": joblib.load("pregame_lr.pkl"),
            "forest": joblib.load("pregame_rf.pkl"),
            "gbm": joblib.load("pregame_gbm.pkl"),
            "scaler": joblib.load("pregame_scaler.pkl"),
            "weights": weights,
        }
    return _models


#each team's end-of-season form, taken from the last game it appears in. a team's home
#stats only exist on rows where it was the home team, so the home and away snapshots
#get pulled separately rather than from one row
def _load_snapshots():
    global _snapshots
    if _snapshots is not None:
        return _snapshots

    df = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    df = df[df["season"] == LATEST_SEASON].sort_values("date")

    snapshots = {}
    for team in bref.TEAM_CODES:
        as_home = df[df["home"] == team]
        as_away = df[df["away"] == team]
        if as_home.empty or as_away.empty:
            continue

        # the final row for a team already reflects every game before it, and since
        # features are emitted pre-game we also fold in that last result
        last_home = as_home.iloc[-1]
        last_away = as_away.iloc[-1]
        latest = as_home.iloc[-1] if last_home["date"] >= last_away["date"] else as_away.iloc[-1]
        side = "home" if last_home["date"] >= last_away["date"] else "away"

        snapshots[team] = {
            "net_rating": float(latest[f"{side}_net_rating_todate"]),
            "form_last10": float(latest[f"{side}_form_last10"]),
            "games_played": int(latest[f"{side}_games_played"]),
            "home_win_pct": float(last_home["home_venue_win_pct"]),
            "away_win_pct": float(last_away["away_venue_win_pct"]),
        }

    _snapshots = snapshots
    return _snapshots


def _load_reference():
    global _reference
    if _reference is None:
        prior = bref.previous_season(LATEST_SEASON)
        _reference = bref.team_ratings(prior).join(
            bref.team_player_strength(prior), how="outer", rsuffix="_p"
        )
    return _reference


def _prior(team, column, default):
    reference = _load_reference()
    if team in reference.index and column in reference.columns:
        value = reference.loc[team, column]
        if pd.notna(value):
            return float(value)
    return default


#assemble one feature row in exactly the order the model expects
def build_features(home, away, home_rest=2, away_rest=2):
    snapshots = _load_snapshots()
    if home not in snapshots or away not in snapshots:
        raise ValueError(f"no data for {home} or {away}")

    hs, aws = snapshots[home], snapshots[away]

    srs_advantage = _prior(home, "srs", 0.0) - _prior(away, "srs", 0.0)
    per_home, per_away = _prior(home, "top3_per", 15.0), _prior(away, "top3_per", 15.0)
    player_eff_advantage = per_home - per_away

    # both teams have a full season played, so the blend weight is 1 and the blended
    # rating is just the current-season net rating
    weight_home = min(hs["games_played"] / BLEND_GAMES, 1.0)
    weight_away = min(aws["games_played"] / BLEND_GAMES, 1.0)
    blended_home = weight_home * hs["net_rating"] + (1 - weight_home) * _prior(home, "srs", 0.0)
    blended_away = weight_away * aws["net_rating"] + (1 - weight_away) * _prior(away, "srs", 0.0)
    nrtg_advantage = blended_home - blended_away

    win_pct_home = (hs["home_win_pct"] + hs["away_win_pct"]) / 2
    win_pct_away = (aws["home_win_pct"] + aws["away_win_pct"]) / 2

    row = {
        "home_srs_advantage": srs_advantage,
        "home_nrtg_advantage": nrtg_advantage,
        "home_pace_delta": _prior(home, "pace", 100.0) - _prior(away, "pace", 100.0),
        "home_player_top3_eff_advantage": player_eff_advantage,
        "home_player_top3_pts_advantage": _prior(home, "top3_pts", 18.0) - _prior(away, "top3_pts", 18.0),
        "home_historical_win_pct_advantage": win_pct_home - win_pct_away,
        "strength_composite": (0.4 * srs_advantage + 0.4 * nrtg_advantage
                               + 0.2 * player_eff_advantage),
        "elite_gap": _prior(home, "top1_per", 15.0) - _prior(away, "top1_per", 15.0),
        **_rest_flags(home_rest, away_rest),
        "home_form_last10": hs["form_last10"],
        "away_form_last10": aws["form_last10"],
        "form_diff": hs["form_last10"] - aws["form_last10"],
        "home_net_rating_todate": hs["net_rating"],
        "away_net_rating_todate": aws["net_rating"],
        "home_venue_win_pct": hs["home_win_pct"],
        "away_venue_win_pct": aws["away_win_pct"],
        "venue_win_pct_diff": hs["home_win_pct"] - aws["away_win_pct"],
        "home_games_played": hs["games_played"],
        "away_games_played": aws["games_played"],
        "min_games_played": min(hs["games_played"], aws["games_played"]),
    }
    return row


#win probability for the home team, plus the stat gaps that drove it
def predict(home, away, home_rest=2, away_rest=2):
    models = _load_models()
    row = build_features(home, away, home_rest, away_rest)
    X = np.array([[row[c] for c in FEATURE_COLUMNS]], dtype=float)

    #serve the tuned blend, not the forest on its own. the forest is the better
    #standalone model on accuracy (0.6808 to 0.6679) but the blend is better
    #calibrated, and once it feeds into blend.py that is what matters: blended Q1
    #accuracy is 0.6972 with the blend against 0.6654 with the forest alone
    weights = models["weights"]
    probability = (
        weights["lr"] * models["logistic"].predict_proba(models["scaler"].transform(X))[:, 1]
        + weights["rf"] * models["forest"].predict_proba(X)[:, 1]
        + weights["gbm"] * models["gbm"].predict_proba(X)[:, 1]
    )[0]

    return {
        "home": home,
        "away": away,
        "home_win_probability": round(float(probability), 4),
        "away_win_probability": round(1 - float(probability), 4),
        "pick": home if probability >= 0.5 else away,
        # the drivers the dashboard shows under each pick
        "drivers": {
            "srs_gap": round(row["home_srs_advantage"], 2),
            "net_rating_gap": round(row["home_nrtg_advantage"], 2),
            "star_gap": round(row["elite_gap"], 2),
            "rest_diff": row["rest_diff"],
            "form_gap": round(row["form_diff"], 3),
        },
    }


def list_teams():
    return sorted(_load_snapshots().keys())


if __name__ == "__main__":
    print("teams:", len(list_teams()))
    for matchup in [("BOS", "LAL"), ("OKC", "WAS"), ("DET", "BOS")]:
        result = predict(*matchup)
        print(f"{result['away']} @ {result['home']}: "
              f"{result['pick']} {max(result['home_win_probability'], result['away_win_probability']):.1%} "
              f"| drivers {result['drivers']}")
