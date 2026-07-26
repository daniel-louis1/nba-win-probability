#builds the pregame dataset, one row per game, predicting the winner before tip-off.
#different problem from the in-game model: no score and no clock, so everything has to
#come from what we knew the morning of the game, meaning how good each team has looked,
#how good they were last year, who their best players are and how rested they are.
#the whole file follows one rule, that a row may only contain information that existed
#BEFORE that game was played. easy to state and easy to break, so the two places it
#could break are handled explicitly:
#  1. team ratings. basketball-reference only publishes full-season aggregates, and a
#     full-season rating already knows how the season ended, so using it to predict a
#     November game leaks the future. each row uses LAST season's ratings as a prior,
#     blended with this season's record as of that date. early on the prior dominates
#     and by 20 games the current season has taken over.
#  2. form and rest. computed by walking the schedule in date order and emitting each
#     row before folding that game's result into the running totals.

import numpy as np
import pandas as pd

import bref

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

# how many games it takes before this season's record outweighs last season's rating.
# roughly a quarter of a season, long enough to be signal but short enough to react to
# a team that got much better or much worse over the summer
BLEND_GAMES = 20

# a team playing its first game of the season has no previous game to rest from.
# 3 days is about a normal in-season gap, so it avoids inventing an advantage
DEFAULT_REST_DAYS = 3

# rest beyond this starts looking like rust rather than recovery
EXCESSIVE_REST_DAYS = 5


#last season's team ratings and player quality, used as the prior for this season.
#everything here is from a season that had already finished before the first game of
#this one was played, so none of it can leak
def _season_reference_data(season):
    prior = bref.previous_season(season)
    ratings = bref.team_ratings(prior)
    strength = bref.team_player_strength(prior)
    return ratings.join(strength, how="outer", rsuffix="_p")


#running, as-of-date record for one team. only updated after a row is emitted
class TeamState:
    def __init__(self):
        self.games = 0
        self.wins = 0
        self.points_for = 0
        self.points_against = 0
        self.home_games = 0
        self.home_wins = 0
        self.away_games = 0
        self.away_wins = 0
        self.results = []          # 1/0 per game, most recent last
        self.last_game_date = None

    @property
    def win_pct(self):
        # .500 is the honest guess for a team we haven't seen play yet
        return self.wins / self.games if self.games else 0.5

    @property
    def net_rating(self):
        # points per game differential, a simple stand-in for net rating that we can
        # compute from the schedule alone without another scrape
        if not self.games:
            return 0.0
        return (self.points_for - self.points_against) / self.games

    @property
    def last10(self):
        if not self.results:
            return 0.5
        recent = self.results[-10:]
        return sum(recent) / len(recent)

    @property
    def home_win_pct(self):
        return self.home_wins / self.home_games if self.home_games else 0.5

    @property
    def away_win_pct(self):
        return self.away_wins / self.away_games if self.away_games else 0.5

    def rest_days(self, date):
        if self.last_game_date is None:
            return DEFAULT_REST_DAYS
        return (date - self.last_game_date).days

    def update(self, date, won, scored, allowed, at_home):
        self.games += 1
        self.wins += won
        self.points_for += scored
        self.points_against += allowed
        self.results.append(won)
        self.last_game_date = date
        if at_home:
            self.home_games += 1
            self.home_wins += won
        else:
            self.away_games += 1
            self.away_wins += won


#look up last season's stat, tolerating expansion and relocation gaps
def _prior_value(reference, team, column, default=0.0):
    if team in reference.index and column in reference.columns:
        value = reference.loc[team, column]
        if pd.notna(value):
            return float(value)
    return default


#turn raw rest into the shapes that actually affect a basketball game. rest is not
#linear: one day off is normal, two is an edge, and a week off makes teams sloppy
#rather than fresh, so the raw number plus a few thresholds carries more signal than
#the number on its own
def _rest_flags(rest_home, rest_away):
    diff = rest_home - rest_away
    return {
        "home_rest_days": rest_home,
        "away_rest_days": rest_away,
        "rest_diff": diff,
        "big_rest_advantage_home": int(diff >= 2),
        "big_rest_advantage_away": int(diff <= -2),
        "excessive_rest_home": int(rest_home > EXCESSIVE_REST_DAYS),
        "excessive_rest_away": int(rest_away > EXCESSIVE_REST_DAYS),
        "optimal_rest_home": int(1 <= rest_home <= 2),
        "optimal_rest_away": int(1 <= rest_away <= 2),
        "home_back_to_back": int(rest_home == 0),
        "away_back_to_back": int(rest_away == 0),
    }


#walk one season in date order, emitting a feature row before each game
def build_season(season, include_playoffs=True):
    reference = _season_reference_data(season)
    games = bref.schedule(season)

    if not include_playoffs:
        games = games[games["game_type"] == "regular"]

    state = {code: TeamState() for code in bref.TEAM_CODES}
    rows = []

    for game in games.itertuples(index=False):
        home, away, date = game.home, game.away, game.date
        home_state, away_state = state[home], state[away]

        # last season's prior
        srs_home = _prior_value(reference, home, "srs")
        srs_away = _prior_value(reference, away, "srs")
        pace_home = _prior_value(reference, home, "pace", 100.0)
        pace_away = _prior_value(reference, away, "pace", 100.0)
        per_home = _prior_value(reference, home, "top3_per", 15.0)
        per_away = _prior_value(reference, away, "top3_per", 15.0)
        pts_home = _prior_value(reference, home, "top3_pts", 18.0)
        pts_away = _prior_value(reference, away, "top3_pts", 18.0)
        elite_home = _prior_value(reference, home, "top1_per", 15.0)
        elite_away = _prior_value(reference, away, "top1_per", 15.0)

        # blend the prior with this season's evidence. confidence in the current
        # season grows with games played and caps at 1
        weight_home = min(home_state.games / BLEND_GAMES, 1.0)
        weight_away = min(away_state.games / BLEND_GAMES, 1.0)
        blended_home = weight_home * home_state.net_rating + (1 - weight_home) * srs_home
        blended_away = weight_away * away_state.net_rating + (1 - weight_away) * srs_away

        srs_advantage = srs_home - srs_away
        nrtg_advantage = blended_home - blended_away
        player_eff_advantage = per_home - per_away
        win_pct_advantage = home_state.win_pct - away_state.win_pct

        rest = _rest_flags(home_state.rest_days(date), away_state.rest_days(date))

        row = {
            "season": season,
            "date": date,
            "home": home,
            "away": away,
            "game_type": game.game_type,

            # the 17 original features, now computed leak-free
            "home_srs_advantage": srs_advantage,
            "home_nrtg_advantage": nrtg_advantage,
            "home_pace_delta": pace_home - pace_away,
            "home_player_top3_eff_advantage": player_eff_advantage,
            "home_player_top3_pts_advantage": pts_home - pts_away,
            "home_historical_win_pct_advantage": win_pct_advantage,
            # one summary number so the tree models get the overall mismatch directly
            # instead of having to rediscover it from the parts every split
            "strength_composite": (
                0.4 * srs_advantage + 0.4 * nrtg_advantage + 0.2 * player_eff_advantage
            ),
            # how lopsided the best-player matchup is. a single superstar swings a
            # game more than a slightly deeper bench does
            "elite_gap": elite_home - elite_away,
            **rest,

            # added on top of the original 17: current-season form, which the
            # full-season-ratings version could not express without leaking
            "home_form_last10": home_state.last10,
            "away_form_last10": away_state.last10,
            "form_diff": home_state.last10 - away_state.last10,
            "home_net_rating_todate": home_state.net_rating,
            "away_net_rating_todate": away_state.net_rating,
            "home_venue_win_pct": home_state.home_win_pct,
            "away_venue_win_pct": away_state.away_win_pct,
            "venue_win_pct_diff": home_state.home_win_pct - away_state.away_win_pct,
            # lets the model learn how much to trust the form features early on
            "home_games_played": home_state.games,
            "away_games_played": away_state.games,
            "min_games_played": min(home_state.games, away_state.games),

            "home_won": int(game.home_won),
        }
        rows.append(row)

        # only now does this game become part of what we know
        home_state.update(date, int(game.home_won), game.home_pts, game.away_pts, True)
        away_state.update(date, int(not game.home_won), game.away_pts, game.home_pts, False)

    return pd.DataFrame(rows)


def build_dataset(seasons=SEASONS, include_playoffs=True):
    frames = []
    for season in seasons:
        print(f"\n=== building {season} ===")
        df = build_season(season, include_playoffs=include_playoffs)
        print(f"  {len(df)} games, home win rate {df['home_won'].mean():.4f}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


FEATURE_COLUMNS = [
    "home_srs_advantage", "home_nrtg_advantage", "home_pace_delta",
    "home_player_top3_eff_advantage", "home_player_top3_pts_advantage",
    "home_historical_win_pct_advantage", "strength_composite", "elite_gap",
    "home_rest_days", "away_rest_days", "rest_diff",
    "big_rest_advantage_home", "big_rest_advantage_away",
    "excessive_rest_home", "excessive_rest_away",
    "optimal_rest_home", "optimal_rest_away",
    "home_back_to_back", "away_back_to_back",
    "home_form_last10", "away_form_last10", "form_diff",
    "home_net_rating_todate", "away_net_rating_todate",
    "home_venue_win_pct", "away_venue_win_pct", "venue_win_pct_diff",
    "home_games_played", "away_games_played", "min_games_played",
]


if __name__ == "__main__":
    dataset = build_dataset()
    dataset.to_csv("data/pregame_dataset.csv", index=False)

    print(f"\nsaved data/pregame_dataset.csv: {dataset.shape[0]} games, "
          f"{len(FEATURE_COLUMNS)} features")
    print(dataset["season"].value_counts().sort_index())
    print(f"\noverall home win rate: {dataset['home_won'].mean():.4f}")
    print(dataset[["date", "home", "away", "home_srs_advantage",
                   "home_nrtg_advantage", "rest_diff", "home_won"]].head())
