# NBA Win Probability Dashboard

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)

Live NBA win probability, predicted before tip-off and updated on every play. Built on
**2.38M plays across 5,087 games** from four seasons, plus a season-level dataset of 5,282
games scraped from basketball-reference.

The headline metric here is calibration rather than accuracy. When the model says 85%, the
home team should win about 85% of the time. Accuracy alone scores "51% sure" and "99% sure"
identically, which is the wrong target for a number displayed as a percentage.

---

## Three models

A live win probability model that only sees the score and the clock has a real weakness:
in the first quarter a 4-point lead barely constrains the outcome, so it is close to
guessing. A model that knows which roster is better does not have that problem, but it
learns nothing as the game unfolds. This project runs both and hands off between them.

| | Sees | Answers |
|---|---|---|
| **Pregame** | Team strength, star quality, rest, form | Who wins tonight, before tip-off |
| **In-game** | Score, clock, possession, fouls | Who wins from the current game state |
| **Blend** | Both, weighted by time remaining | The number the dashboard shows |

---

## Results

Measured on seasons the models never trained on. The pregame split is by season, and the
in-game split is by `gameId` so no game appears in both training and test.

| model | accuracy | log loss | Brier | AUC | baseline |
|---|---|---|---|---|---|
| pregame | 0.6679 | 0.6095 | 0.2103 | 0.7221 | 0.5552 |
| in-game | 0.7513 | 0.4866 | 0.1633 | 0.8350 | 0.5635 |
| **blended** | **0.7622** | **0.4826** | | | 0.5635 |

Baseline is always picking the home team.

### Calibration

Predicted probability against what actually happened, on the held-out test season:

| predicted | actual |
|---|---|
| 60-70% | 66.7% |
| 70-80% | 73.7% |
| 80-90% | 82.5% |

Most buckets land within two points of the diagonal.

### Accuracy by phase of game

A single average hides where the in-game model is weak. It is barely better than a coin
flip in the first quarter and nearly free in the fourth:

| phase | accuracy | baseline | lift |
|---|---|---|---|
| Q1 | 0.6237 | 0.5626 | +0.061 |
| Q2 | 0.7089 | 0.5624 | +0.146 |
| Q3 | 0.7874 | 0.5641 | +0.223 |
| Q4 | 0.8803 | 0.5646 | +0.316 |
| OT | 0.7152 | 0.5758 | +0.139 |

The pregame model is around 67% at tip-off, which beats the in-game model for the whole
first quarter. That is the gap the blend closes.

### The handover

`blend.py` learns how much to trust each model per band of time remaining, by minimising
log loss. Weights are fitted and evaluated on disjoint halves of held-out games:

| time remaining | live model weight |
|---|---|
| 48-36 min | 0.28 |
| 36-24 min | 0.64 |
| 24-12 min | 0.90 |
| under 12 min | 1.00 |

Past halftime the pregame prior is dropped entirely. On held-out games this lifts first
quarter accuracy from 0.6484 to 0.6972, and overall log loss from 0.4997 to 0.4826.

---

## Model details

**In-game:** `Linear(5→16) → ReLU → Linear(16→16) → ReLU → Linear(16→1) → Sigmoid`

| Feature | Description |
|---|---|
| `score_diff` | Home score minus away score |
| `seconds_remaining` | Seconds left in the game |
| `possession` | Which team has the ball (1 = home) |
| `home_fouls` | Home team foul count |
| `away_fouls` | Away team foul count |

**Pregame:** 30 features, served as a calibrated blend of logistic regression and a random
forest (0.85 / 0.15). Tuning the blend on log loss gave a gradient booster a weight of
exactly 0.00, so it is trained and reported but not served.

Features cover team strength (SRS, net rating, pace), star quality (top-3 PER and scoring,
best-player gap), rest (days off, back-to-backs, rust thresholds), current form (last 10,
net rating to date, home and road splits) and games played, which lets the model learn how
much to trust early-season form.

Most important by feature importance: `home_nrtg_advantage`, `strength_composite`,
`venue_win_pct_diff`, `home_historical_win_pct_advantage`, `home_net_rating_todate`.

### Avoiding leakage

Two things had to be handled carefully:

**Team ratings.** Basketball-reference only publishes full-season aggregates, and a
full-season rating already knows how the season ended. Using it to predict a November game
leaks the future. Each row instead uses last season's ratings as a prior, blended with this
season's record as of that date, ramping to full weight over 20 games.

**Form and rest.** Computed by walking the schedule in date order and emitting each row
before folding that game's result into the running totals.

The in-game model had a leak too. `train_test_split` ran on individual plays, and since
every play of a game shares one label, the same game landed in both train and test. Now
split on `gameId` with `GroupShuffleSplit`, verified zero overlap.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | `nba_api`, `requests`, `BeautifulSoup`, `pandas` |
| In-game model | `PyTorch` |
| Pregame model | `scikit-learn` (logistic regression, random forest, calibrated) |
| Serving | `Flask`, `joblib` |
| Frontend | Vanilla JS, Chart.js |

---

## Getting Started

**1. Clone and install**
```bash
git clone https://github.com/daniel-louis1/nba-win-probability.git
cd nba-win-probability
pip install -r requirements.txt
```

**2. Run the server**
```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

All trained model artifacts are committed, so the app runs immediately on a fresh clone.
No dataset build or training required.

### Rebuilding from scratch

```bash
python pipeline.py          # collect play-by-play (30-60 min)
python clean_data.py        # dedupe and drop preseason, non-destructive
python pregame_pipeline.py  # build the pregame dataset (BR pages are cached)
python train_pregame.py     # train pregame models, print calibration
python train.py             # retrain the in-game network
python evaluate.py          # accuracy by phase of game
python blend.py             # fit the handover weights
```

`bref.py` rate limits itself to one request every 4 seconds and caches every page to disk,
so each page downloads exactly once.

---

## API

### `GET /pregame?home=BOS&away=LAL`
Win probability before tip-off. Optional `&home_rest=2&away_rest=2`.

```json
{
  "home": "BOS", "away": "LAL",
  "home_win_probability": 0.6755,
  "away_win_probability": 0.3245,
  "pick": "BOS",
  "drivers": { "srs_gap": 6.83, "net_rating_gap": 6.4, "star_gap": -4.8,
               "rest_diff": 0, "form_gap": 0.1 }
}
```

### `POST /predict`
In-game state to win probability. Contract unchanged.

**Request:**
```json
{ "score_diff": 5, "seconds_remaining": 180, "possession": 1,
  "home_fouls": 4, "away_fouls": 6 }
```

**Response:**
```json
{ "probability": 0.82 }
```

### `GET /live`
Current live game with the blended prediction. Returns the live-only and pregame numbers
alongside it, so the dashboard can show what the model thought before tip-off next to what
it thinks now. Falls back to demo mode when no game is live.

### `GET /teams`
All 30 team tricodes, for the matchup picker.

### `GET /`
The dashboard.

---

## Project Structure

```
nba-win-probability/
├── app.py                 # Flask API, all five routes
├── model.py               # PyTorch model definition
├── pipeline.py            # play-by-play collection, nba_api to CSVs
├── refetch.py             # backfill for games the pipeline missed
├── clean_data.py          # dedupe and preseason removal, non-destructive
├── train.py               # in-game training, split by gameId
├── evaluate.py            # accuracy by quarter and time remaining
├── bref.py                # basketball-reference scraper, cached and rate limited
├── pregame_pipeline.py    # pregame dataset, leak-free by construction
├── train_pregame.py       # pregame training and calibration report
├── pregame_service.py     # matchup to probability, no network per request
├── blend.py               # learns and serves the pregame to in-game handover
├── templates/index.html   # dashboard UI
├── model.pth              # in-game weights
├── scaler.pkl             # fitted StandardScaler
└── pregame_*.pkl          # pregame models, scaler and weights
```

---

## Notes and limitations

The pregame model does not know about injuries, trades or rest days announced on game day.
A star sitting out is invisible to it, which is the largest single source of error.

Team strength early in a season leans on last year's roster, which is wrong for teams that
changed significantly over the summer. The blend ramps to current-season form over 20 games.

The random forest alone is more accurate standalone, 0.6808 against the served blend's
0.6679, but is worse once its output feeds the in-game blend. Serving it drops blended first
quarter accuracy from 0.6972 to 0.6654. Calibration, not accuracy, is what matters
downstream.

`stats.nba.com` is intermittently unreachable and the live CDN endpoints reject non-browser
clients, so `/live` falls back to demo mode more often than it should.
