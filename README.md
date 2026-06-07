# NBA Win Probability Dashboard

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)

Real-time NBA win probability tracker powered by a neural network trained on **2.6M plays across 4 seasons**. Predictions update every 5 seconds during live games and stream to a dark, Kalshi-inspired dashboard — **74% accuracy** on held-out test data.

![Demo](demo.gif)

---

## Features

- **Live game tracking** — polls the NBA API every 5 seconds and runs each play through the model in real time
- **Demo mode** — auto-activates when no game is live, simulating a BOS vs LAL game with realistic momentum swings
- **Win probability chart** — scrolling line chart with crosshair tooltip showing exact probability at any moment
- **Chance to Win bar** — visual probability split with a 50% center tick
- **REST API** — `/predict` and `/live` endpoints for programmatic access

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | `nba_api`, `pandas` |
| Model | `PyTorch` — 3-layer fully connected network (5 → 16 → 16 → 1) |
| Serving | `Flask`, `scikit-learn` (StandardScaler) |
| Frontend | Vanilla JS, Chart.js |

---

## Model

The model takes 5 in-game features and outputs a win probability between 0 and 1:

| Feature | Description |
|---|---|
| `score_diff` | Home score minus away score |
| `seconds_remaining` | Seconds left in the game |
| `possession` | Which team has the ball (1 = home) |
| `home_fouls` | Home team foul count |
| `away_fouls` | Away team foul count |

**Architecture:** Linear(5→16) → ReLU → Linear(16→16) → ReLU → Linear(16→1) → Sigmoid  
**Training data:** 2.6M plays, 2022–2026 NBA seasons  
**Test accuracy:** 74%

---

## Getting Started

**1. Clone and install**
```bash
git clone https://github.com/daniel-louis1/nba-win-probability.git
cd nba-win-probability
pip install flask torch nba_api scikit-learn joblib numpy pandas
```

**2. Build the dataset** (~30–60 min)
```bash
python pipeline.py
```

**3. Train the model**
```bash
python train.py
```

**4. Run the server**
```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

> **Shortcut:** `model.pth` and `scaler.pkl` are included — skip steps 2–3 to run immediately with the pre-trained weights.

---

## API

### `POST /predict`
Send a game state, get back a win probability.

**Request:**
```json
{
  "score_diff": 5,
  "seconds_remaining": 180,
  "possession": 1,
  "home_fouls": 4,
  "away_fouls": 6
}
```

**Response:**
```json
{ "probability": 0.82 }
```

### `GET /live`
Fetches the current NBA game state, runs it through the model, and returns a prediction. Falls back to demo mode when no game is live.

---

## Project Structure

```
nba-win-probability/
├── pipeline.py       # data collection — nba_api → CSVs
├── model.py          # PyTorch model definition
├── train.py          # training loop
├── app.py            # Flask API + demo mode logic
├── templates/
│   └── index.html    # dashboard UI
├── model.pth         # pre-trained model weights
└── scaler.pkl        # fitted StandardScaler
```
