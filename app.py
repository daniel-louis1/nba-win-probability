from flask import Flask, request, jsonify, render_template
import torch
from model import WinProbabilityModel
import joblib
import numpy as np
import time
from nba_api.live.nba.endpoints import scoreboard, playbyplay


def parse_clock(clock_str, period):
    min = clock_str.split("M")[0].split("T")[1]
    seconds = clock_str.split("M")[1].split("S")[0]
    min = float(min)
    seconds = float(seconds)
    seconds_left_in_quarter = (min * 60) + seconds
    seconds_remaining = (4 - period) * 720 + seconds_left_in_quarter
    return seconds_remaining


def get_demo_state():
    # simulates a dramatic NBA game with big momentum swings
    # cycles through a full game every 4 minutes so the dashboard always looks alive
    elapsed = time.time() % 240  # 240 second cycle
    seconds_remaining = max(1, 2880 - (elapsed / 240) * 2880)
    period = min(4, int((2880 - seconds_remaining) / 720) + 1)
    mins = int((seconds_remaining % 720) // 60)
    secs = int(seconds_remaining % 60)
    clock_str = f"PT{mins:02d}M{secs:02d}.00S"

    # dramatic score swings — multiple sine waves combined for realistic momentum shifts
    progress = elapsed / 240
    score_diff = int(
        np.sin(progress * 5 * np.pi) * 12 +      # big swings
        np.sin(progress * 11 * np.pi) * 6 +       # medium runs
        np.sin(progress * 23 * np.pi) * 3          # small play-by-play noise
    )
    score_diff = max(-20, min(20, score_diff))     # cap at ±20
    home_score = 52 + int(progress * 55) + max(0, score_diff)
    away_score = 52 + int(progress * 55) + max(0, -score_diff)
    possession = 1 if int(elapsed * 2) % 2 == 0 else 0
    home_fouls = min(10, int(progress * 12))
    away_fouls = min(10, int(progress * 10))

    return {
        'score_diff': score_diff,
        'seconds_remaining': seconds_remaining,
        'possession': possession,
        'home_fouls': home_fouls,
        'away_fouls': away_fouls,
        'home_score': home_score,
        'away_score': away_score,
        'clock': clock_str,
        'period': period,
        'home_team': 'BOS',
        'away_team': 'LAL'
    }


app = Flask(__name__)

model = WinProbabilityModel()
# retrieves the file and loads the weights saved after training on 2.6M rows
model.load_state_dict(torch.load("model.pth"))
# switch to prediction mode
model.eval()

scaler = joblib.load('scaler.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # get all features from json
    score_diff = data['score_diff']
    seconds_remaining = data['seconds_remaining']
    possession = data['possession']
    home_fouls = data['home_fouls']
    away_fouls = data['away_fouls']

    # scale the input
    features = np.array([[score_diff, seconds_remaining, possession, home_fouls, away_fouls]])
    features_scaled = scaler.transform(features)

    # convert to tensor for prediction
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor)
        # convert to plain Python number and send back to dashboard
        return jsonify({'probability': prediction.item()})


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# live route fetches current play-by-play from NBA game, extracts 5 features from
# latest play, sends to model & returns the prediction + game state to dashboard
@app.route('/live')
def live():
    try:
        board = scoreboard.ScoreBoard()
        games = board.get_dict()['scoreboard']['games']
        live_game = next((game for game in games if game['gameStatus'] == 2), None)

        if live_game is None:
            raise Exception("No live games")

        # pulling game info
        game_id = live_game['gameId']
        home_team = live_game['homeTeam']['teamTricode']
        away_team = live_game['awayTeam']['teamTricode']
        home_score = live_game['homeTeam']['score']
        away_score = live_game['awayTeam']['score']

        # pulling the most recent action from play-by-play data
        play_by_play = playbyplay.PlayByPlay(game_id)
        actions = play_by_play.get_dict()['game']['actions']

        if not actions:
            raise Exception("No play data")

        latest_action = actions[-1]

        # extracting the 5 features
        score_diff = home_score - away_score
        period = latest_action['period']
        clock_str = latest_action['clock']
        seconds_remaining = parse_clock(clock_str, period)
        possession = 1 if latest_action['teamTricode'] == home_team else 0
        home_fouls = sum(1 for act in actions if act['actionType'] == 'foul' and
                         act['teamTricode'] == home_team)
        away_fouls = sum(1 for act in actions if act['actionType'] == 'foul' and
                         act['teamTricode'] == away_team)

        # scale features
        features = np.array([[score_diff, seconds_remaining, possession, home_fouls, away_fouls]])
        features_scaled = scaler.transform(features)

        # convert to tensor
        input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(input_tensor)

        return jsonify({
            'probability': prediction.item(),
            'game_state': {
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'clock': clock_str,
                'period': f'Q{period}'
            }
        })

    except Exception:
        # fallback to demo mode when no live game or API is unavailable
        demo = get_demo_state()
        features = np.array([[demo['score_diff'], demo['seconds_remaining'],
                              demo['possession'], demo['home_fouls'], demo['away_fouls']]])
        features_scaled = scaler.transform(features)
        input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(input_tensor)

        return jsonify({
            'probability': prediction.item(),
            'demo': True,
            'game_state': {
                'home_team': demo['home_team'],
                'away_team': demo['away_team'],
                'home_score': demo['home_score'],
                'away_score': demo['away_score'],
                'clock': demo['clock'],
                'period': f"Q{demo['period']}"
            }
        })


if __name__ == "__main__":
    app.run(debug=True)
