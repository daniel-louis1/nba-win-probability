import os
import torch
import pandas as pd
from model import WinProbabilityModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import joblib

def load_data():
    seasons = ["2022-23", "2023-24", "2024-25", "2025-26"]
    frames = []

    for season in seasons:
        #prefer the cleaned file if clean_data.py has been run. the raw downloads
        #contain duplicate rows from the old broken resume check, and 2024-25 was
        #over half duplicates, which silently weighted those games several times
        clean_path = f"data/{season}_clean.csv"
        raw_path = f"data/{season}_dataset.csv"
        path = clean_path if os.path.exists(clean_path) else raw_path
        frames.append(pd.read_csv(path, dtype={'gameId': str}, low_memory=False))

    df_all = pd.concat(frames)
    #leave out all pre-season and all star games
    df_all = df_all[df_all['gameId'].str.startswith(('0022', '0042', '0062'))]
    return df_all


def preprocess(df):
 df = df.dropna(subset = ['score_diff', 'seconds_remaining', 'possession', 'home_fouls', 'away_fouls'])
 X = df[['score_diff', 'seconds_remaining', 'possession', 'home_fouls', 'away_fouls']]
 y = df['home_team_won']

 #splitting on plays leaked the answer: every play of a game shares one home_team_won
 #label, so a random play-level split put the same game in train AND test. the model
 #could memorize "this game was a home win" from one play and get graded on another.
 #GroupShuffleSplit keeps whole games together so the test set is genuinely unseen.
 splitter = GroupShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
 train_idx, test_idx = next(splitter.split(X, y, groups = df['gameId']))

 X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
 y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

 #fit the scaler on train only. fitting on everything lets test-set means and
 #standard deviations bleed into training
 scaler = StandardScaler()
 X_train = scaler.fit_transform(X_train_raw)
 X_test = scaler.transform(X_test_raw)

 print(f"Train: {len(X_train)} plays from {df['gameId'].iloc[train_idx].nunique()} games")
 print(f"Test:  {len(X_test)} plays from {df['gameId'].iloc[test_idx].nunique()} games")

 return X_train, X_test, y_train, y_test, scaler

def train(model, X_train, y_train):
   criterion = torch.nn.BCELoss()
   optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
   X_tensor = torch.tensor(X_train, dtype = torch.float32)
   y_tensor = torch.tensor(y_train.values, dtype = torch.float32).unsqueeze(1)
   for epoch in range(200):
      #forward pass
      predictions = model(X_tensor)
      #calculate loss
      loss = criterion(predictions, y_tensor)
      #backward pass
      optimizer.zero_grad()
      loss.backward()
      #update weights
      optimizer.step()

      if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
   df = load_data()
   X_train, X_test, y_train, y_test, scaler = preprocess(df)
   #create model and run training data
   model = WinProbabilityModel()
   train(model, X_train, y_train)
   #now try unseen data
   X_test = torch.tensor(X_test, dtype = torch.float32)
   with torch.no_grad():
        test_predictions = model(X_test)
    #finally convert, above 0.5 is home team (1) wins, below is they lost (0)
   predicted = (test_predictions >= 0.5).float()
   y_test = torch.tensor(y_test.values, dtype = torch.float32)
   #remove extra col
   correct_predictions = predicted.squeeze() == y_test
   accuracy = correct_predictions.float().mean()

   #accuracy alone is a weak measure for a win probability model. being 51% sure and
   #being 99% sure both count as "right", but only one of them is a useful broadcast
   #graphic. log loss and brier score grade how well calibrated the number actually is.
   probs = test_predictions.squeeze().numpy()
   truth = y_test.numpy()
   #a model that just predicts the home team wins every time is the bar to clear
   baseline = max(truth.mean(), 1 - truth.mean())

   print(f"Accuracy:        {accuracy:.4f}")
   print(f"Home-pick basel: {baseline:.4f}")
   print(f"Log loss:        {log_loss(truth, probs):.4f}")
   print(f"Brier score:     {brier_score_loss(truth, probs):.4f}")
   print(f"ROC AUC:         {roc_auc_score(truth, probs):.4f}")

   torch.save(model.state_dict(), "model.pth")
   #the scaler has to be saved alongside the weights. app.py loads scaler.pkl at
   #startup, and serving with a scaler fit on different data silently skews every
   #prediction. this was previously never written out, so scaler.pkl went stale.
   joblib.dump(scaler, "scaler.pkl")

   print("Model saved!")


