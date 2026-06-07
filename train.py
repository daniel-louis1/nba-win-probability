import torch 
import pandas as pd
from model import WinProbabilityModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_data():
    df_2022 = pd.read_csv("data/2022-23_dataset.csv", dtype={'gameId': str})
    df_2023 = pd.read_csv("data/2023-24_dataset.csv", dtype={'gameId': str})
    df_2024 = pd.read_csv("data/2024-25_dataset.csv", dtype={'gameId': str})
    df_2025 = pd.read_csv("data/2025-26_dataset.csv", dtype={'gameId': str})

    df_all = pd.concat([df_2022, df_2023, df_2024, df_2025])
    #leave out all pre-season and all star games
    df_all = df_all[df_all['gameId'].str.startswith(('0022', '0042', '0062'))]    
    return df_all


def preprocess(df):
 df = df.dropna(subset = ['score_diff', 'seconds_remaining', 'possession', 'home_fouls', 'away_fouls'])
 X = df[['score_diff', 'seconds_remaining', 'possession', 'home_fouls', 'away_fouls']]
 y = df['home_team_won']

 scaler = StandardScaler()
 X_scaled = scaler.fit_transform(X)
 X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2)

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
   print(f"Accuracy: {accuracy:.4f}")
   torch.save(model.state_dict(), "model.pth")
   
   print("Model saved!")


