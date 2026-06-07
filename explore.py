#playbyplay has every single event in a game
from nba_api.stats.endpoints import playbyplayv3
#leaguegamefinder has a list of games (game IDs) 
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd


gameIDsList = leaguegamefinder.LeagueGameFinder(player_or_team_abbreviation = "T", season_nullable = "2025-26")
gamesDF = gameIDsList.get_data_frames()[0]
gameID = gamesDF['GAME_ID']
print(gameID)

events = playbyplayv3.PlayByPlayV3(gameID[0])

print(events.get_data_frames()[0]['location'].unique())



