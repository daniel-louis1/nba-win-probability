#basketball-reference scraper, the second data source for the pregame model. pulls
#team ratings (SRS, net rating, pace, four factors), player quality (PER, points,
#VORP) and full season schedules.
#two things make this messier than a normal scrape:
#  1. BR rate limits hard, roughly 20 requests a minute gets a temporary ban, so
#     live requests are spaced out and every page is cached to disk. after the first
#     run the whole pipeline builds with zero network calls.
#  2. most of the interesting tables are wrapped in HTML comments to keep scrapers
#     out, so the comment markers get stripped before parsing.

import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE = "https://www.basketball-reference.com"
CACHE_DIR = "data/bref"

# BR bans around 20 requests/min, so stay well under it
RATE_LIMIT_SECONDS = 4.0

# a browser user-agent, BR returns 403 to the default requests one
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_last_request_time = 0.0


# BR writes out full team names, the rest of the project uses NBA tricodes
TEAM_NAME_TO_CODE = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

# BR uses its own abbreviations in the player tables that disagree with NBA tricodes
BR_CODE_FIXES = {"BRK": "BKN", "PHO": "PHX", "CHO": "CHA", "CHH": "CHA"}

TEAM_CODES = sorted(set(TEAM_NAME_TO_CODE.values()))


#'2025-26' becomes 2026, BR labels a season by the year it ends in
def season_to_year(season):
    return int(season.split("-")[0]) + 1


#'2025-26' becomes '2024-25'
def previous_season(season):
    start = int(season.split("-")[0]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _cache_path(url):
    slug = url.replace(BASE + "/", "").replace("/", "_")
    return os.path.join(CACHE_DIR, slug)


#get a page, from disk if we already have it, so every page downloads exactly once
def _fetch(url):
    global _last_request_time

    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(url)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # space out live requests so BR doesn't ban us
    wait = RATE_LIMIT_SECONDS - (time.time() - _last_request_time)
    if wait > 0:
        time.sleep(wait)

    print(f"  fetching {url}")
    response = requests.get(url, headers=HEADERS, timeout=30)
    _last_request_time = time.time()

    if response.status_code == 429:
        # BR asked us to slow down, back off once and retry
        print("  rate limited, backing off 60s")
        time.sleep(60)
        response = requests.get(url, headers=HEADERS, timeout=30)
        _last_request_time = time.time()

    response.raise_for_status()
    # BR serves utf-8 but doesn't always say so, and player names break without this
    response.encoding = "utf-8"

    with open(path, "w", encoding="utf-8") as f:
        f.write(response.text)
    return response.text


def _soup(url):
    html = _fetch(url)
    # the good tables are hidden inside HTML comments, so drop the comment markers
    html = html.replace("<!--", "").replace("-->", "")
    return BeautifulSoup(html, "lxml")


#turn a BR table into a DataFrame keyed by the data-stat attribute. table ids drift
#between seasons (advanced vs advanced_stats), so this takes a list of candidates and
#uses whichever one the page actually has
def _table_to_df(soup, table_ids):
    table = None
    for table_id in table_ids:
        table = soup.find("table", id=table_id)
        if table is not None:
            break
    if table is None:
        raise ValueError(f"none of these tables were on the page: {table_ids}")

    rows = []
    for tr in table.find_all("tr"):
        classes = tr.get("class") or []
        # skip the repeated header rows BR sprinkles through long tables
        if "thead" in classes:
            continue
        cells = tr.find_all(["th", "td"])
        row = {c.get("data-stat"): c.get_text(strip=True) for c in cells if c.get("data-stat")}
        # header row has no real values
        if not row or all(v == "" for v in row.values()):
            continue
        rows.append(row)

    return pd.DataFrame(rows)


def _numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


#season-long team strength from the advanced team table. one row per team indexed by
#tricode: SRS, strength of schedule, off/def/net rating, pace and the four factors
def team_ratings(season):
    year = season_to_year(season)
    soup = _soup(f"{BASE}/leagues/NBA_{year}.html")
    df = _table_to_df(soup, ["advanced-team", "misc_stats", "misc"])

    # BR marks playoff teams with a trailing asterisk, and closes with a League Average row
    df["team"] = df["team"].str.rstrip("*").str.strip()
    df = df[df["team"].isin(TEAM_NAME_TO_CODE)].copy()
    df["team_code"] = df["team"].map(TEAM_NAME_TO_CODE)

    numeric_cols = [
        "wins", "losses", "mov", "sos", "srs", "off_rtg", "def_rtg", "net_rtg", "pace",
        "efg_pct", "tov_pct", "orb_pct", "ft_rate",
        "opp_efg_pct", "opp_tov_pct", "drb_pct", "opp_ft_rate", "ts_pct", "age",
    ]
    df = _numeric(df, numeric_cols)

    df["win_pct"] = df["wins"] / (df["wins"] + df["losses"])
    df["season"] = season

    keep = ["season", "team_code", "wins", "losses", "win_pct", "srs", "sos", "mov",
            "off_rtg", "def_rtg", "net_rtg", "pace", "efg_pct", "tov_pct", "orb_pct",
            "ft_rate", "opp_efg_pct", "opp_tov_pct", "drb_pct", "opp_ft_rate", "age"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].set_index("team_code").sort_index()


#per-player advanced and per-game stats, merged. players traded mid-season get a
#combined 2TM/3TM row plus one row per team, and the combined rows get dropped so
#every remaining row belongs to a real team
def player_stats(season):
    year = season_to_year(season)

    adv = _table_to_df(_soup(f"{BASE}/leagues/NBA_{year}_advanced.html"),
                       ["advanced", "advanced_stats"])
    pg = _table_to_df(_soup(f"{BASE}/leagues/NBA_{year}_per_game.html"),
                      ["per_game_stats", "per_game"])

    for df in (adv, pg):
        df["team_code"] = df["team_name_abbr"].replace(BR_CODE_FIXES)

    adv = _numeric(adv, ["per", "mp", "ws", "bpm", "vorp", "usg_pct", "games"])
    pg = _numeric(pg, ["pts_per_g", "mp_per_g", "games"])

    # drop the aggregate rows for traded players, we want per-team attribution
    adv = adv[~adv["team_code"].str.contains("TM", na=False)]
    pg = pg[~pg["team_code"].str.contains("TM", na=False)]

    adv = adv[adv["team_code"].isin(TEAM_CODES)]
    pg = pg[pg["team_code"].isin(TEAM_CODES)]

    merged = adv.merge(
        pg[["name_display", "team_code", "pts_per_g", "mp_per_g"]],
        on=["name_display", "team_code"],
        how="left",
    )
    merged["season"] = season
    return merged.rename(columns={"name_display": "player"})


#collapse player stats into a per-team measure of star power. a team's ceiling in a
#single game leans on its best few players, so this takes the top 3 by scoring and by
#PER rather than a roster-wide average. the minutes and games thresholds keep a
#4-game call-up with a fluke PER out of the top 3
def team_player_strength(season):
    players = player_stats(season)

    rows = []
    for code in TEAM_CODES:
        team = players[players["team_code"] == code]
        if len(team) == 0:
            continue

        # only players with a real sample get to define the team's ceiling
        rotation = team[(team["mp"].fillna(0) >= 400) & (team["games"].fillna(0) >= 15)]
        if len(rotation) < 3:
            rotation = team

        top_pts = rotation.nlargest(3, "pts_per_g")["pts_per_g"]
        top_per = rotation.nlargest(3, "per")["per"]
        top_vorp = rotation.nlargest(5, "vorp")["vorp"]

        rows.append({
            "team_code": code,
            "season": season,
            "top3_pts": top_pts.mean(),
            "top1_pts": top_pts.max(),
            "top3_per": top_per.mean(),
            "top1_per": top_per.max(),
            "top5_vorp": top_vorp.sum(),
            "rotation_size": len(rotation),
        })

    return pd.DataFrame(rows).set_index("team_code").sort_index()


#every game of a season with date, teams, final score and game type. BR's schedule
#pages only cover regular season, play-in and playoffs, preseason is never listed,
#which is exactly what we want here.
#playoff games aren't tagged, so the play-in round is used as the boundary: anything
#before the first play-in game is regular season, anything after the last is playoffs
def schedule(season):
    year = season_to_year(season)
    index = _soup(f"{BASE}/leagues/NBA_{year}_games.html")

    # the month pages are linked from the season index, safer than guessing month names
    month_links = []
    filter_div = index.find("div", class_="filter")
    if filter_div:
        month_links = [a.get("href") for a in filter_div.find_all("a") if a.get("href")]
    if not month_links:
        month_links = [f"/leagues/NBA_{year}_games.html"]

    frames = []
    for link in month_links:
        soup = _soup(BASE + link)
        try:
            frames.append(_table_to_df(soup, ["schedule"]))
        except ValueError:
            # a month page can exist with no games played yet
            continue

    games = pd.concat(frames, ignore_index=True)
    games = games[games["visitor_team_name"].isin(TEAM_NAME_TO_CODE)]

    games["date"] = pd.to_datetime(games["date_game"], format="mixed")
    games["away"] = games["visitor_team_name"].map(TEAM_NAME_TO_CODE)
    games["home"] = games["home_team_name"].map(TEAM_NAME_TO_CODE)
    games = _numeric(games, ["visitor_pts", "home_pts"])
    games = games.rename(columns={"visitor_pts": "away_pts"})

    # games that haven't been played yet have no score
    games = games.dropna(subset=["home_pts", "away_pts"])

    remarks = games.get("game_remarks", pd.Series("", index=games.index)).fillna("")
    is_play_in = remarks.str.contains("Play-In", na=False)

    games["game_type"] = "regular"
    if is_play_in.any():
        play_in_start = games.loc[is_play_in, "date"].min()
        play_in_end = games.loc[is_play_in, "date"].max()
        games.loc[games["date"] > play_in_end, "game_type"] = "playoff"
        games.loc[is_play_in, "game_type"] = "play_in"
        # anything between the play-in window that isn't play-in is still regular season
        games.loc[(games["date"] < play_in_start), "game_type"] = "regular"
    else:
        # older seasons without a play-in: fall back to the 1230-game regular season
        ordered = games.sort_values("date").index
        games.loc[ordered[1230:], "game_type"] = "playoff"

    games["home_won"] = (games["home_pts"] > games["away_pts"]).astype(int)
    games["overtime"] = games.get("overtimes", pd.Series("", index=games.index)).fillna("") != ""
    games["season"] = season

    keep = ["season", "date", "home", "away", "home_pts", "away_pts",
            "home_won", "game_type", "overtime"]
    return games[keep].sort_values(["date", "home"]).reset_index(drop=True)


if __name__ == "__main__":
    # quick smoke test so the scraper can be checked without running the whole pipeline
    season = "2025-26"

    ratings = team_ratings(season)
    print(f"\nteam ratings {season}: {ratings.shape}")
    print(ratings[["wins", "losses", "srs", "net_rtg", "pace"]].head())

    strength = team_player_strength(season)
    print(f"\nplayer strength {season}: {strength.shape}")
    print(strength.sort_values("top3_per", ascending=False).head())

    games = schedule(season)
    print(f"\nschedule {season}: {games.shape}")
    print(games["game_type"].value_counts())
    print(games.head())
