#trains the pregame model and grades it on calibration, not just accuracy.
#accuracy treats "51% sure the home team wins" and "99% sure" as the same answer when
#both turn out right. for a dashboard that prints a percentage next to a logo that is
#the wrong thing to optimise. what matters is that when the model says 85%, the home
#team really does win about 85% of the time, and brier score and log loss measure
#exactly that. accuracy is still reported against the always-pick-home baseline
#because it is the number people expect to see.
#the split is by season, never random:
#    train      2022-23, 2023-24
#    validation 2024-25, used only to pick the ensemble weights
#    test       2025-26, touched once at the very end
#a random split would let the model train on March and get tested on January of the
#same season, which it can't do in real life

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from pregame_pipeline import FEATURE_COLUMNS

TRAIN_SEASONS = ["2022-23", "2023-24"]
VALIDATION_SEASON = "2024-25"
TEST_SEASON = "2025-26"


def load_splits(path="data/pregame_dataset.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    train = df[df["season"].isin(TRAIN_SEASONS)]
    validation = df[df["season"] == VALIDATION_SEASON]
    test = df[df["season"] == TEST_SEASON]

    def xy(frame):
        return frame[FEATURE_COLUMNS].values, frame["home_won"].values

    print(f"train      {len(train):5d} games ({', '.join(TRAIN_SEASONS)})")
    print(f"validation {len(validation):5d} games ({VALIDATION_SEASON})")
    print(f"test       {len(test):5d} games ({TEST_SEASON})")
    return xy(train), xy(validation), xy(test)


#one row of the scoreboard, lower log loss and brier are better
def evaluate(name, y_true, probs):
    predictions = (probs >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, predictions),
        "log_loss": log_loss(y_true, probs),
        "brier": brier_score_loss(y_true, probs),
        "auc": roc_auc_score(y_true, probs),
    }


#the honesty check: bucket the predictions and compare what the model claimed against
#what actually happened. a well calibrated model tracks the diagonal
def calibration_table(y_true, probs, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probs >= low) & (probs < high)
        if mask.sum() == 0:
            continue
        rows.append({
            "bucket": f"{low:.0%}-{high:.0%}",
            "games": int(mask.sum()),
            "predicted": probs[mask].mean(),
            "actual": y_true[mask].mean(),
            "error": abs(probs[mask].mean() - y_true[mask].mean()),
        })
    return pd.DataFrame(rows)


def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits()

    # logistic regression needs scaling, the tree models don't care.
    # fit on train only so validation and test stay unseen
    scaler = StandardScaler().fit(X_train)

    print("\ntraining models...")
    logistic = LogisticRegression(max_iter=2000, C=0.1)
    logistic.fit(scaler.transform(X_train), y_train)

    # both tree models get wrapped in cross-validated Platt scaling. random forests
    # and boosted trees are systematically overconfident near 0 and 1, and since we
    # are grading on calibration that has to be corrected. doing it with cv=5 INSIDE
    # the training seasons keeps the validation and test seasons untouched. an earlier
    # version fit the correction on the validation season instead and it overfit
    # badly, making test log loss worse rather than better
    forest = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=500,
            # these games are noisy and the feature count is small, so an unrestricted
            # tree memorises the training seasons instead of generalising
            max_depth=6,
            min_samples_leaf=40,
            random_state=42,
            n_jobs=-1,
        ),
        method="sigmoid",
        cv=5,
    )
    forest.fit(X_train, y_train)

    # this was XGBoost originally. on macOS, XGBoost and PyTorch each load their own
    # OpenMP runtime and the combination segfaults the moment both are in one process,
    # which app.py is, since it serves the in-game net and this model together.
    # scikit-learn's histogram gradient booster is the same family of algorithm, has
    # no native dependency to install, and was the weakest of the three models anyway
    boosted = CalibratedClassifierCV(
        HistGradientBoostingClassifier(
            max_iter=400,
            max_depth=3,
            learning_rate=0.03,
            max_leaf_nodes=8,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=42,
        ),
        method="sigmoid",
        cv=5,
    )
    boosted.fit(X_train, y_train)

    def predict(model, X, scale=False):
        return model.predict_proba(scaler.transform(X) if scale else X)[:, 1]

    val_probs = {
        "logistic": predict(logistic, X_val, scale=True),
        "forest": predict(forest, X_val),
        "gbm": predict(boosted, X_val),
    }
    test_probs = {
        "logistic": predict(logistic, X_test, scale=True),
        "forest": predict(forest, X_test),
        "gbm": predict(boosted, X_test),
    }

    # pick the ensemble weight on validation, by log loss. weighting by log loss
    # rather than accuracy is the whole point: we want the blend that produces the
    # most trustworthy probabilities, not the most coin flips called correctly.
    # search all three weights, not just forest vs booster. logistic regression turned
    # out to be competitive on its own, and leaving it out of the blend was throwing
    # away a model that disagrees with the trees in useful places
    best_weights, best_loss = None, np.inf
    for w_lr in np.arange(0, 1.01, 0.05):
        for w_rf in np.arange(0, 1.01 - w_lr, 0.05):
            w_gbm = 1 - w_lr - w_rf
            blend = (w_lr * val_probs["logistic"]
                     + w_rf * val_probs["forest"]
                     + w_gbm * val_probs["gbm"])
            loss = log_loss(y_val, blend)
            if loss < best_loss:
                best_weights, best_loss = (w_lr, w_rf, w_gbm), loss

    w_lr, w_rf, w_gbm = best_weights
    print(f"\nbest blend on validation: lr={w_lr:.2f} rf={w_rf:.2f} gbm={w_gbm:.2f} "
          f"(log loss {best_loss:.4f})")

    for probs in (val_probs, test_probs):
        probs["ensemble"] = (w_lr * probs["logistic"]
                             + w_rf * probs["forest"]
                             + w_gbm * probs["gbm"])

    # baseline: always pick the home team
    home_rate = y_train.mean()
    baseline_probs = np.full(len(y_test), home_rate)

    print(f"\n{'=' * 62}")
    print(f"VALIDATION ({VALIDATION_SEASON}), used for tuning")
    print("=" * 62)
    val_rows = [evaluate(n, y_val, p) for n, p in val_probs.items()]
    print(pd.DataFrame(val_rows).to_string(index=False, float_format="%.4f"))

    print(f"\n{'=' * 62}")
    print(f"TEST ({TEST_SEASON}), never used for any decision")
    print("=" * 62)
    test_rows = [evaluate("always-home baseline", y_test, baseline_probs)]
    test_rows += [evaluate(n, y_test, p) for n, p in test_probs.items()]
    results = pd.DataFrame(test_rows)
    print(results.to_string(index=False, float_format="%.4f"))

    print("\ncalibration of the final model on the test season:")
    print(calibration_table(y_test, test_probs["ensemble"])
          .to_string(index=False, float_format="%.4f"))

    print("\ntop features (random forest, averaged over calibration folds):")
    forest_importances = np.mean(
        [fold.estimator.feature_importances_ for fold in forest.calibrated_classifiers_],
        axis=0,
    )
    importance = (pd.Series(forest_importances, index=FEATURE_COLUMNS)
                  .sort_values(ascending=False).head(10))
    print(importance.to_string(float_format="%.4f"))

    joblib.dump(forest, "pregame_rf.pkl")
    joblib.dump(boosted, "pregame_gbm.pkl")
    joblib.dump(logistic, "pregame_lr.pkl")
    joblib.dump(scaler, "pregame_scaler.pkl")
    joblib.dump({"lr": w_lr, "rf": w_rf, "gbm": w_gbm, "features": FEATURE_COLUMNS},
                "pregame_weights.pkl")
    print("\nmodels saved")

    return results


if __name__ == "__main__":
    main()
