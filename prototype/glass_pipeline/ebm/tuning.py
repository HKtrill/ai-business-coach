from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV


def tune_ebm(X_train, y_train, interactions, random_state=42):
    param_distributions = {
        'max_bins': [128, 256, 512],
        'learning_rate': [0.01, 0.02, 0.05],
        'min_samples_leaf': [5, 10, 20],
        'max_leaves': [3, 4],
        'max_rounds': [5000, 8000],
        'outer_bags': [8, 14],
        'inner_bags': [0, 4],
    }

    ebm = ExplainableBoostingClassifier(
        random_state=random_state,
        n_jobs=1,
        interactions=len(interactions),
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        ebm,
        param_distributions=param_distributions,
        n_iter=15,
        cv=cv,
        scoring='f1',
        n_jobs=1,
        verbose=2,
        refit=True,
        random_state=random_state,
    )

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_
