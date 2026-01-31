def define_ebm_interactions(X):
    interactions = []

    if 'contact' in X.columns and 'macro_index' in X.columns:
        interactions.append(('contact', 'macro_index'))

    if 'contact' in X.columns and 'month_sin' in X.columns:
        interactions.append(('contact', 'month_sin'))

    if 'has_previous' in X.columns and 'log_campaign' in X.columns:
        interactions.append(('has_previous', 'log_campaign'))

    return interactions
