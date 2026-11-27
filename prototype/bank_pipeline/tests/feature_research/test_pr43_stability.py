import numpy as np
import pandas as pd
import pytest

from feature_research import stage4_stability as s


def test_identical_scores_class_and_cut_interventions():
    y = np.array([0, 1, 0, 1])
    p = np.array([.125, .375, .625, .875])
    result = s.operating_rules(y, np.array([0, 0, 1, 1]), p,
                               {'nested': (.5, .125), 'class_only': (.25, .125), 'cut_only': (.5, .375)}, s.n.load_policy()['cost'])
    ref = result['nested']['pooled']
    assert ref['kept'] == 4 and ref['coverage'] == 1
    assert result['class_only']['pooled']['changed_decisions'] == 1
    c = result['cut_only']['pooled']
    assert (c['changed_decisions'], c['changed_confidence'], c['changed_retention']) == (0, 0, 2)
    assert c['positive_retention'] == c['negative_retention'] == .5
    assert c['loss'] == .125 and c['delta']['loss'] == -.625


def test_stratified_draws_and_cut_safeguards():
    y = np.tile([0, 0, 1, 1], 20)
    k = np.repeat(np.arange(5), 16)
    d = y.copy()
    d[1::4] = 1
    conf = np.where(y != d, .1, .8)
    frame = pd.DataFrame(dict(decision=d, confidence=conf, kept=conf >= .8))
    policy = s.n.load_policy()
    policy['coverage']['min_retained_each_class'] = 1
    policy['uncertainty']['replicates'] = 24
    for w in s.stratified_row_weights(y, k, 24, 4302):
        np.testing.assert_array_equal(np.bincount(k*2+y, weights=w), np.bincount(k*2+y))
    r = s.stability(y, k, {'glass_formula': frame, 'logit_margin': frame}, policy)
    ref = r['glass_formula']['summaries']
    assert r['glass_formula']['infeasible'] == 0 and ref['selected_cut']['n'] == 24
    assert ref['selected_cut']['upper'] == .8
    assert ref['locked.coverage']['lower'] < ref['locked.coverage']['upper']
    assert ref['out_of_bag.coverage']['n'] == 24
    assert r['logit_margin']['summaries']['locked.relative_retention_failure']['frequency'] == 0
    assert ref['locked.fold_floor_failure']['frequency'] > 0
    cnt, risks, brackets = s.n.matched_risks({'arm': s.n.risk_curve(y, d, conf, policy['cost'])}, .5)
    assert cnt == 40 and risks is None and brackets['arm'] == (0, 60)


def test_accepted_evidence_and_budget():
    r = s.n.records.read_run_record(s.RUN, 'stage4_comparators', root=s.n.LOGS)
    assert r['content_sha256'] == s.SHA
    assert [u for u in s.n.UPSTREAMS if s.eligible(r['results'][u])] == ['glass']
    b = s.budget(s.n.load_policy())
    assert sum(b['base_fits']['MLP']) == 111 and sum(b['base_fits']['stage3_XGB']) == 31
    assert b['holdout_rows']+b['fit_rows'] == 32950
    assert b['holdout'] == s.n.load_policy()['search']['holdout']


def test_unverified_prediction_archive_stops_before_analysis(monkeypatch):
    policy = s.n.load_policy()
    table = pd.DataFrame(index=[11, 28])
    monkeypatch.setattr(s.comp.diag, 'verified_tables', lambda: (
        {'glass': table}, {}, policy, {}, {}, {}, None, None))

    def fail(run_id, index, record_name):
        assert run_id == s.RUN and record_name == 'stage4_comparators'
        assert index.equals(table.index)
        raise ValueError('archive mismatch')
    monkeypatch.setattr(s.n, 'load_predictions', fail)
    with pytest.raises(ValueError, match='archive mismatch'):
        s.run()


def test_historical_training_and_deployment_share_one_abstention_cut(monkeypatch):
    from types import SimpleNamespace

    table = pd.DataFrame(dict(y=[0, 1, 0, 1], stage3_fold=[0, 0, 1, 1]), index=[10, 12, 21, 27])
    oof = dict(proba=[.125, .375, .625, .875], threshold_per_row=[.25, .25, .75, .75], fold_id=table.stage3_fold)
    operating = dict(space='raw', threshold=.5, train_side_thresholds='fold-nested per row')
    meta = SimpleNamespace(index_train=table.index, operating=operating, learner=SimpleNamespace(y_train=table.y, oof=oof))
    artifact = dict(path='unused', sha256='a'*64, bytes=10)
    record = dict(arbiter_sha256=artifact['sha256'], configuration={'operating': operating},
                  replay=dict(min_confidence=.125, train_coverage=1., train_f2=1.))
    monkeypatch.setattr(s.inputs.stage4_records, 'require_passed', lambda *_: {'arbiters': {'meta_xgb': artifact}})
    monkeypatch.setattr(s.inputs, 'read_verified', lambda *_: b'')
    monkeypatch.setattr(s.inputs.joblib, 'load', lambda _: meta)
    rules = s.historical(table, 'blackbox', {'historical_reference': {'meta_xgb': record}}, s.n.load_policy())
    deploy = rules['deployment_on_frozen_rows']['pooled']
    assert deploy['changed_decisions'] == 2 and deploy['changed_retention'] == 0
    assert deploy['cuts'] == [.125] and deploy['thresholds'] == [.5]
    meta.index_train = table.index[::-1]
    with pytest.raises(ValueError, match='historical rows'):
        s.historical(table, 'blackbox', {'historical_reference': {'meta_xgb': record}}, s.n.load_policy())
