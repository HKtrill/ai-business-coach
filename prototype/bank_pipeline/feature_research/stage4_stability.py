"""PR 43 Step 5A. Operating-rule transfer and resampling on frozen scores."""
import sys
from time import perf_counter
from collections import defaultdict

import numpy as np
import pandas as pd

from feature_research import stage4_comparators as comp, training_inputs as inputs
from stage1_calibration.evidence import stratified_row_weights

n = comp.n
RUN = "c4_20261125_220626"
SHA = "509270be711b616fc5b31c21462ae18de3df09b46bad5772b3fd9b5f31d9b61d"
NAME = "frozen_stability"
FIELDS = ("coverage", "positive_retention", "negative_retention", "effective_recall", "f2", "loss")
DESIGN = ("Descriptive Step 5A convention: pair row draws within outer fold and class using policy v1's "
          "secondary 95% percentile convention. Keep scores and pre-deferral decisions fixed. Evaluate "
          "locked fold cuts on the draw; reselect only a pooled deferral cut on that draw and evaluate it "
          "both on the draw and its out-of-bag rows. Cut reselection is conditional on frozen models. "
          "Matched-coverage brackets use the original outer evaluation rows. Never promote from this analysis.")


def operating_rules(y, fold, p, pairs, cost):
    """First pair is the reference. Every intervention uses the identical probability vector."""
    out = {}
    for name, (t, cut) in pairs.items():
        d, c = p >= t, np.abs(p - t)
        keep = c >= cut
        if not out:
            d0, c0, keep0 = d, c, keep
        rows = {}
        for label, m in [('pooled', np.ones(len(y), bool)), *((str(k), fold == k) for k in np.unique(fold))]:
            v = n._outcome(y[m], d[m], keep[m], cost)
            rows[label] = {**{key: v[key] for key in FIELDS}, 'rows': int(m.sum()), 'kept': int(v['kept']),
                           'positive_kept': int(v['positive_kept']), 'negative_kept': int(v['negative_kept']),
                           'positives': int(y[m].sum()),
                           'thresholds': np.unique(np.broadcast_to(t, p.shape)[m]).tolist(),
                           'cuts': np.unique(np.broadcast_to(cut, p.shape)[m]).tolist(),
                           'changed_decisions': int(np.sum(d[m] != d0[m])), 'changed_retention': int(np.sum(keep[m] != keep0[m])),
                           'changed_confidence': int(np.sum(c[m] != c0[m])), 'mean_abs_confidence_change': float(np.abs(c[m]-c0[m]).mean())}
            if out:
                base = next(iter(out.values()))[label]
                rows[label]['delta'] = {key: v[key]-base[key]
                                        if None not in (v[key], base[key]) else None for key in FIELDS}
        out[name] = rows
    return out


def historical(table, arm, frozen, policy):
    name = 'meta_xgb' if arm == 'blackbox' else 'meta_xgb_on_glass'
    rec = frozen['historical_reference'][name]
    art = inputs.stage4_records.require_passed(inputs.CROSSOVER, 'crossover_baseline')['arbiters'][name]
    inputs._equal(art['sha256'], rec['arbiter_sha256'], name)
    pin = inputs.PinnedFile(name, art['path'], art['sha256'], art['bytes'])
    meta = inputs.joblib.load(inputs.BytesIO(inputs.read_verified(pin, inputs.BANK_ROOT)))
    inputs._equal(meta.index_train, table.index, 'historical rows')
    inputs._equal(meta.learner.y_train, table.y, 'historical labels')
    inputs._equal(meta.operating['space'], 'raw', 'historical space')
    inputs._equal(meta.operating['threshold'], rec['configuration']['operating']['threshold'], 'deployment threshold')
    inputs._equal(meta.operating['train_side_thresholds'], 'fold-nested per row', 'historical threshold semantics')
    oof = meta.learner.oof
    inputs._equal(oof['fold_id'], table.stage3_fold, 'historical folds')
    cut = rec['replay']['min_confidence']
    result = operating_rules(table.y.to_numpy(), np.asarray(oof['fold_id']), np.asarray(oof['proba'], float),
                             {'train_nested_class': (np.asarray(oof['threshold_per_row']), cut),
                              'deployment_on_frozen_rows': (meta.operating['threshold'], cut)}, policy['cost'])
    for field in ('coverage', 'f2'):
        np.testing.assert_allclose(result['train_nested_class']['pooled'][field],
                                   rec['replay']['train_'+field], rtol=0, atol=1e-12)
    return result


def assess(y, d, keep, fold, w, policy):
    """Weighted totals and the registered floor check on every evaluation fold."""
    def counts(mask):
        return np.bincount(fold, weights=w*mask, minlength=int(fold.max())+1)
    p, neg = counts(y == 1), counts(y == 0)
    kp, kn = counts(keep & (y == 1)), counts(keep & (y == 0))
    cov = policy['coverage']
    enough = ((kp+kn >= cov['min_total']*(p+neg)) & (kp >= cov['min_each_class']*p)
              & (kn >= cov['min_each_class']*neg) & (np.minimum(kp, kn) >= cov['min_retained_each_class']))
    if min(p.sum(), neg.sum()) == 0:
        return None
    return dict(coverage=(kp+kn).sum()/w.sum(), positive_retention=kp.sum()/p.sum(),
                negative_retention=kn.sum()/neg.sum(), loss=np.average(comp.linear.realized(y, d, keep, policy['cost']), weights=w),
                fold_floor_failure=float(not enough.all()))


def stability(y, fold, frames, policy):
    cfg = policy['uncertainty']
    samples = defaultdict(list)
    fails = defaultdict(int)
    # Use the registered operating_cut and risk_curve directly, including retain-all and ties.
    for w in stratified_row_weights(y, fold, cfg['replicates'], cfg['seed']):
        ix = np.repeat(np.arange(len(y)), w)
        reference = {}
        for name, frame in frames.items():
            d, c, fixed = (frame[key].to_numpy() for key in ('decision', 'confidence', 'kept'))
            cut = n.operating_cut(n.risk_curve(y[ix], d[ix], c[ix], policy['cost']), policy['coverage'])
            if cut is None:
                fails[name] += 1
            modes = [('locked', fixed, w)]
            if cut is not None:
                samples[name, 'selected_cut'].append(cut)
                modes.extend([('reselected', c >= cut, w), ('out_of_bag', c >= cut, w == 0)])
            for mode, keep, weight in modes:
                values = assess(y, d, keep, fold, weight, policy)
                if values is None:
                    continue
                if name == 'glass_formula':
                    reference[mode] = values
                if name == 'logit_margin' and mode in reference:
                    values['relative_retention_failure'] = float(any(
                        values[k] < reference[mode][k]-policy['decision']['max_retention_drop']
                        for k in ('positive_retention', 'negative_retention')))
                for k, v in values.items():
                    samples[name, mode+'.'+k].append(v)
    report = {name: dict(infeasible=fails[name], summaries={}) for name in frames}
    for (name, metric), values in samples.items():
        if metric.endswith('failure'):
            summary = {'frequency': float(np.mean(values))}
        else:
            q = np.quantile(values, [.025, .5, .975]).tolist()
            summary = dict(zip(('lower', 'median', 'upper'), q))
        report[name]['summaries'][metric] = dict(n=len(values), **summary)
    return report



def eligible(result):
    if result['linear_disposition'] != 'reject':
        raise ValueError('Unexpected accepted linear disposition')
    return (result['secondary']['logit_margin']['secondary_result'] == 'improved' and
            all(result['secondary']['logit_margin']['safeguards'].values()))


def upstream(table, arm, frozen, outer, ref, arrays, accepted, policy):
    y = table.y.to_numpy()
    frames = {name: comp.linear.reference_frame(name, table, outer, ref, arrays, policy, arm)[0]
              for name in ('glass_formula', 'meta_xgb')}
    m = frames['meta_xgb']
    folds = pd.DataFrame(ref['procedures'][arm]['meta_xgb']['rules']
                         ['registered']['folds']).set_index('outer_fold')
    t, cut = folds.threshold.loc[outer].to_numpy(), m.cut.to_numpy()
    glob = n.lock('registered', comp.META, m, y, policy)
    if glob is None:
        raise ValueError('No feasible pooled registered operating pair')
    rules = operating_rules(y, outer, m.ranking.to_numpy(), {
        'nested': (t, cut), 'global_class_only': (glob['threshold'], cut),
        'global_abstention_only': (t, glob['cut']), 'global_both': (glob['threshold'], glob['cut'])}, policy['cost'])
    for field in FIELDS:
        np.testing.assert_allclose(rules['nested']['pooled'][field],
                                   accepted['arms']['meta_f2_raw']['pooled']['outer'][field], rtol=0, atol=1e-12)
    if eligible(accepted):
        pieces = []
        cuts = pd.DataFrame(accepted['arms']['logit_margin']['folds']).set_index('outer_fold').cut
        for k, model in enumerate(ref['procedures'][arm]['glass_formula']['configuration']):
            rows = table[outer == k].drop(columns='y')
            score, _ = comp.normalized(rows, model, comp.FORMULA.score(model, rows))
            pieces.append(score.assign(kept=score.confidence >= cuts[k], cut=cuts[k]))
        f = frames['logit_margin'] = pd.concat(pieces, verify_integrity=True).loc[table.index]
        observed = n._outcome(y, f.decision.to_numpy(), f.kept.to_numpy(), policy['cost'])
        for k in FIELDS:
            np.testing.assert_allclose(observed[k], accepted['arms']['logit_margin']
                                       ['pooled']['outer'][k], rtol=0, atol=1e-12)
    locked_cuts = {name: [float(f.cut[outer == k].iloc[0]) for k in np.unique(outer)] for name, f in frames.items()}
    historical_rules = historical(table, arm, frozen, policy)
    matched = {}
    for k in np.unique(outer):
        mask = outer == k
        curves = {name: n.risk_curve(y[mask], f.decision.to_numpy()[mask], f.confidence.to_numpy()[mask], policy['cost'])
                  for name, f in frames.items()}
        matched[str(k)] = {str(g): n.matched_risks(curves, g) for g in policy['coverage']['grid']}
    return dict(historical=historical_rules, registered=rules, locked_cuts=locked_cuts,
                exact_matches=matched, stability=stability(y, outer, frames, policy))


def budget(policy):
    return dict(holdout=policy['search']['holdout'], diagnostic_triggers=policy['search']['conditional'],
                fit_rows=26360, holdout_rows=6590, fixed_parameters=True,
                # Each row gives OOF producers, their inner base-model fits, and the final refit.
                base_fits={'MLP': [10, 100, 1], 'stage3_XGB': [5, 25, 1], 'raw_Meta_XGB': [5, 0, 1],
                           'conditional_LR': [10, 100, 1], 'conditional_EBM': [5, 25, 1],
                           'RF_router_objects': [5, 0, 1], 'conditional_GLASS_router_objects': [5, 0, 1]},
                rf_forests='Each of six router fits uses five OOF forests per pass, 5x5 nested eligibility forests '
                'and two refits: 37 per router, 222 total. Do not add router-object counts to forest counts.',
                maps_and_transforms='Stage 1: 111 training-local encoder/scaler designs and 11 sigmoid maps per upstream. '
                'Stage 3: 31 feature designs and 0-6 gated maps. Stage 2: six outer/full feature designs plus '
                'five label-excluded inner designs per router where required. Cache matching feature identities. '
                'Raw fixed-parameter Meta: five OOF producers and one refit, no calibration; fit the class and '
                'abstention cuts on fitting-partition OOF scores. No hyperparameter search. The accepted formula '
                'and its normalized variant share fitted voter cuts and weights.',
                independent_rules='Capture ordered rule identities, held-out matches, positives and producing base rates '
                'during five conditional GLASS producer fits. No extra fit to count held-out matches. Unseen or '
                'under-supported rules remain unresolved. A separate five-fold replay costs five more router fits '
                'if those identities cannot be retained; require review before authorizing it.',
                measurements='PR39 c4_20261115_204125 execution_blackbox and execution_glass_recovery wall times '
                '4895.7 and 12675.7 s cover broader executions with different work/reuse. PR38 handoff: '
                'RF outer CV 118 s; GLASS outer CV 1192.6 s and full fit 170 s, older feature pathways.',
                estimate='Provisional Windows scheduling allowances: 1-3 h black-box, another 2-6 h for conditional '
                'GLASS. These are planning ranges, not benchmarks. Time the first producer of each family, '
                'multiply by remaining fits at their actual partition sizes, and review before expansion.',
                contrasts='Fit producers once per upstream. Compare fitting OOF vs holdout distributions by stage/route. '
                'On identical holdout rows compare full refit vs fold average with a locked arbiter, then hold '
                'upstream features fixed and compare full vs fold-averaged Meta. Use fit-side OOF Meta as the '
                'distribution reference. Track confidence, route transitions, coverage, class retention and loss. '
                'Compare GLASS rule/base rates per producing fold vs full fit. Lock rule for aggregating categorical '
                'routes before fitting. Freeze black-box results before considering GLASS.',
                adoption='Evaluate both references per upstream and normalized margins on GLASS if expanded. Missing '
                'valid independent rule evidence for the retained GLASS variant needs resolution before adoption. '
                'An unmeasured arm-specific shift remains a PR45 prerequisite. PR45 needs one accepted regeneration '
                'per affected upstream; PR49 multiplies fresh producer costs by independent splits. Random-split '
                'OS timings and conditional bootstrap intervals cannot establish temporal performance.')


def run():
    started = perf_counter()
    tables, frozen, policy, identity, ref, arrays, outer, _ = comp.diag.verified_tables()
    record, extra = n.load_predictions(RUN, index=tables['glass'].index, record_name='stage4_comparators')
    del extra  # archive verified; the eligible normalized formula uses the Step 2C configurations
    linear = n.records.read_run_record(comp.LINEAR_RUN, 'linear_arbiter', root=n.LOGS)
    diagnostics = n.records.read_run_record(comp.linear.DIAGNOSTIC_RUN, 'voter_diagnostics', root=n.LOGS)
    comp.linear.eligibility(diagnostics, ref, identity)
    if (record['content_sha256'] != SHA or record['inputs'] != identity or record['nested'] != diagnostics['nested_reference']
            or record['diagnostic'] != diagnostics['content_sha256'] or record['linear']['content_sha256'] != linear['content_sha256']
            or linear['content_sha256'] != comp.LINEAR_SHA):
        raise ValueError('Step 5A input chain differs from the accepted runs')
    results = {u: upstream(t, u, frozen, outer, ref, arrays,
                           record['results'][u], policy) for u, t in tables.items()}
    run_id = n.records.start_run(commit=5, seeds={'bootstrap': policy['uncertainty']['seed']}, root=n.LOGS, notebook=n.NOTEBOOK,
                                 code_files=(*n.CODE_FILES, 'feature_research/stage4_stability.py', 'feature_research/stage4_comparators.py',
                                             'feature_research/linear_arbiter.py', 'feature_research/voter_diagnostics.py', 'stage1_calibration/evidence.py'))
    evidence = dict(inputs=identity, comparator=dict(run_id=RUN, content_sha256=SHA), nested=record['nested'],
                    linear=record['linear'], diagnostic=diagnostics['content_sha256'], model_fits=0, seconds=perf_counter()-started, design=DESIGN,
                    settings=policy['uncertainty'], results=results, budget=budget(policy), limitations=n.LIMITATIONS[0]+' '+n.LIMITATIONS[-1],
                    interpretation='PR39 uses one historical deferral cut in both conditions. Pooled PR43 global selections '
                    'and mixed interventions are same-row descriptive results. Bootstrap summaries concern operating '
                    'safeguards only; probability quality, ranking, and secondary dispositions remain in Step 4B.',
                    reproduce='python -m feature_research.stage4_stability')
    path = n.records.write_run_record(run_id, NAME, evidence)
    verify(run_id.run_id)
    return path


def verify(run_id):
    rec = n.records.read_run_record(run_id, NAME, root=n.LOGS)
    policy = n.load_policy()
    if (rec['comparator'] != dict(run_id=RUN, content_sha256=SHA) or rec['model_fits'] != 0 or
            rec['inputs']['policy']['canonical_sha256'] != n.POLICY_SHA256 or rec['settings'] != policy['uncertainty']):
        raise ValueError('Invalid stability source or budget')
    for u in n.UPSTREAMS:
        r = rec['results'][u]
        expected = {'meta_xgb', 'glass_formula'} | ({'logit_margin'} if u == 'glass' else set())
        if set(r['stability']) != expected:
            raise ValueError('Unexpected stability arms')
        for arm in r['stability'].values():
            total = policy['uncertainty']['replicates']
            if (arm['infeasible'] or arm['summaries']['selected_cut']['n'] != total or
                    any(v['n'] != total for v in arm['summaries'].values())):
                raise ValueError('Incomplete resample accounting')
        for group in ('historical', 'registered'):
            for cells in r[group].values():
                all_rows = cells['pooled']
                if (all_rows['rows'] != policy['population']['train_rows'] or
                        all_rows['positives'] != policy['population']['train_positives'] or
                        all_rows['kept'] != all_rows['positive_kept'] + all_rows['negative_kept'] or
                        sum(cells[str(k)]['kept'] for k in range(5)) != all_rows['kept']):
                    raise ValueError('Invalid frozen population accounting')
    return rec['content_sha256']


if __name__ == '__main__':
    print(verify(sys.argv[2]) if sys.argv[1:2] == ['--verify'] else run())
