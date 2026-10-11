"""Meaningful checks of cardinality, leakage, target integrity and match grouping."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from build_data import FEATURES1,FEATURES2
P=Path(__file__).resolve().parent;D=P/'data';O=P/'output'
def main():
    a=pd.read_csv(D/'regression_2_1.csv');b=pd.read_csv(D/'regression_2_2.csv');m=pd.read_csv(D/'matches_fifa.csv')
    meta=pd.read_csv(D/'team_match_metadata.csv');audit=pd.read_csv(D/'feature_audit.csv');split=pd.read_csv(D/'split_manifest.csv')
    assert a.shape==(104,10) and b.shape==(208,10)
    assert list(a.columns)==['match_id']+FEATURES1+['goal_difference']
    assert list(b.columns)==['row_id']+FEATURES2+['goals']
    assert a.match_id.is_unique and b.row_id.is_unique
    assert np.isfinite(a[FEATURES1].to_numpy()).all() and np.isfinite(b[FEATURES2].to_numpy()).all()
    check=a.merge(m,on='match_id',validate='one_to_one');assert (check.goal_difference==check.goals_a-check.goals_b).all()
    paired=b.merge(meta,on='row_id',validate='one_to_one')
    assert (paired.groupby('match_id').size()==2).all()
    assert len(set(meta.team))==48
    for r in m.itertuples():
        x=paired[paired.match_id==r.match_id].set_index('team')
        assert x.loc[r.team_a,'goals']==r.goals_a and x.loc[r.team_b,'goals']==r.goals_b
        for f in ['elo_difference','rest_difference','home_advantage']:assert np.isclose(x[f].sum(),0)
    for r in audit.itertuples():
        for key in ['team_a_history','team_b_history']:
            rows=json.loads(getattr(r,key)); assert len(rows)==10
            assert all(pd.Timestamp(x['date'])<pd.Timestamp(r.date) for x in rows)
            assert all(x['source_match']!=r.match_id for x in rows)
    pairs=paired.merge(split[['match_id','split']],on='match_id',validate='many_to_one')
    assert (pairs.groupby('match_id')['split'].nunique()==1).all()
    assert split.loc[split.split=='Training','date'].max()<split.loc[split.split=='Holdout','date'].min()
    assert set(split.loc[split.split=='Training','match_id']).isdisjoint(split.loc[split.split=='Holdout','match_id'])
    for k in ['2_1','2_2']:
        p=pd.read_csv(O/f'predictions_{k}.csv'); stored=json.loads((O/f'model_{k}.json').read_text())
        assert np.isclose(np.mean(abs(p.target-p.prediction)),stored['holdout']['MAE'])
        assert np.isclose(np.sqrt(np.mean((p.target-p.prediction)**2)),stored['holdout']['RMSE'])
    print('PASS: exact dimensions, eight predictors, unique IDs, finite values, FIFA targets, paired rows, strict prior-date histories, chronological grouping and recalculated metrics.')
if __name__=='__main__':main()
