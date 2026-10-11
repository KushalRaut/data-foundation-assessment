"""Build exactly 104 and 208 rows from archived FIFA reports, without future information.
Run from any directory. Raw reports and historical results are bundled for reproducibility.
"""
from pathlib import Path
from collections import defaultdict, deque
import json, re, hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
ALIAS={'Korea Republic':'South Korea','Czechia':'Czech Republic','USA':'United States',
       'Cabo Verde':'Cape Verde','Congo DR':'DR Congo','IR Iran':'Iran',
       "Côte d'Ivoire":'Ivory Coast','Türkiye':'Turkey','Bosnia & Herzegovina':'Bosnia and Herzegovina'}
def name(s):return ALIAS.get(s,s)
FEATURES1=['elo_difference','win_rate_difference','draw_rate_difference','rest_difference',
           'home_advantage','world_cup_experience_difference','head_to_head_balance','competitive_share_difference']
FEATURES2=['elo_difference','rest_difference','home_advantage','competitive_share',
           'goals_for_mean','opponent_goals_against_mean','failed_to_score_rate','opponent_clean_sheet_rate']
FEATURE_INFO={
'elo_difference':('Both','Elo points / 100','Team Elo minus opponent Elo, divided by 100. Analyst-computed ratings, initial 1500, K=20, home adjustment 100, from all archived matches before this date.'),
'win_rate_difference':('2.1','Proportion difference','Team win fraction minus opponent win fraction, each over the latest 10 prior internationals.'),
'draw_rate_difference':('2.1','Proportion difference','Team draw fraction minus opponent draw fraction over the latest 10 prior internationals.'),
'rest_difference':('Both','Days','Difference in calendar days since each side last played, each capped at 14. Source dates are used; this is not exact recovery hours.'),
'home_advantage':('Both','-1, 0, or 1','1 if team is playing in its own country; -1 if the opponent is; otherwise 0. FIFA first-listed team is not assumed to be the home side.'),
'world_cup_experience_difference':('2.1','Log count difference','log(1 + team prior World Cup finals matches) minus log(1 + opponent prior finals matches), including earlier 2026 matches.'),
'head_to_head_balance':('2.1','Smoothed result balance','Sum of result signs from the team perspective in the latest five prior meetings, divided by number of meetings + 2. No previous meeting gives 0; shootout outcomes are draws.'),
'competitive_share_difference':('2.1','Proportion difference','Difference in fractions of last 10 internationals whose source tournament is not Friendly.'),
'competitive_share':('2.2','Proportion','Fraction of the team\'s last 10 internationals whose source tournament is not Friendly. Shared concept with competitive_share_difference.'),
'goals_for_mean':('2.2','Goals per match','Mean team goals scored across its last 10 prior internationals, including extra time and excluding shootout kicks.'),
'opponent_goals_against_mean':('2.2','Goals per match','Mean goals conceded by the opponent across its last 10 prior internationals.'),
'failed_to_score_rate':('2.2','Proportion','Fraction of team\'s last 10 prior internationals in which it scored zero goals.'),
'opponent_clean_sheet_rate':('2.2','Proportion','Fraction of opponent\'s last 10 prior internationals in which it conceded zero goals.')}

def parse_fifa():
    rows=[]
    for file in sorted((ROOT/'raw/fifa').glob('*.txt')):
        text=file.read_text().replace('\xa0',' ').replace('\x00','fi')
        lines=text.splitlines()
        m=re.match(r'(.+?)\s+v\.\s+(.+?)\s+(\d+)-(\d+)(.*)',lines[1])
        assert m, file
        meta=re.match(r'#(\d+)\s*\|\s*(.+?)\s*\|\s*(.*)',lines[2]); assert meta,file
        place=meta[3].split(' / ')
        rows.append(dict(match_id=f'M{int(meta[1]):03d}',date=pd.to_datetime(meta[2],format='%d %B %Y'),
            team_a=name(m[1].strip()),team_b=name(m[2].strip()),goals_a=int(m[3]),goals_b=int(m[4]),
            stage=lines[0],venue=place[1],country={'MEX':'Mexico','CAN':'Canada','USA':'United States'}[place[2].strip()],
            extra_time=('AET' in m[5]),shootout=('PSO' in m[5]),
            source_url=f'https://fdp.fifa.org/assetspublic/ce281/r{file.stem}/pdf/FullTimeMatchReport-English.pdf'))
    df=pd.DataFrame(rows).sort_values(['date','match_id']).reset_index(drop=True)
    assert len(df)==104 and df.match_id.nunique()==104
    assert set(df.match_id)=={f'M{i:03d}' for i in range(1,105)}
    assert len(set(df.team_a)|set(df.team_b))==48
    return df

def build():
    matches=parse_fifa(); matches.to_csv(DATA/'matches_fifa.csv',index=False)
    hist=pd.read_csv(ROOT/'raw/history.csv',parse_dates=['date'])
    for c in ['home_team','away_team']:hist[c]=hist[c].map(name)
    assert not hist[['date','home_team','away_team','home_score','away_score']].isna().any().any()
    # Audit score/date differences against the independent historical collection.
    comparison=[]
    old=hist[(hist.tournament=='FIFA World Cup') & hist.date.between('2026-06-11','2026-07-19')]
    for m in matches.itertuples():
        found=old[((old.home_team==m.team_a)&(old.away_team==m.team_b))|((old.home_team==m.team_b)&(old.away_team==m.team_a))]
        if len(found)!=1:raise ValueError(f'Cannot audit {m.match_id}')
        r=found.iloc[0]
        if r.home_team!=m.team_a:
            r=r.copy(); r.home_score,r.away_score=r.away_score,r.home_score
        comparison.append(dict(match_id=m.match_id,score_agrees=bool((r.home_score==m.goals_a)&(r.away_score==m.goals_b)),
            fifa_date=str(m.date.date()),secondary_date=str(r.date.date()),date_agrees=bool(r.date==m.date)))
    pd.DataFrame(comparison).to_csv(DATA/'source_comparison.csv',index=False)
    assert all(r['score_agrees'] for r in comparison)
    hist=hist[hist.date<'2026-06-11'].copy()
    ambiguous=hist.duplicated(['date','home_team','away_team'],keep=False)
    hist[ambiguous].to_csv(DATA/'excluded_ambiguous_history.csv',index=False)
    excluded_count=int(ambiguous.sum())
    hist=hist[~ambiguous].copy()
    base=hist.rename(columns={'home_team':'team_a','away_team':'team_b','home_score':'goals_a','away_score':'goals_b'})
    base['match_id']='';base['country']=hist.country
    wc=matches.copy();wc['tournament']='FIFA World Cup';wc['neutral']=~((wc.team_a==wc.country)|(wc.team_b==wc.country))
    allgames=pd.concat([base,wc],ignore_index=True).sort_values(['date','team_a','team_b'])
    ratings=defaultdict(lambda:1500.0); recent=defaultdict(lambda:deque(maxlen=10)); meetings=defaultdict(lambda:deque(maxlen=5)); experience=defaultdict(int)
    rows1=[];rows2=[];audit=[]; metadata=[]
    for date,day in allgames.groupby('date',sort=True):
        # Extract every feature BEFORE updating any outcome from the same calendar day.
        for m in day.itertuples():
            a,b=m.team_a,m.team_b
            if not m.match_id:continue
            ra,rb=list(recent[a]),list(recent[b]);assert len(ra)==len(rb)==10,(a,b)
            assert max(r['date'] for r in ra+rb)<date
            def stats(rs):
                return dict(win=np.mean([r['gf']>r['ga'] for r in rs]),draw=np.mean([r['gf']==r['ga'] for r in rs]),
                            comp=np.mean([r['competitive'] for r in rs]),gf=np.mean([r['gf'] for r in rs]),ga=np.mean([r['ga'] for r in rs]),
                            fts=np.mean([r['gf']==0 for r in rs]),cs=np.mean([r['ga']==0 for r in rs]),
                            rest=min(14,(date-max(r['date'] for r in rs)).days))
            sa,sb=stats(ra),stats(rb);key=tuple(sorted([a,b])); h=list(meetings[key]); orient=1 if a==key[0] else -1
            common=dict(elo_difference=(ratings[a]-ratings[b])/100,rest_difference=sa['rest']-sb['rest'],
                        home_advantage=int(a==m.country)-int(b==m.country))
            r1=dict(match_id=m.match_id,**common,win_rate_difference=sa['win']-sb['win'],draw_rate_difference=sa['draw']-sb['draw'],
                    world_cup_experience_difference=np.log1p(experience[a])-np.log1p(experience[b]),
                    head_to_head_balance=orient*sum(h)/(len(h)+2),competitive_share_difference=sa['comp']-sb['comp'],goal_difference=m.goals_a-m.goals_b)
            rows1.append(r1)
            for team,opp,st,so,sign,goals in [(a,b,sa,sb,1,m.goals_a),(b,a,sb,sa,-1,m.goals_b)]:
                rid=f'{m.match_id}_{"A" if sign==1 else "B"}'
                rows2.append(dict(row_id=rid,**{k:sign*v for k,v in common.items()},competitive_share=st['comp'],
                    goals_for_mean=st['gf'],opponent_goals_against_mean=so['ga'],failed_to_score_rate=st['fts'],opponent_clean_sheet_rate=so['cs'],goals=goals))
                metadata.append(dict(row_id=rid,match_id=m.match_id,date=str(date.date()),team=team,opponent=opp,stage=m.stage))
            audit.append(dict(match_id=m.match_id,date=str(date.date()),team_a_history_end=str(max(r['date'] for r in ra).date()),
                team_b_history_end=str(max(r['date'] for r in rb).date()),team_a_history_count=len(ra),team_b_history_count=len(rb),
                team_a_elo=ratings[a],team_b_elo=ratings[b],team_a_history=json.dumps(ra,default=str),team_b_history=json.dumps(rb,default=str)))
        changes=defaultdict(float)
        for m in day.itertuples():
            a,b=m.team_a,m.team_b;ga,gb=int(m.goals_a),int(m.goals_b)
            advantage=100*(int(a==m.country)-int(b==m.country)) if m.match_id else (0 if m.neutral else 100)
            expectancy=1/(1+10**((ratings[b]-ratings[a]-advantage)/400))
            score=1 if ga>gb else .5 if ga==gb else 0
            changes[a]+=20*(score-expectancy);changes[b]-=20*(score-expectancy)
            for t,gf,gc,o in [(a,ga,gb,b),(b,gb,ga,a)]:
                recent[t].append(dict(date=date,gf=gf,ga=gc,opponent=o,competitive=m.tournament!='Friendly',source_match=m.match_id))
                experience[t]+=int(m.tournament=='FIFA World Cup')
            key=tuple(sorted([a,b]));meetings[key].append(int(np.sign(ga-gb))*(1 if a==key[0] else -1))
        for t,change in changes.items():ratings[t]+=change
    d1=pd.DataFrame(rows1)[['match_id']+FEATURES1+['goal_difference']]
    d2=pd.DataFrame(rows2)[['row_id']+FEATURES2+['goals']]
    assert d1.shape==(104,10) and d2.shape==(208,10)
    assert not d1.isna().any().any() and not d2.isna().any().any()
    d1.to_csv(DATA/'regression_2_1.csv',index=False);d2.to_csv(DATA/'regression_2_2.csv',index=False)
    pd.DataFrame(metadata).to_csv(DATA/'team_match_metadata.csv',index=False)
    pd.DataFrame(audit).to_csv(DATA/'feature_audit.csv',index=False)
    # Model IDs and dates remain metadata; neither is included as a predictor.
    dictionary=[dict(variable=k,model=v[0],unit=v[1],definition=v[2]) for k,v in FEATURE_INFO.items()]
    pd.DataFrame(dictionary).to_csv(DATA/'data_dictionary.csv',index=False)
    manifest=[]
    for f in sorted((ROOT/'raw').rglob('*')):
        if f.is_file() and (f.parent.name=='fifa' or f.name in ['history.csv','history_README.md']):
            manifest.append(dict(path=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    (DATA/'source_hashes.json').write_text(json.dumps(manifest,indent=2))
    summary=dict(matches=len(matches),team_rows=len(d2),teams=48,total_goals=int(matches.goals_a.sum()+matches.goals_b.sum()),
        extra_time_matches=int(matches.extra_time.sum()),shootouts=int(matches.shootout.sum()),
        historical_rows=len(hist),excluded_ambiguous_history_rows=excluded_count,date_disagreements=sum(not r['date_agrees'] for r in comparison),score_disagreements=0,
        shared_concepts=['elo','rest','home advantage','competitive share'],
        min_history_count=10,missing_predictors=0,source_access_date='2026-10-05')
    (DATA/'validation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':build()
