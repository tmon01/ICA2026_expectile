"""Build a standalone HTML report and tables from frozen final test prices.
Run compare_overall.py and compare_shortfalls.py first. No fitting occurs.
"""
from pathlib import Path
import json,base64,html,hashlib
import pandas as pd,numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
M=['mean','quantile','expectile'];labels={'mean':'Mean','quantile':'Quantile','expectile':'Expectile'}
A=json.loads((OUT/'analysis.json').read_text());S=json.loads((ROOT/'summary.json').read_text());T=json.loads((OUT/'tail_results.json').read_text())
tail={x['model']:x for x in T['metrics']};a=A['results']
df=pd.read_csv(ROOT/'test_policy_pricing_matched.csv');y=df.observed_period_loss.to_numpy();e=df.Exposure.to_numpy()
assert len(df)==133780 and np.isclose(y.sum(),10976159.65)
meta=dict(policies=len(df),total_loss=float(y.sum()),total_exposure=float(e.sum()),
 pricing_sha256=hashlib.sha256((ROOT/'test_policy_pricing_matched.csv').read_bytes()).hexdigest(),
 frozen_prices=True,no_refit=True,training_claim_exclusion=None)
(OUT/'evaluation_manifest.json').write_text(json.dumps(meta,indent=2))
sections=[]
def section(title,body):sections.append((title,body))
def para(text):return '<p>'+html.escape(text)+'</p>'
def table(frame):return frame.to_html(index=False,escape=True,border=0)
def model_table(specs,source):
 rows=[]
 for title,key,fmt in specs:
  r={'指標':title}
  for m in M:r[labels[m]]=format(source[m][key],fmt)
  rows.append(r)
 return pd.DataFrame(rows)
def fig(filename,caption):
 p=OUT/filename
 if not p.exists():p=ROOT/filename
 return '<figure><img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'
settings=[]
for m in M:
 r=S['selected'][m];p=S['severity_parameters'][m]
 settings.append({'モデル':labels[m],'α':r['alpha'],'倍率':r['multiplier'],'best_iteration':r['best_iteration'],'valid LR (%)':100*r['valid']['loss_ratio'],'test LR (%)':100*r['test']['loss_ratio'],'深さ':p['max_depth'],'学習率':p['eta'],'min_child_weight':p['min_child_weight'],'L2':p['reg_lambda'],'L1':p['reg_alpha'],'subsample':p['subsample'],'colsample_bytree':p['colsample_bytree']})
settings=pd.DataFrame(settings);settings.to_csv(OUT/'selected_models.csv',index=False)
section('対象・比較条件',para('trainの最大損害4,075,400.56ユーロを含む、外れ値除外なし・深さ1で正則化とサンプリングを探索後の最終モデル。Frequencyは共通のPoisson GLM。Severityは事故単位の正の損害額。下記の条件で学習・選択したモデルを用い、確定したtest価格で全比較を計算した。')+para(f'test {len(df):,}契約、Exposure {e.sum():,.2f}年、実績損害 {y.sum():,.2f}ユーロ。testの除外・損害額の丸めなし。valid LRの実数値は下表参照。目標約50%だが完全一致ではない。train17,970件の観測されたSeverity明細を全て使用。事故件数あり・損害額不明の9,116契約とfreqに結合できない195事故明細は、従来と同じく対象外。「外れ値除外なし」は観測・結合可能な損害額を落としていないという意味。')+table(settings.round(6).fillna('—')))
section('分析設計',para('契約単位でtrain/valid/testを分割してから事故明細へ展開した。train 454,849、valid 80,268、test 133,780契約。FrequencyはPoisson GLM、Severityは正の事故損害額に対するXGBoostで、全モデルに同じFrequencyを使う。')+para('Meanはvalid RMSE、Quantileはpinball loss、Expectileは非対称二乗損失で反復数と設定を選ぶ。Quantileの損失はα max(y−q,0)+(1−α) max(q−y,0)、Expectileはα max(y−q,0)²+(1−α) max(q−y,0)²。いずれも事故単位の損失。')+para('価格はFrequency×Severity×Exposure。Quantile/Expectileはvalid損害率50%に近いαを採用し、Meanは定数倍率でQ/Eのvalid価格総計の平均に合わせる。αは価格の水準を決めるための指標であり、保険会社の安全確率や契約総損害の信頼水準を直接意味しない。testは選択に使わない。'))
overall_specs=[('提示額総計 (€)','total_premium',',.2f'),('損害率','loss_ratio','.6f'),('Exposure加重の年額提示価格 (€)','annual_tariff',',.2f'),('期間MAE (€)','period_mae',',.2f'),('期間RMSE (€)','period_rmse',',.2f'),('年額損害のExposure加重MAE (€)','annual_weighted_mae',',.2f'),('年額損害のExposure加重RMSE (€)','annual_weighted_rmse',',.2f'),('参考：提示価格を0.5倍した年額RMSE (€)','annual_rmse_after_fixed_target_deload',',.2f'),('Gini（Exposure考慮）','exposure_weighted_gini','.6f'),('高価格側10%Exposureの損害捕捉率','top10_claim_capture','.6f'),('自身の上位10%群の損害率','top10_loss_ratio','.6f'),('十分位の配分誤差：全体test LR基準','decile_shape_gap_fraction','.6f'),('十分位の配分誤差：50% LR基準','decile_absolute_calibration_gap_fraction','.6f'),('ゼロ価格の契約割合','zero_policy_share','.6f'),('ゼロ価格のExposure割合','zero_exposure_share','.6f'),('ゼロ価格群の損害シェア','zero_claim_share','.6f'),('ゼロ価格群の損害あり契約数','zero_claim_count','d')]
overall=model_table(overall_specs,a);overall.to_csv(OUT/'overall_comparison_table.csv',index=False)
section('全体精度・順位付け・ゼロ価格・価格配分',table(overall)+para('Giniは年額提示価格によるExposure加重の損害集中度で、oracleで正規化したGiniではない。十分位の配分誤差は Σ群 |実績損害−基準LR×提示額| / 全実績損害。全体test LR基準は全体水準を除いた配分を見る診断であり、testに基づく価格変更ではない。0.5倍提示価格のRMSEは参考値で、Quantile/Expectileを平均損害推定量とみなす根拠にはならない。')+fig('pricing_comparison.png','左：高価格群からの累積損害捕捉。右：各モデル自身の十分位別実績年額損害。群の構成はモデルごとに異なる。'))
tail_specs=[('赤字契約数','deficit_contracts','d'),('赤字契約割合','deficit_probability','.6f'),('Exposure加重の赤字割合','exposure_weighted_deficit_probability','.6f'),('赤字額合計 (€)','total_shortfall',',.2f'),('1契約当たり不足額 (€)','mean_shortfall',',.4f'),('赤字契約に限った平均不足額 (€)','mean_shortfall_given_deficit',',.2f'),('不足額二乗の平均 (€²)','second_shortfall_moment',',.2f'),('不足額二乗平均の平方根 (€)','root_second_shortfall_moment',',.2f'),('不足額 ES95 (€)','shortfall_ES_0.95',',.2f'),('不足額 ES99 (€)','shortfall_ES_0.99',',.2f'),('不足額 ES99.5 (€)','shortfall_ES_0.995',',.2f'),('不足額 ES99.9 (€)','shortfall_ES_0.999',',.2f'),('黒字額合計 (€)','total_surplus',',.2f'),('黒字−赤字の純額 (€)','net_surplus',',.2f'),('FOC比：α=.66','expectile_foc_ratio_066','.6f'),('集計FOCからの暗黙α','implied_expectile_alpha','.6f'),('非対称二乗スコア α=.66','expectile_score_066',',.2f'),('Pinballスコア α=.948','pinball_score_0948',',.4f')]
tails=model_table(tail_specs,tail);tails.to_csv(OUT/'downside_comparison_table.csv',index=False)
section('赤字・不足額・極端テール・黒字',para('Y=実績期間損害、P=期間提示価格、e=Exposure、U=max(Y−P,0)、V=max(P−Y,0)。赤字割合はtestで実現した契約割合で、会社全体の赤字確率ではない。ESqは全契約のUの上位(1−q)平均で、赤字契約だけの条件付きESではない。端数は比例配分。')+table(tails)+para('FOC比は αΣU / ((1−α)ΣV)。1に近いことは集計でのバランスを示すだけで条件付きExpectileの正しさを保証しない。全体保険料を揃える比較では判別力が弱い。黒字−赤字には事業費・資本コストを含めない。α=.66と.948は3モデルに共通の固定評価水準であり、今回のSeverityモデルの選択αではない。'))
pen=pd.read_csv(ROOT/'asymmetric_penalties.csv');pv=pen.pivot(index='X',columns='model',values='score')[M];ix=pv.div(pv['mean'],axis=0)*100
ix=ix.rename(columns=labels).reset_index();ix.to_csv(OUT/'penalty_indices.csv',index=False)
section('赤字の二乗をX倍する比較',para('L_X=Σ[(X U²+V²)/e] / Σe。Mean=100の指数。小さい方が良い。SeverityモデルをXごとに再学習した結果ではない。赤字額そのものをX倍する線形評価とも異なる。')+table(ix.round(6))+table(pv.rename(columns=labels).reset_index().round(2)))
sc=[]
for alpha in sorted(set([.5,.66,.8,.9,.95,S['selected']['expectile']['alpha']])):
 row={'alpha':alpha}
 for m in M:
  p=df[m+'_period'].to_numpy();u=np.maximum(y-p,0);v=np.maximum(p-y,0)
  row[labels[m]]=np.sum((alpha*u*u+(1-alpha)*v*v)/e)/e.sum()
 sc.append(row)
pb=[]
for alpha in [.948,S['selected']['quantile']['alpha']]:
 row={'alpha':alpha}
 for m in M:
  p=df[m+'_period'].to_numpy();u=np.maximum(y-p,0);v=np.maximum(p-y,0);row[labels[m]]=np.sum(alpha*u+(1-alpha)*v)/e.sum()
 pb.append(row)
pd.DataFrame(sc).to_csv(OUT/'expectile_scores.csv',index=False);pd.DataFrame(pb).to_csv(OUT/'pinball_scores.csv',index=False)
section('α別の非対称スコア',para('非対称二乗スコア=Σ[(αU²+(1−α)V²)/e]/Σe。Pinball=Σ[αU+(1−α)V]/Σe。いずれも小さい方が良い。今回選択したαも追加したが、Frequency×SeverityのQuantile/Expectileは契約総損害の同じαのQuantile/Expectileではなく、最適性は保証されない。')+table(pd.DataFrame(sc).round(3))+table(pd.DataFrame(pb).round(5)))
for title,filename,description in [
 ('Simple Quantile Plot','simple_quantile.png','各モデル自身の年額提示価格でExposureを10等分。各系列を自身の全体平均で指数化。'),
 ('Double Lift Chart：全3ペア','double_lift.png','モデル間価格比でExposureを10等分。同じパネルでは共通の契約群で比較。Frequencyが共通のためSeverity比で順位付けし、浮動小数誤差による同順位分割を避ける。'),
 ('Double Lift：Expectile / Mean','double_lift_expectile_mean.png','右端はExpectileがMeanより相対的に高く値付けする群。損害分布の極端テールではない。'),
 ('Loss Ratio Chart','loss_ratio.png','自身の年額価格の十分位ごとに、実績損害/提示額を表示。縦軸はパネルごとに異なる。100%は損害と提示額が一致する水準。'),
 ('高リスク群の保守性','high_risk_tail.png','左：各モデル自身の上位10%/5%/1%Exposure群。右：Meanの順位を共通にした契約群。群の選び方で結果が変わる。')]:
 section(title,para(description)+fig(filename,description))
ht=pd.read_csv(ROOT/'high_risk_tail.csv');htrows=[]
for _,r in ht.iterrows():
 m=r['model'];htrows.append({'モデル':labels[m],'上位Exposure (%)':100*r.top_exposure,'実績損害 (€)':r.loss,'実績年額損害 (€)':r.actual_rate,'提示年額 (€)':r[m+'_rate'],'損害率 (%)':100*r[m+'_lr'],'全損害の捕捉率 (%)':100*r.loss_share})
common=pd.read_csv(ROOT/'common_mean_rank_groups.csv')[['segment','mean_lr','quantile_lr','expectile_lr']]
section('高リスク群：数表',table(pd.DataFrame(htrows).round(4))+para('以下はMeanの順位で固定した共通群。損害率は比率表示。')+table(common.round(6)))
cirows=[]
for metric,coll in [('top10_capture',A['top10_capture_difference_bootstrap']),*A['error_difference_bootstrap'].items()]:
 for pair,v in coll.items():cirows.append({'metric':metric,'comparison':pair,'estimate':v['estimate'],'low':v['low'],'high':v['high']})
for m,coll in T['paired_poisson_bootstrap_vs_mean'].items():
 for metric,v in coll.items():cirows.append({'metric':metric,'comparison':m+'_minus_mean','estimate':v['delta'],'low':v['low'],'high':v['high']})
ci=pd.DataFrame(cirows);ci.to_csv(OUT/'bootstrap_intervals.csv',index=False)
section('不確実性：paired Poisson bootstrap 600回',para('同じ契約に同じ再標本化重みを付けた差の95%区間。正の差は指標の増加。RMSE・不足額などは負が有利、捕捉率は正が有利。モデル・価格・上位群への割当てを固定し、再学習・α選択の不確実性は含めない。反復ごとの上位群再選択も行わない。多数の指標を見る探索的比較で、多重比較調整なし。')+table(ci.round(6)))
section('十分位の全数表',para('同順位を先に集約し、境界をまたぐ群はExposure比率で実績損害・全モデル提示額を配分する。実績で同順位を並べ替えない。以下は原数値と損害率を含む。')+table(pd.read_csv(ROOT/'simple_quantile_and_lr.csv').round(5)))
section('Double Liftの全数表',table(pd.read_csv(ROOT/'double_lift.csv').round(5)))
section('学習確認と選択過程',para('最大損害を含む全観測Severityを学習に使用。best iterationは0始まり。選択後の保存価格に基づく比較であり、評価工程では再学習しない。')+fig('learning_curves.png','保存済みの選択モデルの学習曲線。trainとvalidをそれぞれ初回損失で指数化。')+table(pd.read_csv(ROOT/'validation_grid.csv').round(6))+table(pd.read_csv(ROOT/'hyperparameter_search.csv').drop(columns=['parameters']).round(6)))
section('深さ1での正則化・サンプリング探索',fig('hyperparameter_regularization.png','各min_child_weightとL2で、サンプリング設定を最適化した固定αの検証損失。')+fig('hyperparameter_sampling.png','各subsampleとcolsample_bytreeで、正則化設定を最適化した固定αの検証損失。'))
refinement=json.loads((ROOT/'final_refinement.json').read_text())
section('チューニング方針',para('学習率0.01、max_depth=1に固定。min_child_weight=50/200/500、reg_lambda=100/1000/10000、subsample=.6/.8/1、colsample_bytree=.6/.8/1の全81通りを各モデルで比較した。初期10水準と追加α探索でvalid損害率50%へ調整。その後、上位3設定を各候補αで本来の検証損失により再比較した。計算量を抑えた有限候補探索であり、大域最適性は主張しない。'))
if (ROOT/'initial_alpha_search.csv').exists():
 section('初期10水準と追加α探索の記録',para('最終の同時探索へ進む直前の、設定ごとのα探索記録。最終モデルの比較表ではなくvalid上の選択過程。各モデルの設定はinitial_alpha_search_parameters.jsonに収録。')+table(pd.read_csv(ROOT/'initial_alpha_search.csv').round(6)))
if (ROOT/'joint_alpha_search.csv').exists():
 section('αと設定を同時に確認した最終探索',para('αを固定するたびに候補設定を本来のvalid損失で比較し、そのαでの損失最小モデルの損害率を使って50%への近さを判定した。native_loss_winner=Trueの行だけがα選択の対象。αが変わると最良設定も変わり得るため、この確認を行った。')+table(pd.read_csv(ROOT/'joint_alpha_search.csv').round(6)))
section('valid RMSEの確認',fig('validation_rmse_curves.png','各モデルのvalid Severity RMSE。採用反復は本来の損失の最小点で、Quantile/ExpectileではRMSEの最小点と一致しない場合がある。'))
dl=pd.read_csv(ROOT/'double_lift.csv').query("comparison == 'expectile_vs_mean'").iloc[-1]
pen_deviation=float(np.nanmax(np.abs(pv.div(pv['mean'],axis=0).to_numpy()*100-100)))
pen_winners=[f"X={x:g}: {labels[pv.loc[x].idxmin()]}" for x in pv.index]
best_gini=max(M,key=lambda m:a[m]['exposure_weighted_gini'])
best_short=min(M,key=lambda m:tail[m]['total_shortfall'])
best_def=min(M,key=lambda m:tail[m]['deficit_probability'])
section('結果の読み取り',para('3モデルともゼロ価格は0件。Severityを正の事故損害で学習しているため、この構成ではゼロ損害が多数ある契約総損害を直接Quantile回帰した場合のゼロ価格の問題は生じていない。')+para(f"Giniの点推定が最大なのは{labels[best_gini]}、赤字額合計が最小なのは{labels[best_short]}、赤字割合が最小なのは{labels[best_def]}。順位付けと不足額は異なる観点なので、単一の勝者にまとめず評価する。")+para('非対称二乗ペナルティの最小モデル：'+'、'.join(pen_winners)+f'。Mean基準の相対差は、全X・モデルを通じて最大{pen_deviation:.4f}%。Severityの損失を最適化しても、Frequencyを掛けた契約全体の非対称損失が最小になる保証はない。')+para(f"Expectile/MeanのDouble Lift第10群では、実績年額損害€{dl.actual_rate:.2f}、Mean提示€{dl.mean_rate:.2f}、Expectile提示€{dl.expectile_rate:.2f}。Expectile提示はMean比{100*(dl.expectile_rate/dl.mean_rate-1):.1f}%高い。この群は価格比による上位群であり、実績損害分布の極端テールそのものではない。高い価格を付けることだけで予測精度の優位性は結論できない。")+para('Bootstrapの区間は固定モデル・固定価格に対するもの。学習とα選択の不確実性、繰り返し参照してきたtestによる探索性は残る。今後の採用判断には別期間などの未使用データによる確認が望ましい。'))
section('再現方法',para('ZIPを展開し、python -m pip install -r analysis/requirements.txt、python run.py の順に実行すると、保存価格から全図表とレポートを再生成する。学習・チューニングから再実行する場合はpython run.py --train。保存モデルによる新規契約への価格付けはpython analysis/predict.py policies.csv quotes.csv。')+para('入力データ、分割ID、乱数seed、採用パラメータ、検証スコア、モデル、価格、依存バージョンとハッシュを同梱する。最大損害を含む全観測損害を保持し、testによるパラメータ選択・価格調整は行わない。'))
overview=next(x for x in sections if x[0]=='結果の読み取り')
sections.remove(overview);sections.insert(1,overview)
css='body{font-family:system-ui,sans-serif;max-width:1250px;margin:36px auto;padding:0 24px;color:#172b3a;line-height:1.65}h1{font-size:28px}h2{margin-top:40px;border-bottom:2px solid #3479a8;padding-bottom:8px}table{border-collapse:collapse;font-size:13px;width:max-content;min-width:70%;max-width:100%}td,th{padding:7px 10px;border-bottom:1px solid #d8e1e8;text-align:right}th{background:#edf3f8}td:first-child,th:first-child{text-align:left}figure{margin:20px 0}img{width:100%;height:auto}figcaption{font-size:13px;color:#546373}nav{background:#edf3f8;padding:14px}section{overflow-x:auto}p{max-width:1050px}@media print{nav{display:none}section{overflow:visible}h2{break-after:avoid}figure,table{break-inside:avoid}}'
nav='<nav>'+''.join(f'<a href="#s{i}">{html.escape(t)}</a>　' for i,(t,_) in enumerate(sections))+'</nav>'
body=''.join(f'<section id="s{i}"><h2>{html.escape(t)}</h2>{b}</section>' for i,(t,b) in enumerate(sections))
(OUT/'all_comparisons.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>Frequency × Severityモデル比較：深さ1・学習率0.01・valid損害率50%</title><style>'+css+'</style><body><h1>Frequency × Severityモデル比較：深さ1・学習率0.01・valid損害率50%</h1>'+nav+body+'</body></html>',encoding='utf-8')
print('overall');print(overall.to_string(index=False));print('downside');print(tails.to_string(index=False));print('CI');print(ci.to_string(index=False))
