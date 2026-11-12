import json, glob, pandas as pd, numpy as np
S="/tmp/claude-0/-home-user-TCC/f17788a8-ecde-51ec-9026-5b920241eecd/scratchpad"
R=S+"/refinado/resultados/"
M=["INGENUO","ARIMA","PROPHET","LSTM","GRU","BILSTM","BIGRU","NHITS"]
NM={"INGENUO":"Ingênuo sazonal","ARIMA":"SARIMA","PROPHET":"Prophet","LSTM":"LSTM","GRU":"GRU","BILSTM":"Bi-LSTM","BIGRU":"Bi-GRU","NHITS":"N-HiTS"}
PR=["ALAMBARI","BLOQUEIO DE BOITUVA","BOITUVA","PORTO FELIZ","ITU","SOROCABA"]
NP={"ALAMBARI":"Alambari","BLOQUEIO DE BOITUVA":"Bloqueio de Boituva","BOITUVA":"Boituva","PORTO FELIZ":"Porto Feliz","ITU":"Itu","SOROCABA":"Sorocaba"}
def f(x,n=2):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "–"
    s=f"{x:,.{n}f}".replace(",","X").replace(".",",").replace("X",".")
    return s
B=lambda s,cond:("*"+s) if cond else s
rd=lambda p:pd.read_csv(p,encoding="utf-8-sig")
out=json.load(open(S+"/export/dados.json"))
d=pd.concat([pd.read_csv(x) for x in glob.glob(R+"*_metricas_por_origem.csv")])
# Tabela consolidada
t8=rd(R+"completos_tabela8_consolidada.csv"); rows=[]
for h in (7,90,180):
    g=t8[t8.DIAS==h].set_index("MODELO")
    for m in M:
        r=g.loc[m]
        rows.append([f"{h} dias",NM[m],B(f(r.SMAPE,1),r.SMAPE==g.SMAPE.min()),f"{f(r.SMAPE_MIN,1)} a {f(r.SMAPE_MAX,1)}",B(f(r.MAPE,1),r.MAPE==g.MAPE.min()),
          B(f(r.MAE_REL),r.MAE_REL==g.MAE_REL.min()),B(f(r.RMSE_REL),r.RMSE_REL==g.RMSE_REL.min()),B(f(r.DTW_REL),r.DTW_REL==g.DTW_REL.min()),
          "–" if pd.isna(r.PCT_VENCE_INGENUO) else f"{r.PCT_VENCE_INGENUO:.0f}",B(f(r.POSTO_MEDIO),r.POSTO_MEDIO==g.POSTO_MEDIO.min())])
out["t10"]=rows
# tipo de semana
t9=rd(R+"completos_tabela9_tipo_semana.csv").set_index("TIPO"); nomes={"semana comum":"Semana comum","feriado no período previsto":"Feriado no período previsto","semana seguinte a feriado":"Semana seguinte a feriado","todas":"Todas"}
rows=[]
for k,v in nomes.items():
    r=t9.loc[k]; vals=[r["SMAPE_"+m] for m in M]; mn=min(vals)
    rows.append([v,str(int(r.JANELAS))]+[B(f(x,1),x==mn) for x in vals])
out["t11"]=rows
# Friedman e postos
t10=rd(R+"completos_tabela10_significancia.csv"); rows=[]
for r in t10.itertuples():
    ps=[getattr(r,"POSTO_"+m) for m in M]; mn=min(ps)
    rows.append([f"{int(r.DIAS)} dias",str(int(r.BLOCOS)),"< 0,001" if r.FRIEDMAN_P<0.001 else f(r.FRIEDMAN_P,3)]+[B(f(x),x==mn) for x in ps])
out["t12"]=rows
# Prophet x outros
pe=rd(R+"completos_pares_efeito.csv"); rows=[]
def pv(p): return "p < 0,001" if p<0.001 else "p = "+f(p,3)
for m in [x for x in M if x!="PROPHET"]:
    linha=[f"Prophet × {NM[m]}"]
    for h in (7,90,180):
        q=pe[(pe.DIAS==h)&(((pe.A=="PROPHET")&(pe.B==m))|((pe.B=="PROPHET")&(pe.A==m)))].iloc[0]
        if q.A=="PROPHET": dif,pct=q.DIF_MEDIANA_B_MENOS_A_PP,q.PCT_BLOCOS_A_MENOR
        else: dif,pct=-q.DIF_MEDIANA_B_MENOS_A_PP,100-q.PCT_BLOCOS_A_MENOR
        linha.append(f"{f(dif)} p.p.; {pct:.0f}%; {pv(q.P_HOLM)}")
    rows.append(linha)
out["t13"]=rows
# sensibilidade
t11=rd(R+"completos_tabela11_sensibilidade.csv").set_index("DIAS")
sel=[("Ingênuo sazonal","INGENUO"),("SARIMA, ordem de menor AIC (avaliação principal)","ARIMA"),("SARIMA, ordem da busca passo a passo",None),("ARIMA sem componente sazonal","ARIMA_SEM_SAZONALIDADE"),
     ("Prophet com janelas ajustadas, reajustado em cada origem (avaliação principal)","PROPHET"),("Prophet com janela padrão, reajustado em cada origem",None),("Prophet com janela padrão e ajuste único no treino","PROPHET_AJUSTE_UNICO"),
     ("Prophet sem feriados","PROPHET_SEM_FERIADOS"),("Prophet sem feriados e sem sazonalidade anual","PROPHET_SEM_FERIADOS_SEM_ANUAL")]
orig={"SARIMA, ordem da busca passo a passo":(6.765359,7.660683,7.604348),"Prophet com janela padrão, reajustado em cada origem":(4.635816,4.709156,4.681025)}
out["t14"]=[[n]+([f(x,1) for x in orig[n]] if c is None else [f(t11.loc[h,c],1) for h in (7,90,180)]) for n,c in sel]
# sementes
sm=rd(R+"completos_redes_sementes.csv"); rows=[]
for r in sm.itertuples():
    rows.append([f"{int(r.DIAS)} dias",NM[r.MODELO],f"{f(r.SMAPE_MEDIA_SEMENTES)} ± {f(r.SMAPE_DP_SEMENTES)}",f"{r.MELHOR_EPOCA_MIN} a {r.MELHOR_EPOCA_MAX}",str(r.EPOCAS_EXECUTADAS),f"{r.TEMPO_TREINO_MIN:.0f}",f(r.PARAMETROS_POR_SEMENTE,0)])
out["t15"]=rows
# apêndice A
w=d.pivot_table(index=["PRACA","DIAS","ORIGEM"],columns="MODELO",values="SMAPE")
vence=w.apply(lambda c:(c<w.INGENUO)).groupby(level=["PRACA","DIAS"]).mean()*100
g=d.groupby(["PRACA","DIAS","MODELO"])[["MAE","RMSE","MAPE","SMAPE","DTW"]].mean()
rows=[]
for p in PR:
    for h in (7,90,180):
        gg=g.loc[(p,h)]
        for m in M:
            r=gg.loc[m]; ism=lambda c,n: B(f(r[c],n), r[c]==gg[c].min())
            rows.append([NP[p],f"{h} dias",NM[m],ism("MAE",0),ism("RMSE",0),ism("MAPE",2),ism("SMAPE",2),ism("DTW",0),"–" if m=="INGENUO" else f(vence.loc[(p,h),m],1)])
out["t16"]=rows
# apêndice B
def pp(arq,n):
    t=rd(R+arq); rows=[]
    for p in PR:
        for h in (7,90,180):
            r=t[(t.PRACA==p)&(t.DIAS==h)].iloc[0]; v=[r[m] for m in M]; mn=min(v)
            rows.append([NP[p],f"{h} dias"]+[B(f(x,n),x==mn) for x in v])
    return rows
out["t4"]=pp("completos_tabela4_smape_praca.csv",1); out["t7"]=pp("completos_tabela7_dtw_praca.csv",0)
json.dump(out,open(S+"/export/sec4.json","w"),ensure_ascii=False)
for k in ["t10","t11","t12","t13","t14","t15"]: print(k, out[k][:2])
print(out["t16"][:3], len(out["t16"]))
