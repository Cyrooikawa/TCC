import json, glob, math, pandas as pd, numpy as np
S="/tmp/claude-0/-home-user-TCC/f17788a8-ecde-51ec-9026-5b920241eecd/scratchpad"; R=S+"/refinado/resultados/"
M=["INGENUO","ARIMA","PROPHET","LSTM","GRU","BILSTM","BIGRU","NHITS"]; NETS=M[3:]
NM={"INGENUO":"Ingênuo sazonal","ARIMA":"SARIMA","PROPHET":"Prophet","LSTM":"LSTM","GRU":"GRU","BILSTM":"Bi-LSTM","BIGRU":"Bi-GRU","NHITS":"N-HiTS"}
PR=["ALAMBARI","BLOQUEIO DE BOITUVA","BOITUVA","PORTO FELIZ","ITU","SOROCABA"]
NP={"ALAMBARI":"Alambari","BLOQUEIO DE BOITUVA":"Bloqueio de Boituva","BOITUVA":"Boituva","PORTO FELIZ":"Porto Feliz","ITU":"Itu","SOROCABA":"Sorocaba"}
def f(x,n=2):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "–"
    return f"{x:,.{n}f}".replace(",","X").replace(".",",").replace("X",".")
B=lambda s,c:("*"+s) if c else s
rd=lambda p:pd.read_csv(p,encoding="utf-8-sig")
d=pd.concat([pd.read_csv(x) for x in glob.glob(R+"*_metricas_por_origem.csv")]); d["ORIGEM"]=pd.to_datetime(d.ORIGEM)
o={}
o["t1"]=[["Alambari","(3,1,2)","(1,0,2,7)"],["Bloqueio de Boituva","(0,1,2)","(2,0,1,7)"],["Boituva","(3,1,2)","(1,0,2,7)"],["Porto Feliz","(3,1,2)","(1,0,2,7)"],["Itu","(3,1,2)","(1,0,2,7)"],["Sorocaba","(3,1,2)","(1,0,2,7)"]]
def porpraca(col,n,mods=M,cmp=min):
    g=d.groupby(["PRACA","DIAS","MODELO"])[col].mean().unstack()
    rows=[]
    for p in PR:
        for h in (7,90,180):
            v=[g.loc[(p,h),m] for m in mods]; mn=cmp(v)
            rows.append([NP[p],str(h)]+[B(f(x,n),x==mn) for x in v])
    return rows
o["t4"]=porpraca("SMAPE",1); o["t7"]=porpraca("DTW",0)
w=d.pivot_table(index=["PRACA","DIAS","ORIGEM"],columns="MODELO",values="SMAPE")
v=(w[M[1:]].lt(w.INGENUO,axis=0)).groupby(level=["PRACA","DIAS"]).mean()*100
rows=[[NP[p],str(h)]+[f"{v.loc[(p,h),m]:.0f}" for m in M[1:]] for p in PR for h in (7,90,180)]
vm=v.groupby(level="DIAS").mean()
rows+= [["Média das praças" if h==7 else "",str(h)]+[f"{vm.loc[h,m]:.0f}" for m in M[1:]] for h in (7,90,180)]
o["t5"]=rows
am=rd(R+"arima_modelos.csv").set_index("PRACA")
pv=lambda x:"< 0,001" if x<0.001 else f(x,3)
o["t6"]=[[NP[p],"(3,1,2)(1,0,2,7)" if p!="BLOQUEIO DE BOITUVA" else "(0,1,2)(2,0,1,7)",f(am.loc[p,"ADF_P_VALOR"],3),pv(am.loc[p,"LJUNGBOX_P_7DIAS"]),pv(am.loc[p,"LJUNGBOX_P_14DIAS"])] for p in PR]
t8=rd(R+"completos_tabela8_consolidada.csv"); rows=[]
for h in (7,90,180):
    g=t8[t8.DIAS==h].set_index("MODELO")
    for m in M:
        r=g.loc[m]
        rows.append([f"{h} dias",NM[m],B(f(r.SMAPE,1),r.SMAPE==g.SMAPE.min()),f"{f(r.SMAPE_MIN,1)} a {f(r.SMAPE_MAX,1)}",B(f(r.MAPE,1),r.MAPE==g.MAPE.min()),
          B(f(r.MAE_REL),r.MAE_REL==g.MAE_REL.min()),B(f(r.RMSE_REL),r.RMSE_REL==g.RMSE_REL.min()),B(f(r.DTW_REL),r.DTW_REL==g.DTW_REL.min()),
          "–" if pd.isna(r.PCT_VENCE_INGENUO) else f"{r.PCT_VENCE_INGENUO:.0f}",B(f(r.POSTO_MEDIO),r.POSTO_MEDIO==g.POSTO_MEDIO.min())])
o["t8"]=rows
t9=rd(R+"completos_tabela9_tipo_semana.csv").set_index("TIPO"); rows=[]
for k,nome in [("semana comum","Semana comum"),("feriado no período previsto","Feriado no período previsto"),("semana seguinte a feriado","Semana seguinte a feriado"),("todas","Todas")]:
    r=t9.loc[k]; vals=[r["SMAPE_"+m] for m in M]; mn=min(vals); rows.append([nome,str(int(r.JANELAS))]+[B(f(x,1),x==mn) for x in vals])
o["t9"]=rows
fh=rd(R+"faixas_redes_por_horizonte_treinado.csv").pivot_table(index=["MODELO","REDE_DE"],columns="FAIXA",values="SMAPE")
rows=[]
for m in NETS:
    for h in (7,90,180):
        r=fh.loc[(m,h)]; rows.append([NM[m],f"{h} dias"]+[f(r.get(c),1) if pd.notna(r.get(c)) else "–" for c in ["DIAS_1_7","DIAS_8_90","DIAS_91_180"]])
o["t10"]=rows
fr=rd(R+"testes_blocos_por_origem_friedman.csv").set_index("DIAS")
mo=w.groupby(level=["DIAS","ORIGEM"]).mean()
mo=d.pivot_table(index=["DIAS","ORIGEM","PRACA"],columns="MODELO",values="SMAPE")[M].groupby(level=["DIAS","ORIGEM"]).mean()
rows=[]
for h in (7,90,180):
    x=mo.loc[h].iloc[::math.ceil(h/7)]; pr=x.rank(axis=1).mean(); mn=pr.min()
    p=fr.loc[h,"FRIEDMAN_P_POR_ORIGEM"]
    rows.append([f"{h} dias",str(len(x)),pv(p)]+[B(f(pr[m]),pr[m]==mn) for m in M])
o["t11"]=rows
pa=rd(R+"testes_blocos_por_origem_pares.csv"); rows=[]
pt=d.pivot_table(index=["DIAS","ORIGEM","PRACA"],columns="MODELO",values="SMAPE")[M]
for m in [k for k in M if k!="PROPHET"]:
    l=[f"Prophet × {NM[m]}"]
    q=pa[(pa.DIAS==7)&(((pa.A=="PROPHET")&(pa.B==m))|((pa.B=="PROPHET")&(pa.A==m)))].iloc[0]
    dif,pct=(q.DIF_MEDIANA_B_MENOS_A_PP,q.PCT_BLOCOS_A_MENOR) if q.A=="PROPHET" else (-q.DIF_MEDIANA_B_MENOS_A_PP,100-q.PCT_BLOCOS_A_MENOR)
    l.append(f"{f(dif)} p.p.; {pct:.0f}%; p {pv(q.P_HOLM) if q.P_HOLM<0.001 else '= '+pv(q.P_HOLM)}")
    for h in (90,180):
        xa=mo.loc[h]; prc=pt.loc[h].groupby(level="PRACA").mean()
        l.append(f"{f((xa[m]-xa.PROPHET).median())} p.p.; {100*(xa.PROPHET<xa[m]).mean():.0f}%; {int((prc.PROPHET<prc[m]).sum())} de 6")
    rows.append(l)
o["t12"]=rows
sens=rd(R+"completos_tabela11_sensibilidade.csv").set_index("DIAS"); un=rd(R+"sensibilidade_prophet_ajuste_unico.csv").groupby(["DIAS","PRACA"]).SMAPE.mean().groupby("DIAS").mean()
L=lambda vals:[f(x,1) for x in vals]
best=[sens.loc[h,NETS].min() for h in (7,90,180)]
o["t13"]=[["Ingênuo sazonal"]+L([sens.loc[h,"INGENUO"] for h in (7,90,180)]),
 ["SARIMA, menor AIC na busca exaustiva (avaliação principal)"]+L([sens.loc[h,"ARIMA"] for h in (7,90,180)]),
 ["SARIMA, busca passo a passo do auto_arima (configuração original)"]+L([6.765359,7.660683,7.604348]),
 ["SARIMA, busca exaustiva com o critério do auto_arima"]+L([6.83,9.93,11.00]),
 ["ARIMA sem componente sazonal"]+L([sens.loc[h,"ARIMA_SEM_SAZONALIDADE"] for h in (7,90,180)]),
 ["Prophet (avaliação principal)"]+L([sens.loc[h,"PROPHET"] for h in (7,90,180)]),
 ["Prophet com a mesma janela para todos os feriados (configuração original)"]+L([4.635816,4.709156,4.681025]),
 ["Prophet com ajuste único no treino"]+L([un.loc[h] for h in (7,90,180)]),
 ["Prophet sem calendário de feriados"]+L([sens.loc[h,"PROPHET_SEM_FERIADOS"] for h in (7,90,180)]),
 ["Prophet sem feriados e sem sazonalidade anual"]+L([sens.loc[h,"PROPHET_SEM_FERIADOS_SEM_ANUAL"] for h in (7,90,180)]),
 ["Melhor rede (N-HiTS em 7 e 180 dias; GRU em 90)"]+L(best)]
sm=rd(R+"completos_redes_sementes.csv")
o["t14"]=[[f"{int(r.DIAS)} dias",NM[r.MODELO],f"{f(r.SMAPE_MEDIA_SEMENTES)} ± {f(r.SMAPE_DP_SEMENTES)}",f"{r.MELHOR_EPOCA_MIN} a {r.MELHOR_EPOCA_MAX}",str(r.EPOCAS_EXECUTADAS),f"{r.TEMPO_TREINO_MIN:.0f}",f(r.PARAMETROS_POR_SEMENTE,0)] for r in sm.itertuples()]
orig={"ALAMBARI":("(0,1,1)(0,0,1)₇","7,0","7,7","7,7"),"BLOQUEIO DE BOITUVA":("(2,1,1)(1,0,1)₇","5,3","5,5","5,4"),"BOITUVA":("(3,1,2)(0,0,2)₇","9,6","11,1","10,8"),
      "PORTO FELIZ":("(2,1,0)(1,0,1)₇","7,1","8,7","8,7"),"ITU":("(0,1,2)(0,0,2)₇","6,5","7,4","7,4"),"SOROCABA":("(2,1,2)(1,0,1)₇","5,0","5,6","5,7")}
g=d[d.MODELO=="ARIMA"].groupby(["PRACA","DIAS"]).SMAPE.mean()
rows=[[NP[p],*orig[p][:1],*orig[p][1:],"(0,1,2)(2,0,1)₇" if p=="BLOQUEIO DE BOITUVA" else "(3,1,2)(1,0,2)₇"]+[f(g.loc[(p,h)],1) for h in (7,90,180)] for p in PR]
rows.append(["Média","","6,8","7,7","7,6",""]+[f(g.groupby("DIAS").mean().loc[h],1) for h in (7,90,180)])
o["t15"]=rows
json.dump(o,open(S+"/export/cap4.json","w"),ensure_ascii=False)
for k,v in o.items(): print(k, v[:2])
