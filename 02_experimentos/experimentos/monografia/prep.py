import json, pandas as pd, numpy as np
S="/tmp/claude-0/-home-user-TCC/f17788a8-ecde-51ec-9026-5b920241eecd/scratchpad"
R=S+"/refinado/resultados/"; O="/home/user/TCC/02_experimentos/resultados/"; E="/home/user/TCC/02_experimentos/experimentos/resultados/"
M=["INGENUO","ARIMA","PROPHET","LSTM","GRU","BILSTM","BIGRU","NHITS"]
NM={"INGENUO":"Ingênuo sazonal","ARIMA":"SARIMA","PROPHET":"Prophet","LSTM":"LSTM","GRU":"GRU","BILSTM":"Bi-LSTM","BIGRU":"Bi-GRU","NHITS":"N-HiTS"}
PR=["ALAMBARI","BLOQUEIO DE BOITUVA","BOITUVA","PORTO FELIZ","ITU","SOROCABA"]
NP={"ALAMBARI":"Alambari","BLOQUEIO DE BOITUVA":"Bloqueio de Boituva","BOITUVA":"Boituva","PORTO FELIZ":"Porto Feliz","ITU":"Itu","SOROCABA":"Sorocaba"}
f=lambda x,n=2:("—" if pd.isna(x) else f"{x:.{n}f}".replace(".",","))
rd=lambda p:pd.read_csv(p,encoding="utf-8-sig")
out={}
# antes/depois vs oficial
def ler(p):
    d=rd(p); d["ORIGEM"]=pd.to_datetime(d.ORIGEM).dt.date; return d
ad=[]
for m,nov,hs in [("ARIMA",E+"sarima__menor_aic.csv",(7,90,180)),("PROPHET",E+"prophet__janelas_longas.csv",(7,90,180)),("LSTM",E+"redes/paciencia40_LSTM_180_por_origem.csv",(180,))]:
    a=ler(O+f"{m.lower()}_metricas_por_origem.csv"); b=ler(nov)
    j=a.merge(b,on=["PRACA","DIAS","ORIGEM"],suffixes=("_A","_D"))
    for h in hs:
        s=j[j.DIAS==h]; pp=s.groupby("PRACA")[["SMAPE_A","SMAPE_D"]].mean()
        A,D=pp.SMAPE_A.mean(),pp.SMAPE_D.mean()
        ad.append([NM[m],f"{h} dias",f(A),f(D),f(round(D,2)-round(A,2)).replace("-","−"),f"{100*(s.SMAPE_D<s.SMAPE_A).mean():.0f}%",f"{int((pp.SMAPE_D<pp.SMAPE_A).sum())} de 6"])
out["antes_depois"]=ad
t8=rd(R+"completos_tabela8_consolidada.csv")
out["t8"]=[[f"{int(r.DIAS)} dias",NM[r.MODELO],f(r.SMAPE),f(r.MAPE),f(r.MAE_REL),f(r.RMSE_REL),f(r.DTW_REL),("—" if pd.isna(r.PCT_VENCE_INGENUO) else f"{r.PCT_VENCE_INGENUO:.0f}%")] for r in t8.itertuples()]
t4=rd(R+"completos_tabela4_smape_praca.csv")
def porpraca(t,n):
    rows=[]
    for p in PR:
        for h in (7,90,180):
            r=t[(t.PRACA==p)&(t.DIAS==h)].iloc[0]; v=[r[m] for m in M]; mn=min(v)
            rows.append([NP[p],str(h)]+[("*"+f(x,n)) if x==mn else f(x,n) for x in v])
    return rows
out["t4"]=porpraca(t4,2)
t7=rd(R+"completos_tabela7_dtw_praca.csv"); out["t7"]=porpraca(t7,0)
t9=rd(R+"completos_tabela9_tipo_semana.csv")
out["t9"]=[[r.TIPO[0].upper()+r.TIPO[1:],str(int(r.JANELAS))]+[f"{f(getattr(r,'SMAPE_'+m),1)} ({getattr(r,'PARTICIPACAO_ERRO_'+m):.0f}%)" for m in M] for r in t9.itertuples() if r.TIPO!="todas"]
fx=rd(R+"completos_faixas_antecedencia.csv").groupby("MODELO")[["DIAS_1_7","DIAS_8_90","DIAS_91_180"]].mean()
out["faixas"]=[[NM[m]]+[f(fx.loc[m,c]) for c in fx.columns] for m in M]
p=rd(R+"testes_blocos_por_origem_pares.csv"); p7=p[p.DIAS==7]
mat=[]
for i,a in enumerate(M[1:],1):
    row=[NM[a]]
    for b in M[:-1]:
        q=p7[((p7.A==a)&(p7.B==b))|((p7.A==b)&(p7.B==a))]
        if M.index(b)>=i: row.append("")
        else:
            v=q.P_HOLM.iloc[0]; row.append(("*" if v<0.05 else "")+("< 0,001" if v<0.001 else f(v,3)))
    mat.append(row)
out["sig"]=mat; out["sig_cols"]=[NM[m] for m in M[:-1]]
fr=rd(R+"testes_blocos_por_origem_friedman.csv"); out["friedman"]=[[f"{int(r.DIAS)} dias",str(int(r.BLOCOS_POR_ORIGEM)),("< 0,001" if r.FRIEDMAN_P_POR_ORIGEM<0.001 else f(r.FRIEDMAN_P_POR_ORIGEM,3))] for r in fr.itertuples()]
o=rd(S+"/zip/02_codigo_e_resultados/diagnostico/sarima_busca_completa_ordens.csv")
out["ordens"]=[[NP[r.PRACA],r.ORDEM_ATUAL,f(r.AIC_FINAL_MILHARES_ATUAL,1),r.ORDEM_MENOR_AIC_AJUSTE_FINAL,f(r.AIC_B_MENOR_AIC_AJUSTE_FINAL,1),f(r.LJUNGBOX_P14_MENOR_AIC_AJUSTE_FINAL,3)] for r in o.set_index("PRACA").loc[PR].reset_index().itertuples()]
json.dump(out,open(S+"/export/dados.json","w"),ensure_ascii=False,indent=0)
for k,v in out.items(): print(k, v[:3] if isinstance(v,list) else v)
