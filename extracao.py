import pandas as pd
import glob
import os

caminho = r'C:\Users\Cyroo\Documents\TCC\DIARIO'
arquivos = glob.glob(os.path.join(caminho, "**", "*.csv"), recursive=True)
print(f"Foram encontrados {len(arquivos)} arquivos csv")

if len(arquivos)>0:
    lista_df=[]

    for arquivo in arquivos:
        df_temp = pd.read_csv(arquivo)
        df_temp['aruqivo_origem'] = os.path.basename(arquivo)

        lista_df.append(df_temp)
df_final = pd.concat(lista_df,ignore_index=True)

print(f"Total de linhas no arquivo final: {len(df_final)}")
print(df_final.head())
caminho_salvar = os.path.join(caminho, "base_unificada_tcc.csv")
df_final.to_csv(caminho_salvar, index=False, encoding='utf-8-sig')
print(f"Arquivo salvo em: {caminho_salvar}")
