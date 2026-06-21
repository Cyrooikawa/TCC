import pandas as pd
import glob
import os

caminho = r'C:\Users\Cyroo\Documents\TCC\DIARIO'
arquivos = glob.glob(os.path.join(caminho, "**", "*.csv"), recursive=True)

nome_arquivo_saida = "base_unificada_tcc.csv"
caminho_salvar = os.path.join(caminho, nome_arquivo_saida)


print(f"Foram encontrados {len(arquivos)} arquivos csv.")

if len(arquivos) > 0:
    lista_df = []

    for arquivo in arquivos:
        if os.path.basename(arquivo) == nome_arquivo_saida:
            continue 

        df_temp = pd.read_csv(arquivo, dtype=str, low_memory=False)
        
        df_temp['arquivo_origem'] = os.path.basename(arquivo)

        lista_df.append(df_temp)
    
    if len(lista_df) > 0:
        df_final = pd.concat(lista_df, ignore_index=True)

        print(f"Total de linhas no arquivo final: {len(df_final)}")
        print(df_final.head())
        
        # Salvando
        df_final.to_csv(caminho_salvar, index=False, encoding='utf-8-sig')
        print(f"\nArquivo salvo em: {caminho_salvar}")
    else:
        print("Nenhum arquivo novo para unificar.")

else:
    print("Nenhum arquivo CSV foi encontrado na pasta especificada.")
