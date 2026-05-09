import pandas as pd
import glob
import os

caminho = r'C:\Users\Cyroo\Documents\TCC\DIARIO'
# Procura todos os CSVs
arquivos = glob.glob(os.path.join(caminho, "**", "*.csv"), recursive=True)

# Nome do arquivo final que queremos salvar
nome_arquivo_saida = "base_unificada_tcc.csv"
caminho_salvar = os.path.join(caminho, nome_arquivo_saida)

print(f"Foram encontrados {len(arquivos)} arquivos csv.")

if len(arquivos) > 0:
    lista_df = []

    for arquivo in arquivos:
        # A MÁGICA AQUI: Se o arquivo atual for o próprio arquivo unificado, ele pula e não lê!
        if os.path.basename(arquivo) == nome_arquivo_saida:
            print("Ignorando arquivo unificado antigo para evitar duplicatas...")
            continue 

        # Adicionei dtype=str para evitar os alertas vermelhos de DtypeWarning
        df_temp = pd.read_csv(arquivo, dtype=str, low_memory=False)
        
        # Correção do erro de digitação
        df_temp['arquivo_origem'] = os.path.basename(arquivo)

        lista_df.append(df_temp)
    
    # Verifica se a lista não ficou vazia após ignorar o arquivo de saída
    if len(lista_df) > 0:
        print("\nUnificando os dados. Isso pode demorar alguns segundos...")
        df_final = pd.concat(lista_df, ignore_index=True)

        print(f"Total de linhas no arquivo final: {len(df_final)}")
        print(df_final.head())
        
        # Salvando
        df_final.to_csv(caminho_salvar, index=False, encoding='utf-8-sig')
        print(f"\n✅ Arquivo salvo em: {caminho_salvar}")
    else:
        print("Nenhum arquivo novo para unificar.")

else:
    print("Nenhum arquivo CSV foi encontrado na pasta especificada.")