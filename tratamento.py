import pandas as pd
import os

caminho_base_unificada = r'C:\Users\Cyroo\Documents\TCC\DIARIO\base_unificada_tcc.csv'
df_total = pd.read_csv(caminho_base_unificada)

# Municipios da região metropolitana de Sorocaba
termos_rms = [
    'SOROCABA', 'VOTORANTIM', 'ITU', 'SALTO', 'TATUI', 'BOITUVA', 
    'PORTO FELIZ', 'ROQUE', 'IBIUNA', 'PIEDADE', 'ARAÇOIABA', 
    'ALUMINIO', 'ALAMBARI', 'QUADRA', 'MAIRINQUE' 
]

padrao = '|'.join(termos_rms)
coluna_filtro = 'PRACA' 

if coluna_filtro in df_total.columns:
    df_rms = df_total[df_total[coluna_filtro].str.contains(padrao, na=False, case=False)].copy()

    # -------------------------------------------------------------------------
    # 2. FILTRO DE EXCLUSÃO (MUDANÇA AQUI: fazer ANTES de limpar os nomes)
    # -------------------------------------------------------------------------
    cidades_excluir = ['ITUPEVA', 'ITUVERAVA', 'ANHANGUERA']
    df_rms = df_rms[~df_rms[coluna_filtro].str.contains('|'.join(cidades_excluir), na=False, case=False)]

    # 3. Fazemos o split inicial
    df_separado = df_rms['PRACA'].str.split(' - ', n=1, expand=True)
    df_rms['PONTO_ID'] = df_separado[0].str.strip()
    df_rms['CIDADE_NOME'] = df_separado[1].fillna(df_rms['PRACA']).str.strip()

    def limpar_cidade(row):
        cidade = str(row['CIDADE_NOME']).upper()
        ponto = str(row['PONTO_ID']).upper()
        
        if cidade.isdigit():
            for t in termos_rms:
                if t in ponto:
                    return t
        
        for t in termos_rms:
            if t in cidade:
                return t
        
        return cidade

    print("Padronizando nomes dos municípios...")
    df_rms['CIDADE_NOME'] = df_rms.apply(limpar_cidade, axis=1)

    # Resultado Final
    print("\n--- AMOSTRA DOS DADOS TRATADOS ---")
    print(df_rms[['PRACA', 'PONTO_ID', 'CIDADE_NOME']].drop_duplicates().sort_values('CIDADE_NOME'))

    # Salvar
    caminho_saida = r'C:\Users\Cyroo\Documents\TCC\DIARIO\base_RMS_final.csv'
    df_rms.to_csv(caminho_saida, index=False, encoding='utf-8-sig')