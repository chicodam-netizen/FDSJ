import os
import streamlit as st
import pandas as pd
from groq import Groq
import io
from datetime import datetime, timedelta
import re
import json
import numpy as np
import traceback
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Configura a página do Streamlit
st.set_page_config(
    page_title="Reestruturação industrial - QRZ",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CONSTANTES ====================

# Mapeamento completo das colunas do tarefas.csv baseado na estrutura fornecida
COLUNAS_TAREFAS = {
    'OQUE': 'Tarefa - product_backlog',
    'ONDE': 'Setor - grooming_detalhamento',
    'PORQUE': 'origem_nao_conformidade',
    'R': 'resultado_rei',
    'E': 'execucao_rei',
    'I': 'investimento_rei',
    'REI': 'matriz_rei',
    'G': 'gravidade_gut',
    'U': 'urgencia_gut',
    'T': 'tendencia_gut',
    'GUT': 'matriz_gut',
    'GUTREI': 'soma_gut_rei',
    'COMO': 'detalhamento_acoes',
    'PERC_OK': 'percentual_ok',
    'QUEM': 'Pessoa responsável pela tarefa',
    'PREV_CONCLUSAO': 'Data da previsao de conclusão da Tarefa',
    'DATA_CONCLUSAO': 'Data de conclusao da Tarefa'
 }

# ==================== FUNÇÕES DE UTILITÁRIOS ====================

def formatar_data_segura(data, formato='%d/%m/%Y'):
    """Formata uma data com segurança"""
    if pd.isna(data) or data is None:
        return "Data não disponível"
    try:
        return data.strftime(formato)
    except:
        return "Data inválida"

def formatar_numero_seguro(valor, formato="R$ {:,.2f}"):
    """Formata um número com segurança"""
    if pd.isna(valor) or valor is None:
        return "Valor não disponível"
    try:
        if formato.startswith("R$"):
            return f"R$ {valor:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        else:
            return formato.format(valor)
    except:
        return "Valor inválido"

def classificar_gut(gut_score):
    """Classifica o score GUT em prioridades"""
    if pd.isna(gut_score) or gut_score is None:
        return "Não classificado"
    try:
        gut_score = float(gut_score)
        if gut_score >= 60:
            return "Crítica"
        elif gut_score >= 40:
            return "Alta"
        elif gut_score >= 20:
            return "Média"
        elif gut_score >= 1:
            return "Baixa"
        else:
            return "Sem prioridade"
    except:
        return "Erro na classificação"

def classificar_rei(rei_score):
    """Classifica o score REI em níveis de esforço/impacto"""
    if pd.isna(rei_score) or rei_score is None:
        return "Não classificado"
    try:
        rei_score = float(rei_score)
        if rei_score >= 60:
            return "Alto impacto"
        elif rei_score >= 30:
            return "Médio impacto"
        elif rei_score >= 1:
            return "Baixo impacto"
        else:
            return "Sem impacto"
    except:
        return "Erro na classificação"

# ==================== FUNÇÕES DE EXPORTAÇÃO ====================

def exportar_resposta_txt(resposta, nome_arquivo=None):
    """Exporta a resposta para arquivo .txt"""
    if nome_arquivo is None:
        nome_arquivo = f"resposta_analise_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    return resposta.encode('utf-8'), nome_arquivo

# ==================== FUNÇÕES DE LIMPEZA DE DADOS ====================

def limpar_valor_numerico(valor, nome_coluna=""):
    """
    Limpa e converte um valor para numérico com debug
    """
    if pd.isna(valor) or valor is None or valor == '':
        return None
    
    try:
        # Se já for numérico, retorna direto
        if isinstance(valor, (int, float)):
            return float(valor)
        
        # Converte para string e limpa
        str_valor = str(valor).strip()
        if not str_valor or str_valor.lower() in ['nan', 'null', 'none', '']:
            return None
        
        # Remove caracteres especiais, mantém números, vírgula, ponto e sinal negativo
        str_valor = re.sub(r'[^\d,\-\.]', '', str_valor)
        
        # Substitui vírgula por ponto (padrão brasileiro)
        str_valor = str_valor.replace(',', '.')
        
        # Corrige múltiplos pontos (ex: 1.234.56 -> 1234.56)
        partes = str_valor.split('.')
        if len(partes) > 2:
            str_valor = ''.join(partes[:-1]) + '.' + partes[-1]
        
        # Remove pontos que são separadores de milhar (ex: 1.234 -> 1234)
        if str_valor.count('.') == 1 and len(str_valor.split('.')[0]) > 3:
            str_valor = str_valor.replace('.', '')
        
        if str_valor and str_valor != '.':
            return float(str_valor)
        else:
            return None
            
    except Exception as e:
        return None

def limpar_coluna_numerica(serie, nome_coluna=""):
    """Aplica limpeza em uma série/coluna inteira"""
    return serie.apply(lambda x: limpar_valor_numerico(x, nome_coluna))

def limpar_dataframe_tarefas(df):
    """
    Limpa especificamente o dataframe de tarefas com foco nas matrizes GUT e REI
    """
    df_limpo = df.copy()
    
    try:
        # Identifica as colunas disponíveis
        colunas_disponiveis = df_limpo.columns.tolist()
        
        # Mapeamento das colunas importantes com seus nomes originais
        mapeamento_colunas = {
            'R': 'resultado_rei',
            'E': 'execucao_rei',
            'I': 'investimento_rei',
            'REI': 'matriz_rei',
            'G': 'gravidade_gut',
            'U': 'urgencia_gut',
            'T': 'tendencia_gut',
            'GUT': 'matriz_gut',
            'GUTREI': 'soma_gut_rei',
            'PERC_OK': 'percentual_ok'
        }
        
        # Renomeia as colunas que existem
        rename_dict = {}
        for col_original, col_novo in mapeamento_colunas.items():
            if col_original in df_limpo.columns:
                rename_dict[col_original] = col_novo
        
        if rename_dict:
            df_limpo = df_limpo.rename(columns=rename_dict)
        
        # Processa colunas da matriz REI
        colunas_rei = ['resultado_rei', 'execucao_rei', 'investimento_rei']
        for coluna in colunas_rei:
            if coluna in df_limpo.columns:
                df_limpo[coluna] = pd.to_numeric(df_limpo[coluna], errors='coerce')
                if df_limpo[coluna].isna().sum() > len(df_limpo) * 0.5:
                    df_limpo[coluna] = limpar_coluna_numerica(df_limpo[coluna], coluna)
        
        # Processa matriz_rei (R*E*I)
        if 'matriz_rei' in df_limpo.columns:
            df_limpo['matriz_rei'] = pd.to_numeric(df_limpo['matriz_rei'], errors='coerce')
            if df_limpo['matriz_rei'].isna().sum() > len(df_limpo) * 0.5:
                df_limpo['matriz_rei'] = limpar_coluna_numerica(df_limpo['matriz_rei'], 'matriz_rei')
        else:
            # Calcula REI se não existir mas tiver os componentes
            if all(col in df_limpo.columns for col in ['resultado_rei', 'execucao_rei', 'investimento_rei']):
                df_limpo['matriz_rei'] = (df_limpo['resultado_rei'] * 
                                         df_limpo['execucao_rei'] * 
                                         df_limpo['investimento_rei'])
        
        # Processa matriz GUT
        colunas_gut = ['gravidade_gut', 'urgencia_gut', 'tendencia_gut', 'matriz_gut']
        for coluna in colunas_gut:
            if coluna in df_limpo.columns:
                df_limpo[coluna] = pd.to_numeric(df_limpo[coluna], errors='coerce')
                if df_limpo[coluna].isna().sum() > len(df_limpo) * 0.5:
                    df_limpo[coluna] = limpar_coluna_numerica(df_limpo[coluna], coluna)
        
        # Processa percentual_ok
        if 'percentual_ok' in df_limpo.columns:
            df_limpo['percentual_ok'] = pd.to_numeric(df_limpo['percentual_ok'], errors='coerce')
        
        # Processa soma_gut_rei
        if 'soma_gut_rei' in df_limpo.columns:
            df_limpo['soma_gut_rei'] = pd.to_numeric(df_limpo['soma_gut_rei'], errors='coerce')
        
        # Converte colunas de data
        if 'PREV_CONCLUSAO' in df_limpo.columns:
            df_limpo['previsao_conclusao'] = pd.to_datetime(df_limpo['PREV_CONCLUSAO'], dayfirst=True, errors='coerce')
        
        if 'DATA_CONCLUSAO' in df_limpo.columns:
            df_limpo['data_conclusao'] = pd.to_datetime(df_limpo['DATA_CONCLUSAO'], dayfirst=True, errors='coerce')
        
        # Processa colunas de texto
        colunas_texto = ['OQUE', 'ONDE', 'PORQUE', 'COMO', 'QUEM']
        for coluna in colunas_texto:
            if coluna in df_limpo.columns:
                novo_nome = {
                    'OQUE': 'product_backlog',
                    'ONDE': 'grooming_detalhamento',
                    'PORQUE': 'origem_nao_conformidade',
                    'COMO': 'detalhamento_acoes',
                    'QUEM': 'piloto_resp'
                }.get(coluna, coluna.lower())
                
                df_limpo[novo_nome] = df_limpo[coluna].astype(str).str.strip()
                df_limpo[novo_nome] = df_limpo[novo_nome].replace(['nan', 'None', 'null', ''], None)
        
        # Adiciona colunas de classificação
        if 'matriz_gut' in df_limpo.columns:
            df_limpo['classificacao_gut'] = df_limpo['matriz_gut'].apply(classificar_gut)
        
        if 'matriz_rei' in df_limpo.columns:
            df_limpo['classificacao_rei'] = df_limpo['matriz_rei'].apply(classificar_rei)
        
        # Calcula status
        if 'data_conclusao' in df_limpo.columns:
            df_limpo['status'] = df_limpo['data_conclusao'].apply(
                lambda x: 'Concluída' if pd.notna(x) else 'Em andamento'
            )
        
        return df_limpo
        
    except Exception as e:
        st.error(f"Erro na limpeza dos dados: {str(e)}")
        st.error(traceback.format_exc())
        return df

# ==================== FUNÇÕES DE CARREGAMENTO ====================

def carregar_csv_upload(uploaded_file):
    """Carrega CSV a partir de upload do usuário"""
    if uploaded_file is not None:
        try:
            # Tenta diferentes separadores e encodings
            separadores = [';', ',', '\t']
            encodings = ['utf-8', 'latin1', 'cp1252']
            
            for sep in separadores:
                for encoding in encodings:
                    try:
                        # Volta para o início do arquivo
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep=sep, encoding=encoding)
                        if len(df.columns) > 1:  # Se conseguiu ler corretamente
                            return df, f"Sucesso: separador '{sep}', encoding {encoding}"
                    except:
                        continue
            
            return None, "Erro: não foi possível ler o arquivo com nenhum separador/encoding"
            
        except Exception as e:
            return None, f"Erro ao ler arquivo: {str(e)}"
    
    return None, "Nenhum arquivo selecionado"

# ==================== FUNÇÕES DE ANÁLISE PREDITIVA ====================

def analise_preditiva_entregas(df):
    """
    Realiza análise preditiva de entrega das tarefas usando médias móveis e tendências
    """
    predicoes = {
        'previsao_geral': {},
        'previsoes_por_responsavel': [],
        'tarefas_em_risco': [],
        'curva_entregas': []
    }
    
    if df is None or df.empty:
        return predicoes
    
    try:
        # Análise de tarefas em andamento
        tarefas_andamento = df[df['status'] == 'Em andamento'] if 'status' in df.columns else df
        
        # Calcula taxa de conclusão histórica
        if 'data_conclusao' in df.columns:
            tarefas_concluidas = df[df['data_conclusao'].notna()].copy()
            if not tarefas_concluidas.empty:
                # Agrupa conclusões por mês
                tarefas_concluidas['mes_conclusao'] = tarefas_concluidas['data_conclusao'].dt.to_period('M')
                conclusoes_por_mes = tarefas_concluidas.groupby('mes_conclusao').size().reset_index(name='quantidade')
                
                # Calcula média móvel simples para previsão
                if len(conclusoes_por_mes) >= 3:
                    ultimos_valores = conclusoes_por_mes['quantidade'].tail(3).values
                    taxa_media = np.mean(ultimos_valores)
                    
                    previsao_proximo_mes = max(0, int(taxa_media))
                    previsao_2_meses = max(0, int(taxa_media * 0.9))
                    previsao_3_meses = max(0, int(taxa_media * 0.8))
                    
                    predicoes['previsao_geral'] = {
                        'taxa_conclusao_media': taxa_media,
                        'previsao_proximo_mes': previsao_proximo_mes,
                        'previsao_2_meses': previsao_2_meses,
                        'previsao_3_meses': previsao_3_meses,
                        'periodos_analisados': len(conclusoes_por_mes),
                        'ultimos_valores': ultimos_valores.tolist()
                    }
                    
                    # Curva de entregas
                    acumulado_atual = len(tarefas_concluidas)
                    predicoes['curva_entregas'] = [
                        {'periodo': 'Atual', 'quantidade_prevista': acumulado_atual, 'acumulado': acumulado_atual},
                        {'periodo': '+1 mês', 'quantidade_prevista': previsao_proximo_mes, 'acumulado': acumulado_atual + previsao_proximo_mes},
                        {'periodo': '+2 meses', 'quantidade_prevista': previsao_2_meses, 'acumulado': acumulado_atual + previsao_proximo_mes + previsao_2_meses},
                        {'periodo': '+3 meses', 'quantidade_prevista': previsao_3_meses, 'acumulado': acumulado_atual + previsao_proximo_mes + previsao_2_meses + previsao_3_meses}
                    ]
                else:
                    media_total = len(tarefas_concluidas) / max(1, len(conclusoes_por_mes))
                    predicoes['previsao_geral'] = {
                        'taxa_conclusao_media': media_total,
                        'previsao_proximo_mes': max(0, int(media_total)),
                        'previsao_2_meses': max(0, int(media_total)),
                        'previsao_3_meses': max(0, int(media_total)),
                        'periodos_analisados': len(conclusoes_por_mes),
                        'ultimos_valores': []
                    }
        
        # Identifica tarefas em risco de atraso
        if 'previsao_conclusao' in tarefas_andamento.columns:
            hoje = datetime.now()
            for _, tarefa in tarefas_andamento.iterrows():
                if pd.notna(tarefa['previsao_conclusao']):
                    dias_para_previsao = (tarefa['previsao_conclusao'] - hoje).days
                    percentual = tarefa.get('percentual_ok', 0) or 0
                    
                    em_risco = False
                    motivo_risco = []
                    
                    if dias_para_previsao < 0:
                        em_risco = True
                        motivo_risco.append(f"Atrasada há {abs(dias_para_previsao)} dias")
                    elif dias_para_previsao < 15 and percentual < 50:
                        em_risco = True
                        motivo_risco.append(f"Prazo curto ({dias_para_previsao} dias) e progresso baixo ({percentual:.0f}%)")
                    elif dias_para_previsao < 30 and percentual < 30:
                        em_risco = True
                        motivo_risco.append(f"Progresso muito baixo para o prazo restante")
                    
                    if em_risco:
                        predicoes['tarefas_em_risco'].append({
                            'tarefa': tarefa.get('product_backlog', 'Sem descrição'),
                            'responsavel': tarefa.get('piloto_resp', 'Não definido'),
                            'previsao': tarefa['previsao_conclusao'].strftime('%d/%m/%Y'),
                            'percentual': percentual,
                            'dias_restantes': dias_para_previsao,
                            'motivo': ', '.join(motivo_risco)
                        })
        
        # Previsão por responsável
        if 'piloto_resp' in df.columns and 'percentual_ok' in df.columns:
            for responsavel in df['piloto_resp'].dropna().unique():
                tarefas_resp = df[df['piloto_resp'] == responsavel]
                tarefas_concluidas_resp = tarefas_resp[tarefas_resp['status'] == 'Concluída'] if 'status' in tarefas_resp.columns else tarefas_resp[tarefas_resp['data_conclusao'].notna()]
                tarefas_andamento_resp = tarefas_resp[tarefas_resp['status'] == 'Em andamento'] if 'status' in tarefas_resp.columns else tarefas_resp[tarefas_resp['data_conclusao'].isna()]
                
                if not tarefas_concluidas_resp.empty and 'data_conclusao' in tarefas_concluidas_resp.columns:
                    dias_conclusao = []
                    for _, t in tarefas_concluidas_resp.iterrows():
                        if pd.notna(t.get('previsao_conclusao')) and pd.notna(t['data_conclusao']):
                            dias = (t['data_conclusao'] - t['previsao_conclusao']).days
                            dias_conclusao.append(dias)
                    tempo_medio = np.mean(dias_conclusao) if dias_conclusao else 0
                else:
                    tempo_medio = 0
                
                predicoes['previsoes_por_responsavel'].append({
                    'responsavel': responsavel,
                    'total_tarefas': len(tarefas_resp),
                    'concluidas': len(tarefas_concluidas_resp),
                    'andamento': len(tarefas_andamento_resp),
                    'taxa_conclusao': len(tarefas_concluidas_resp) / len(tarefas_resp) * 100 if len(tarefas_resp) > 0 else 0,
                    'media_progresso': tarefas_andamento_resp['percentual_ok'].mean() if not tarefas_andamento_resp.empty else 0,
                    'tempo_medio_conclusao': tempo_medio
                })
            
            predicoes['previsoes_por_responsavel'] = sorted(predicoes['previsoes_por_responsavel'], key=lambda x: x['taxa_conclusao'])
        
    except Exception as e:
        st.sidebar.write(f"Erro na análise preditiva: {str(e)}")
    
    return predicoes

# ==================== FUNÇÕES DE ANÁLISE ====================

def gerar_indicadores_iso9001(df):
    """Gera indicadores para acompanhamento da ISO 9001"""
    
    indicadores = {
        'visao_geral': {},
        'por_classificacao_gut': [],
        'por_classificacao_rei': [],
        'por_status': [],
        'por_responsavel': [],
        'metricas_tempo': {},
        'estatisticas_rei': {},
        'estatisticas_gut': {},
        'sumarizado_por_tarefa': []
    }
    
    if df is None or df.empty:
        return indicadores
    
    # Visão geral
    total_tarefas = len(df)
    
    if 'data_conclusao' in df.columns:
        tarefas_concluidas = df['data_conclusao'].notna().sum()
    else:
        tarefas_concluidas = 0
    
    if 'percentual_ok' in df.columns:
        media_percentual = df['percentual_ok'].mean()
        if pd.isna(media_percentual):
            media_percentual = 0
    else:
        media_percentual = 0
    
    indicadores['visao_geral'] = {
        'total_tarefas': total_tarefas,
        'tarefas_concluidas': tarefas_concluidas,
        'tarefas_andamento': total_tarefas - tarefas_concluidas,
        'percentual_concluido': (tarefas_concluidas / total_tarefas * 100) if total_tarefas > 0 else 0,
        'media_percentual_ok': media_percentual
    }
    
    # SUMARIZAÇÃO POR TAREFA (OQUE)
    if 'product_backlog' in df.columns and 'percentual_ok' in df.columns:
        tarefas_agrupadas = df.groupby('product_backlog').agg({
            'percentual_ok': ['mean', 'count', lambda x: (x >= 100).sum()],
            'detalhamento_acoes': 'first',
            'piloto_resp': 'first',
            'matriz_gut': 'mean' if 'matriz_gut' in df.columns else None,
            'matriz_rei': 'mean' if 'matriz_rei' in df.columns else None,
            'previsao_conclusao': 'first' if 'previsao_conclusao' in df.columns else None,
            'status': 'first' if 'status' in df.columns else None
        }).reset_index()
        
        tarefas_agrupadas.columns = ['Tarefa', 'Percentual Médio', 'Quantidade Ocorrências', 'Concluídas', 
                                      'Ação Associada', 'Responsável', 'Média GUT', 'Média REI', 
                                      'Previsão', 'Status']
        
        tarefas_agrupadas['Percentual Conclusão'] = (tarefas_agrupadas['Concluídas'] / tarefas_agrupadas['Quantidade Ocorrências'] * 100).round(2)
        
        if 'Previsão' in tarefas_agrupadas.columns:
            tarefas_agrupadas['Previsão'] = tarefas_agrupadas['Previsão'].apply(
                lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else 'Não definida'
            )
        
        tarefas_agrupadas = tarefas_agrupadas.sort_values('Percentual Conclusão', ascending=False)
        
        tarefas_agrupadas['Status'] = tarefas_agrupadas['Percentual Conclusão'].apply(
            lambda x: '🟢 Concluído' if x >= 100 else '🟡 Em Andamento' if x > 0 else '🔴 Não Iniciado'
        )
        
        tarefas_agrupadas = tarefas_agrupadas[tarefas_agrupadas['Tarefa'].notna()]
        tarefas_agrupadas = tarefas_agrupadas[tarefas_agrupadas['Tarefa'] != 'nan']
        tarefas_agrupadas = tarefas_agrupadas[tarefas_agrupadas['Tarefa'] != 'None']
        
        indicadores['sumarizado_por_tarefa'] = tarefas_agrupadas.to_dict('records')
    
    # Estatísticas da matriz REI
    if 'matriz_rei' in df.columns:
        df_rei_valido = df['matriz_rei'].dropna()
        if not df_rei_valido.empty:
            indicadores['estatisticas_rei'] = {
                'media': float(df_rei_valido.mean()),
                'mediana': float(df_rei_valido.median()),
                'minimo': float(df_rei_valido.min()),
                'maximo': float(df_rei_valido.max()),
                'total_nao_nulos': len(df_rei_valido)
            }
    
    # Estatísticas dos componentes REI
    for comp in ['resultado_rei', 'execucao_rei', 'investimento_rei']:
        if comp in df.columns:
            df_comp_valido = df[comp].dropna()
            if not df_comp_valido.empty:
                indicadores['estatisticas_rei'][f'media_{comp}'] = float(df_comp_valido.mean())
    
    # Estatísticas da matriz GUT
    if 'matriz_gut' in df.columns:
        df_gut_valido = df['matriz_gut'].dropna()
        if not df_gut_valido.empty:
            indicadores['estatisticas_gut'] = {
                'media': float(df_gut_valido.mean()),
                'mediana': float(df_gut_valido.median()),
                'minimo': float(df_gut_valido.min()),
                'maximo': float(df_gut_valido.max())
            }
    
    # Por classificação GUT
    if 'classificacao_gut' in df.columns:
        gut_counts = df['classificacao_gut'].value_counts().reset_index()
        gut_counts.columns = ['classificacao', 'quantidade']
        gut_counts['percentual'] = (gut_counts['quantidade'] / total_tarefas * 100).round(2)
        indicadores['por_classificacao_gut'] = gut_counts.to_dict('records')
    
    # Por classificação REI
    if 'classificacao_rei' in df.columns:
        rei_counts = df['classificacao_rei'].value_counts().reset_index()
        rei_counts.columns = ['classificacao', 'quantidade']
        rei_counts['percentual'] = (rei_counts['quantidade'] / total_tarefas * 100).round(2)
        indicadores['por_classificacao_rei'] = rei_counts.to_dict('records')
    
    # Por status
    if 'status' in df.columns:
        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['status', 'quantidade']
        status_counts['percentual'] = (status_counts['quantidade'] / total_tarefas * 100).round(2)
        indicadores['por_status'] = status_counts.to_dict('records')
    
    # Por responsável
    if 'piloto_resp' in df.columns:
        resp_counts = df['piloto_resp'].value_counts().head(10).reset_index()
        resp_counts.columns = ['responsavel', 'quantidade_tarefas']
        indicadores['por_responsavel'] = resp_counts.to_dict('records')
    
    # Tarefas críticas (GUT >= 60)
    if 'matriz_gut' in df.columns:
        tarefas_criticas = df[df['matriz_gut'] >= 60].copy()
        if not tarefas_criticas.empty:
            cols_to_show = []
            if 'product_backlog' in tarefas_criticas.columns:
                cols_to_show.append('product_backlog')
            if 'piloto_resp' in tarefas_criticas.columns:
                cols_to_show.append('piloto_resp')
            if 'previsao_conclusao' in tarefas_criticas.columns:
                cols_to_show.append('previsao_conclusao')
            cols_to_show.extend(['matriz_gut', 'detalhamento_acoes'] if 'detalhamento_acoes' in tarefas_criticas.columns else ['matriz_gut'])
            
            indicadores['tarefas_criticas'] = tarefas_criticas[cols_to_show].head(10).to_dict('records')
    
    # Tarefas de alto impacto (REI >= 60)
    if 'matriz_rei' in df.columns:
        tarefas_alto_impacto = df[df['matriz_rei'] >= 60].copy()
        if not tarefas_alto_impacto.empty:
            cols_to_show = []
            if 'product_backlog' in tarefas_alto_impacto.columns:
                cols_to_show.append('product_backlog')
            if 'piloto_resp' in tarefas_alto_impacto.columns:
                cols_to_show.append('piloto_resp')
            if 'previsao_conclusao' in tarefas_alto_impacto.columns:
                cols_to_show.append('previsao_conclusao')
            cols_to_show.extend(['matriz_rei', 'detalhamento_acoes'] if 'detalhamento_acoes' in tarefas_alto_impacto.columns else ['matriz_rei'])
            
            indicadores['tarefas_alto_impacto'] = tarefas_alto_impacto[cols_to_show].head(10).to_dict('records')
    
    # Análise de prazos
    if 'previsao_conclusao' in df.columns and 'data_conclusao' in df.columns:
        df_temp = df.dropna(subset=['previsao_conclusao', 'data_conclusao']).copy()
        if not df_temp.empty:
            df_temp['dias_para_conclusao'] = (df_temp['data_conclusao'] - df_temp['previsao_conclusao']).dt.days
            indicadores['metricas_tempo'] = {
                'media_dias_atraso': float(df_temp['dias_para_conclusao'].mean()) if not df_temp['dias_para_conclusao'].empty else 0,
                'max_dias_atraso': int(df_temp['dias_para_conclusao'].max()) if not df_temp['dias_para_conclusao'].empty else 0,
                'tarefas_no_prazo': int((df_temp['dias_para_conclusao'] <= 0).sum()),
                'tarefas_atrasadas': int((df_temp['dias_para_conclusao'] > 0).sum())
            }
    
    return indicadores

def gerar_resumo_para_llm(indicadores, df, pergunta_usuario):
    """Gera resumo estruturado para a LLM com os indicadores"""
    
    # Prepara dados das tarefas de alto impacto
    tarefas_alto_impacto_texto = ""
    if 'tarefas_alto_impacto' in indicadores and indicadores['tarefas_alto_impacto']:
        tarefas_alto_impacto_texto = "\n## TAREFAS DE ALTO IMPACTO (REI >= 60)\n"
        for i, tarefa in enumerate(indicadores['tarefas_alto_impacto'], 1):
            desc = tarefa.get('product_backlog', 'Sem descrição')
            rei = tarefa.get('matriz_rei', 0)
            responsavel = tarefa.get('piloto_resp', 'Não definido')
            previsao = tarefa.get('previsao_conclusao', 'Não definida')
            if isinstance(previsao, pd.Timestamp):
                previsao = previsao.strftime('%d/%m/%Y')
            detalhe = tarefa.get('detalhamento_acoes', '')
            if detalhe and len(str(detalhe)) > 50:
                detalhe = str(detalhe)[:50] + "..."
            tarefas_alto_impacto_texto += f"{i}. {desc} - REI: {rei} - Responsável: {responsavel} - Previsão: {previsao} - Ação: {detalhe}\n"
    
    # Prepara dados das tarefas críticas
    tarefas_criticas_texto = ""
    if 'tarefas_criticas' in indicadores and indicadores['tarefas_criticas']:
        tarefas_criticas_texto = "\n## TAREFAS CRÍTICAS (GUT >= 60)\n"
        for i, tarefa in enumerate(indicadores['tarefas_criticas'], 1):
            desc = tarefa.get('product_backlog', 'Sem descrição')
            gut = tarefa.get('matriz_gut', 0)
            responsavel = tarefa.get('piloto_resp', 'Não definido')
            previsao = tarefa.get('previsao_conclusao', 'Não definida')
            if isinstance(previsao, pd.Timestamp):
                previsao = previsao.strftime('%d/%m/%Y')
            detalhe = tarefa.get('detalhamento_acoes', '')
            if detalhe and len(str(detalhe)) > 50:
                detalhe = str(detalhe)[:50] + "..."
            tarefas_criticas_texto += f"{i}. {desc} - GUT: {gut} - Responsável: {responsavel} - Previsão: {previsao} - Ação: {detalhe}\n"
    
    resumo = f"""# ANÁLISE ISO 9001 - QRZ CONSULTORIA

## 1. VISÃO GERAL DO PROJETO
- **Total de tarefas:** {indicadores['visao_geral'].get('total_tarefas', 0):,}
- **Tarefas concluídas:** {indicadores['visao_geral'].get('tarefas_concluidas', 0):,}
- **Tarefas em andamento:** {indicadores['visao_geral'].get('tarefas_andamento', 0):,}
- **Percentual concluído:** {indicadores['visao_geral'].get('percentual_concluido', 0):.1f}%
- **Média de progresso:** {indicadores['visao_geral'].get('media_percentual_ok', 0):.1f}%

"""

    # Estatísticas REI
    if indicadores.get('estatisticas_rei'):
        resumo += f"""
## 2. ESTATÍSTICAS DA MATRIZ REI (Resultado x Execução x Investimento)
- **Média REI:** {indicadores['estatisticas_rei'].get('media', 0):.2f}
- **Mediana REI:** {indicadores['estatisticas_rei'].get('mediana', 0):.2f}
- **Mínimo REI:** {indicadores['estatisticas_rei'].get('minimo', 0):.2f}
- **Máximo REI:** {indicadores['estatisticas_rei'].get('maximo', 0):.2f}
- **Total de tarefas com REI válido:** {indicadores['estatisticas_rei'].get('total_nao_nulos', 0)}
"""
        if 'media_resultado_rei' in indicadores['estatisticas_rei']:
            resumo += f"- **Média Resultado (R):** {indicadores['estatisticas_rei']['media_resultado_rei']:.2f}\n"
        if 'media_execucao_rei' in indicadores['estatisticas_rei']:
            resumo += f"- **Média Execução (E):** {indicadores['estatisticas_rei']['media_execucao_rei']:.2f}\n"
        if 'media_investimento_rei' in indicadores['estatisticas_rei']:
            resumo += f"- **Média Investimento (I):** {indicadores['estatisticas_rei']['media_investimento_rei']:.2f}\n"

    # Classificação REI
    if indicadores.get('por_classificacao_rei'):
        resumo += "\n## 3. CLASSIFICAÇÃO REI (Impacto/Esforço)\n"
        for item in indicadores['por_classificacao_rei']:
            resumo += f"- **{item['classificacao']}:** {item['quantidade']} tarefas ({item['percentual']:.1f}%)\n"
    
    # Estatísticas GUT
    if indicadores.get('estatisticas_gut'):
        resumo += f"""
## 4. ESTATÍSTICAS DA MATRIZ GUT (Gravidade x Urgência x Tendência)
- **Média GUT:** {indicadores['estatisticas_gut'].get('media', 0):.2f}
- **Mediana GUT:** {indicadores['estatisticas_gut'].get('mediana', 0):.2f}
- **Mínimo GUT:** {indicadores['estatisticas_gut'].get('minimo', 0):.2f}
- **Máximo GUT:** {indicadores['estatisticas_gut'].get('maximo', 0):.2f}
"""

    # Classificação GUT
    if indicadores.get('por_classificacao_gut'):
        resumo += "\n## 5. CLASSIFICAÇÃO GUT (Prioridade)\n"
        for item in indicadores['por_classificacao_gut']:
            resumo += f"- **{item['classificacao']}:** {item['quantidade']} tarefas ({item['percentual']:.1f}%)\n"
    
    # Status das tarefas
    if indicadores.get('por_status'):
        resumo += "\n## 6. STATUS DAS TAREFAS\n"
        for item in indicadores['por_status']:
            resumo += f"- **{item['status']}:** {item['quantidade']} tarefas ({item['percentual']:.1f}%)\n"
    
    # Tarefas críticas (GUT)
    if tarefas_criticas_texto:
        resumo += tarefas_criticas_texto
    
    # Tarefas alto impacto (REI)
    if tarefas_alto_impacto_texto:
        resumo += tarefas_alto_impacto_texto
    
    # Métricas de prazo
    if indicadores.get('metricas_tempo'):
        resumo += f"""
## 7. MÉTRICAS DE PRAZO
- **Média de dias de atraso:** {indicadores['metricas_tempo'].get('media_dias_atraso', 0):.1f} dias
- **Tarefas no prazo:** {indicadores['metricas_tempo'].get('tarefas_no_prazo', 0)}
- **Tarefas atrasadas:** {indicadores['metricas_tempo'].get('tarefas_atrasadas', 0)}
"""
    
    # Pergunta do usuário
    resumo += f"""
## 8. PERGUNTA DO USUÁRIO
{pergunta_usuario}

## 9. INSTRUÇÕES PARA RESPOSTA
Com base nos dados acima, responda à pergunta de forma clara e objetiva, evite a todo custo que haja "alucination":
- Use os números exatos fornecidos
- Destaque os principais gargalos e pontos de atenção
- Sugira priorizações baseadas na matriz GUT
- Considere o impacto/esforço pela matriz REI
- Se for solicitada uma lista, apresente os itens de forma organizada
- Se for solicitada a lista de tarefas, informe sempre o nome do responsável, previsão de conclusão
- Seja específico nas recomendações para a certificação ISO 9001
- Apresente no Chat somente a resposta à pergunta do usuário.
- Se perguntar sobre outro assunto senão o que está nas informações fornecidas, responda "Desculpe, somente estou autorizado a responder sobre os dados fornecidos" e informe o assunto tratado.
Seja sempre cordial!!
"""

    return resumo

# ==================== FUNÇÕES DE GRÁFICOS ====================

def criar_grafico_gauge(valor, titulo, cor="blue"):
    """Cria um gráfico de gauge (velocímetro) para percentuais"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor,
        title={'text': titulo},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': cor},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#ffcccc'},
                {'range': [33, 66], 'color': '#ffffcc'},
                {'range': [66, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': valor
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def criar_grafico_barras_horizontais(df_tarefas, titulo):
    """Cria gráfico de barras horizontais para tarefas"""
    if not df_tarefas.empty:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=df_tarefas['Tarefa'][:15],
            x=df_tarefas['Percentual Conclusão'][:15],
            orientation='h',
            marker=dict(
                color=df_tarefas['Percentual Conclusão'][:15],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Conclusão (%)")
            ),
            text=df_tarefas['Percentual Conclusão'][:15].apply(lambda x: f"{x:.1f}%"),
            textposition='outside'
        ))
        
        fig.update_layout(
            title=titulo,
            xaxis_title="Percentual de Conclusão (%)",
            yaxis_title="Tarefa",
            height=500,
            margin=dict(l=200, r=50, t=50, b=50)
        )
        
        return fig
    return None

def criar_grafico_treemap(df_tarefas, titulo):
    """Cria um treemap para visualizar tarefas e seus percentuais"""
    if not df_tarefas.empty:
        fig = go.Figure(go.Treemap(
            labels=df_tarefas['Tarefa'][:20],
            parents=[""] * len(df_tarefas['Tarefa'][:20]),
            values=df_tarefas['Quantidade Ocorrências'][:20],
            textinfo="label+value+percent root",
            texttemplate="%{label}<br>%{value} ocorrências<br>%{percentRoot:.1f}%",
            marker=dict(
                colors=df_tarefas['Percentual Conclusão'][:20],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Conclusão (%)")
            )
        ))
        
        fig.update_layout(
            title=titulo,
            height=600
        )
        
        return fig
    return None

def criar_grafico_curva_entregas(predicoes):
    """Cria gráfico de curva de entregas previstas"""
    if not predicoes.get('curva_entregas'):
        return None
    
    df_curva = pd.DataFrame(predicoes['curva_entregas'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_curva['periodo'],
        y=df_curva['quantidade_prevista'],
        mode='lines+markers',
        name='Entregas por período',
        line=dict(color='blue', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_curva['periodo'],
        y=df_curva['acumulado'],
        mode='lines+markers',
        name='Acumulado',
        line=dict(color='green', width=3, dash='dash'),
        marker=dict(size=10, symbol='square')
    ))
    
    fig.update_layout(
        title="📈 Previsão de Entregas",
        xaxis_title="Período",
        yaxis_title="Quantidade de Tarefas",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def criar_grafico_tarefas_risco(tarefas_risco):
    """Cria gráfico de tarefas em risco"""
    if not tarefas_risco:
        return None
    
    df_risco = pd.DataFrame(tarefas_risco)
    
    fig = go.Figure(go.Bar(
        x=df_risco['tarefa'][:10],
        y=df_risco['dias_restantes'],
        marker=dict(
            color=df_risco['percentual'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Progresso (%)")
        ),
        text=df_risco['percentual'].apply(lambda x: f"{x:.0f}%"),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="🚨 Tarefas em Risco de Atraso",
        xaxis_title="Tarefa",
        yaxis_title="Dias Restantes",
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig

def criar_grafico_radar(indicadores):
    """Cria gráfico de radar para análise multidimensional"""
    categorias = ['Progresso Geral', 'Eficiência GUT', 'Impacto REI', 'Conclusão por Status', 'Produtividade']
    
    valores = [
        indicadores['visao_geral'].get('percentual_concluido', 0),
        indicadores['estatisticas_gut'].get('media', 0) if indicadores.get('estatisticas_gut') else 0,
        indicadores['estatisticas_rei'].get('media', 0) if indicadores.get('estatisticas_rei') else 0,
        indicadores['visao_geral'].get('media_percentual_ok', 0),
        (indicadores['visao_geral'].get('tarefas_concluidas', 0) / max(1, indicadores['visao_geral'].get('total_tarefas', 1))) * 100
    ]
    
    valores[1] = min(100, (valores[1] / 125) * 100)
    valores[2] = min(100, (valores[2] / 125) * 100)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=valores,
        theta=categorias,
        fill='toself',
        name='Performance',
        line_color='blue',
        fillcolor='rgba(0, 0, 255, 0.2)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[100, 100, 100, 100, 100],
        theta=categorias,
        fill='toself',
        name='Meta',
        line_color='green',
        fillcolor='rgba(0, 255, 0, 0.1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title="📊 Dashboard de Performance GRC",
        height=500,
        showlegend=True
    )
    
    return fig

# ==================== INTERFACE STREAMLIT ====================

# Inicializa estados da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df_tarefas" not in st.session_state:
    st.session_state.df_tarefas = None
if "indicadores" not in st.session_state:
    st.session_state.indicadores = None
if "analise_preditiva" not in st.session_state:
    st.session_state.analise_preditiva = None

# Barra lateral
with st.sidebar:
    st.title("📋 QRZ Consultoria - Reestruturação Industrial")
    st.markdown("### Gestão de Tarefas")
    
    # Configuração da API Key (usa st.secrets para produção)
    groq_api_key = st.text_input(
        "🔑 Sua API Key Groq",
        type="password",
        value=st.secrets.get("GROQ_API_KEY", ""),
        help="Chave da Groq para consultas. Você pode configurar no .streamlit/secrets.toml"
    )
    
    st.divider()
    
    st.subheader("📁 Upload de Dados")
    
    # Upload de arquivo CSV
    uploaded_file = st.file_uploader(
        "Carregue o arquivo tarefas.csv",
        type=['csv'],
        help="Selecione o arquivo CSV com os dados das tarefas"
    )
    
    if uploaded_file is not None:
        if st.button("🔄 Processar Arquivo", type="primary", use_container_width=True):
            with st.spinner("Processando arquivo..."):
                df, msg = carregar_csv_upload(uploaded_file)
                
                if df is not None:
                    st.session_state.df_tarefas = limpar_dataframe_tarefas(df)
                    st.session_state.indicadores = gerar_indicadores_iso9001(st.session_state.df_tarefas)
                    st.session_state.analise_preditiva = analise_preditiva_entregas(st.session_state.df_tarefas)
                    st.success(f"✅ Arquivo processado com sucesso! {msg}")
                else:
                    st.error(f"❌ {msg}")
    
    st.divider()
    
    if st.button("🧹 Limpar Sessão", use_container_width=True):
        for key in ['df_tarefas', 'indicadores', 'analise_preditiva', 'messages']:
            if key in st.session_state:
                if key == 'messages':
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
        st.rerun()
    
    st.caption("Versão 5.0 - Cloud Ready")

# Área principal
st.title("📋 Reestruturação Industrial QRZ Consultoria")

# Interface principal com dados carregados
if st.session_state.df_tarefas is not None and st.session_state.indicadores is not None:
    df = st.session_state.df_tarefas
    indicadores = st.session_state.indicadores
    analise_preditiva = st.session_state.analise_preditiva
    
    # ==================== TABELA PRINCIPAL DE TAREFAS ====================
    st.subheader("📊 Sumarização por Tarefa - Percentuais de Realização")
    
    if indicadores.get('sumarizado_por_tarefa'):
        df_tarefas = pd.DataFrame(indicadores['sumarizado_por_tarefa'])
        
        colunas_exibicao = ['Tarefa', 'Percentual Médio', 'Quantidade Ocorrências', 'Percentual Conclusão', 
                           'Ação Associada', 'Responsável', 'Média GUT', 'Média REI', 'Previsão', 'Status']
        
        colunas_existentes = [col for col in colunas_exibicao if col in df_tarefas.columns]
        df_tarefas_exibicao = df_tarefas[colunas_existentes].copy()
        
        if 'Percentual Médio' in df_tarefas_exibicao.columns:
            df_tarefas_exibicao['Percentual Médio'] = df_tarefas_exibicao['Percentual Médio'].apply(lambda x: f"{x:.1f}%")
        if 'Percentual Conclusão' in df_tarefas_exibicao.columns:
            df_tarefas_exibicao['Percentual Conclusão'] = df_tarefas_exibicao['Percentual Conclusão'].apply(lambda x: f"{x:.1f}%")
        if 'Média GUT' in df_tarefas_exibicao.columns:
            df_tarefas_exibicao['Média GUT'] = df_tarefas_exibicao['Média GUT'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        if 'Média REI' in df_tarefas_exibicao.columns:
            df_tarefas_exibicao['Média REI'] = df_tarefas_exibicao['Média REI'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        
        st.dataframe(df_tarefas_exibicao, use_container_width=True)
        
        csv_tarefas = df_tarefas_exibicao.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download da Tabela de Tarefas",
            csv_tarefas,
            "tarefas_sumarizadas.csv",
            "text/csv",
            key='download-tarefas'
        )
    else:
        st.info("Nenhuma tarefa encontrada para sumarização")
    
    st.divider()
    
    # ==================== MÉTRICAS PRINCIPAIS COM GAUGES ====================
    st.subheader("📈 Métricas de Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fig_gauge = criar_grafico_gauge(
            indicadores['visao_geral'].get('percentual_concluido', 0),
            "Conclusão Geral",
            "green"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        fig_gauge2 = criar_grafico_gauge(
            indicadores['visao_geral'].get('media_percentual_ok', 0),
            "Progresso Médio",
            "blue"
        )
        st.plotly_chart(fig_gauge2, use_container_width=True)
    
    with col3:
        fig_gauge3 = criar_grafico_gauge(
            indicadores.get('estatisticas_gut', {}).get('media', 0),
            "Média GUT",
            "orange"
        )
        st.plotly_chart(fig_gauge3, use_container_width=True)
    
    with col4:
        fig_gauge4 = criar_grafico_gauge(
            indicadores.get('estatisticas_rei', {}).get('media', 0),
            "Média REI",
            "purple"
        )
        st.plotly_chart(fig_gauge4, use_container_width=True)
    
    # ==================== GRÁFICOS MODERNOS ====================
    st.subheader("📊 Visualizações Avançadas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if indicadores.get('sumarizado_por_tarefa'):
            df_tarefas_temp = pd.DataFrame(indicadores['sumarizado_por_tarefa'])
            fig_barras = criar_grafico_barras_horizontais(df_tarefas_temp, "📊 Top Tarefas por Percentual de Conclusão")
            if fig_barras:
                st.plotly_chart(fig_barras, use_container_width=True)
    
    with col2:
        if indicadores.get('sumarizado_por_tarefa'):
            df_tarefas_temp = pd.DataFrame(indicadores['sumarizado_por_tarefa'])
            fig_treemap = criar_grafico_treemap(df_tarefas_temp, "🌳 Distribuição de Ocorrências por Tarefa")
            if fig_treemap:
                st.plotly_chart(fig_treemap, use_container_width=True)
    
    # ==================== ANÁLISE PREDITIVA ====================
    st.subheader("🔮 Análise Preditiva de Entregas")
    
    if analise_preditiva:
        col1, col2 = st.columns(2)
        
        with col1:
            if analise_preditiva.get('previsao_geral'):
                previsao = analise_preditiva['previsao_geral']
                st.metric(
                    "📊 Previsão de Entregas para Próximo Mês",
                    f"{previsao.get('previsao_proximo_mes', 0)} tarefas",
                    delta=f"{previsao.get('taxa_conclusao_media', 0):.1f} por mês",
                    delta_color="normal"
                )
                
                st.metric(
                    "📈 Períodos Analisados",
                    f"{previsao.get('periodos_analisados', 0)} meses",
                    help="Quantidade de meses considerados na análise"
                )
            
            if analise_preditiva.get('tarefas_em_risco'):
                st.warning(f"⚠️ {len(analise_preditiva['tarefas_em_risco'])} tarefas em risco de atraso")
        
        with col2:
            fig_curva = criar_grafico_curva_entregas(analise_preditiva)
            if fig_curva:
                st.plotly_chart(fig_curva, use_container_width=True)
        
        if analise_preditiva.get('tarefas_em_risco'):
            st.subheader("🚨 Tarefas em Risco de Atraso")
            df_risco = pd.DataFrame(analise_preditiva['tarefas_em_risco'])
            st.dataframe(df_risco, use_container_width=True)
        
        fig_risco = criar_grafico_tarefas_risco(analise_preditiva.get('tarefas_em_risco', []))
        if fig_risco:
            st.plotly_chart(fig_risco, use_container_width=True)
        
        if analise_preditiva.get('previsoes_por_responsavel'):
            st.subheader("👥 Produtividade por Responsável")
            df_resp = pd.DataFrame(analise_preditiva['previsoes_por_responsavel'])
            fig_resp = px.bar(
                df_resp,
                x='responsavel',
                y='taxa_conclusao',
                title='Taxa de Conclusão por Responsável',
                color='taxa_conclusao',
                color_continuous_scale='RdYlGn',
                text='taxa_conclusao'
            )
            fig_resp.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_resp.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_resp, use_container_width=True)
    else:
        st.info("Dados insuficientes para análise preditiva. Carregue mais dados ou aguarde mais registros.")
    
    # ==================== GRÁFICO RADAR ====================
    st.subheader("📡 Dashboard de Performance GRC")
    fig_radar = criar_grafico_radar(indicadores)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # ==================== ABAS ADICIONAIS ====================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Todas as Tarefas", 
        "🎯 Priorização GUT", 
        "💰 Análise REI", 
        "📈 Estatísticas",
        "🚨 Tarefas Críticas",
        "💬 Chat Análise"
    ])
    
    with tab1:
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"Mostrando 100 de {len(df)} registros")
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download dos dados",
            csv,
            "tarefas_iso9001.csv",
            "text/csv",
            key='download-csv'
        )
    
    with tab2:
        st.subheader("Classificação GUT - Prioridade das Tarefas")
        
        if indicadores.get('por_classificacao_gut'):
            df_gut = pd.DataFrame(indicadores['por_classificacao_gut'])
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(df_gut, use_container_width=True)
            
            with col2:
                fig_gut = px.pie(
                    df_gut, 
                    values='quantidade', 
                    names='classificacao',
                    title='Distribuição por Prioridade GUT',
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig_gut, use_container_width=True)
        else:
            st.info("Nenhuma classificação GUT disponível")
    
    with tab3:
        st.subheader("Classificação REI - Impacto/Esforço das Tarefas")
        
        if indicadores.get('por_classificacao_rei'):
            df_rei = pd.DataFrame(indicadores['por_classificacao_rei'])
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(df_rei, use_container_width=True)
            
            with col2:
                fig_rei = px.pie(
                    df_rei, 
                    values='quantidade', 
                    names='classificacao',
                    title='Distribuição por Impacto REI',
                    color_discrete_sequence=px.colors.sequential.Greens
                )
                st.plotly_chart(fig_rei, use_container_width=True)
        
        if indicadores.get('estatisticas_rei') and any(k.startswith('media_') for k in indicadores['estatisticas_rei'].keys()):
            st.subheader("📊 Detalhamento dos Componentes REI")
            col1, col2, col3 = st.columns(3)
            
            if 'media_resultado_rei' in indicadores['estatisticas_rei']:
                with col1:
                    st.metric("Média Resultado (R)", 
                             f"{indicadores['estatisticas_rei']['media_resultado_rei']:.1f}")
            if 'media_execucao_rei' in indicadores['estatisticas_rei']:
                with col2:
                    st.metric("Média Execução (E)", 
                             f"{indicadores['estatisticas_rei']['media_execucao_rei']:.1f}")
            if 'media_investimento_rei' in indicadores['estatisticas_rei']:
                with col3:
                    st.metric("Média Investimento (I)", 
                             f"{indicadores['estatisticas_rei']['media_investimento_rei']:.1f}")
    
    with tab4:
        st.subheader("Estatísticas Detalhadas")
        
        if indicadores.get('por_status'):
            st.write("**Status das Tarefas**")
            df_status = pd.DataFrame(indicadores['por_status'])
            st.dataframe(df_status, use_container_width=True)
            
            fig_status = px.bar(
                df_status,
                x='status',
                y='quantidade',
                title='Tarefas por Status',
                color='status',
                text='quantidade'
            )
            st.plotly_chart(fig_status, use_container_width=True)
        
        if indicadores.get('por_responsavel'):
            st.write("**Top Responsáveis**")
            df_resp = pd.DataFrame(indicadores['por_responsavel'])
            st.dataframe(df_resp, use_container_width=True)
            
            fig_resp = px.bar(
                df_resp,
                x='responsavel',
                y='quantidade_tarefas',
                title='Tarefas por Responsável',
                color='quantidade_tarefas',
                text='quantidade_tarefas'
            )
            fig_resp.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_resp, use_container_width=True)
        
        if indicadores.get('metricas_tempo'):
            st.write("**Análise de Prazos**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Média de Atraso", f"{indicadores['metricas_tempo'].get('media_dias_atraso', 0):.1f} dias")
            with col2:
                st.metric("Tarefas no Prazo", indicadores['metricas_tempo'].get('tarefas_no_prazo', 0))
            with col3:
                st.metric("Tarefas Atrasadas", indicadores['metricas_tempo'].get('tarefas_atrasadas', 0))
    
    with tab5:
        st.subheader("🚨 Tarefas Críticas e de Alto Impacto")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Tarefas Críticas (GUT >= 60)**")
            if indicadores.get('tarefas_criticas'):
                df_criticas = pd.DataFrame(indicadores['tarefas_criticas'])
                if 'previsao_conclusao' in df_criticas.columns:
                    df_criticas['previsao_conclusao'] = df_criticas['previsao_conclusao'].apply(
                        lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else 'Não definida'
                    )
                st.dataframe(df_criticas, use_container_width=True)
            else:
                st.info("Nenhuma tarefa crítica encontrada")
        
        with col2:
            st.write("**Tarefas Alto Impacto (REI >= 60)**")
            if indicadores.get('tarefas_alto_impacto'):
                df_alto_impacto = pd.DataFrame(indicadores['tarefas_alto_impacto'])
                if 'previsao_conclusao' in df_alto_impacto.columns:
                    df_alto_impacto['previsao_conclusao'] = df_alto_impacto['previsao_conclusao'].apply(
                        lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else 'Não definida'
                    )
                st.dataframe(df_alto_impacto, use_container_width=True)
            else:
                st.info("Nenhuma tarefa de alto impacto encontrada")
    
    with tab6:
        st.subheader("💬 Chat de Análise")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Faça sua pergunta sobre as tarefas ISO 9001..."):
            if not groq_api_key:
                st.warning("⚠️ Insira sua API Key na barra lateral")
                st.stop()
            
            try:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Analisando dados..."):
                        resumo_llm = gerar_resumo_para_llm(
                            indicadores, 
                            df, 
                            prompt
                        )
                        
                        client = Groq(api_key=groq_api_key)
                        
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": """Você é a QRZ Consultoria, especialista em gestão da qualidade e implementação ISO 9001.
                                Responda com base APENAS nos dados fornecidos sobre as tarefas.
                                Use a matriz GUT para priorização e a matriz REI para análise de esforço.
                                Destaque gargalos, pontos críticos e sugira ações concretas.
                                Seja claro, objetivo e foque na certificação ISO 9001.
                                Apresente a resposta em formato de texto simples, sem formatação HTML."""},
                                {"role": "user", "content": resumo_llm}
                            ],
                            model="openai/gpt-oss-20b",
                            temperature=0.3,
                            max_tokens=4096,
                        )
                        
                        if chat_completion and hasattr(chat_completion, 'choices') and len(chat_completion.choices) > 0:
                            resposta = chat_completion.choices[0].message.content
                            st.markdown(resposta)
                            st.session_state.messages.append({"role": "assistant", "content": resposta})
                            
                            txt_data, txt_filename = exportar_resposta_txt(resposta)
                            st.download_button(
                                "📥 Exportar resposta como TXT",
                                txt_data,
                                txt_filename,
                                "text/plain",
                                key=f"txt_{len(st.session_state.messages)}"
                            )
                        else:
                            st.error("❌ A API retornou uma resposta vazia. Tente novamente.")
                            
            except Exception as e:
                st.error(f"❌ Erro inesperado: {str(e)}")
                st.error(traceback.format_exc())

else:
    st.info("👋 Faça upload do arquivo CSV na barra lateral para iniciar a análise!")
    
    with st.expander("📋 Formato esperado do arquivo tarefas.csv"):
        st.markdown("""
        **tarefas.csv - Campos esperados:**
        
        - `OQUE` - Product Backlog (lista de produtos)
        - `ONDE` - Grooming / Detalhamento do Backlog
        - `PORQUE` - Origem da não conformidade
        - `R` - Resultado (matriz REI)
        - `E` - Execução (matriz REI)
        - `I` - Investimento (matriz REI)
        - `REI` - Resultado da Matriz REI (R*E*I)
        - `G` - Gravidade (matriz GUT)
        - `U` - Urgência (matriz GUT)
        - `T` - Tendência (matriz GUT)
        - `GUT` - Resultado da Matriz GUT (G*U*T)
        - `GUTREI` - Somatório de GUT e REI
        - `COMO` - Detalhamento das ações
        - `PERC_OK` - % de conclusão
        - `QUEM` - Responsável
        - `PREV_CONCLUSAO` - Previsão de conclusão
        - `DATA_CONCLUSAO` - Data de conclusão
        
        **Matriz GUT:**
        - Priorização: GUT >= 60 = Crítica, 40-59 = Alta, 20-39 = Média, 1-19 = Baixa
        
        **Matriz REI:**
        - Impacto: REI >= 60 = Alto impacto, 30-59 = Médio impacto, 1-29 = Baixo impacto
        """)

# Rodapé
st.markdown(
    """
    <div style="text-align: center; color: gray; margin-top: 50px;">
        <hr>
        <p>QRZ Consultoria - Reestruturação Industrial v5.0 - Cloud Ready</p>
    </div>
    """,
    unsafe_allow_html=True
)
