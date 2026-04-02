# FD Consultoria de Dados - Assistente Adaptativo para Análise de Dados
# Versão 7.1 - Com Suporte a PDF e Tratamento Robusto de Tabelas

import os
import streamlit as st
import pandas as pd
from groq import Groq
import io
from datetime import datetime
import re
import json
import numpy as np
import traceback
import pdfplumber
from typing import Tuple, Optional, Dict, Any

# Configura a página do Streamlit
st.set_page_config(
    page_title="FD Consultoria de Dados",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== FUNÇÕES DE UTILITÁRIOS ====================

def formatar_data_segura(data, formato='%d/%m/%Y'):
    """
    Formata uma data com segurança, retornando 'Data não disponível' se for NaT
    """
    if pd.isna(data) or data is None:
        return "Data não disponível"
    try:
        return data.strftime(formato)
    except:
        return "Data inválida"

def formatar_numero_seguro(valor, formato="R$ {:,.2f}"):
    """
    Formata um número com segurança
    """
    if pd.isna(valor) or valor is None:
        return "Valor não disponível"
    try:
        if formato.startswith("R$"):
            return f"R$ {valor:,.2f}"
        else:
            return formato.format(valor)
    except:
        return "Valor inválido"

def converter_para_texto_seguro(df, max_linhas=20):
    """
    Converte DataFrame para string de forma segura
    """
    try:
        # Converte para string, tratando valores especiais
        df_str = df.head(max_linhas).copy()
        
        # Converte todas as colunas para string de forma segura
        for col in df_str.columns:
            try:
                # Trata valores nulos e especiais
                df_str[col] = df_str[col].apply(lambda x: str(x) if pd.notna(x) else "NULO")
            except:
                df_str[col] = "ERRO_CONVERSAO"
        
        return df_str.to_string()
    except Exception as e:
        return f"Erro ao converter dados: {str(e)}"

# ==================== FUNÇÕES DE PROCESSAMENTO DE PDF ====================

def extrair_texto_pdf(arquivo_pdf) -> Tuple[Optional[str], Optional[str]]:
    """
    Extrai texto de um arquivo PDF com tratamento de erro
    
    Returns:
        Tuple[texto_extraido, mensagem_erro]
    """
    try:
        texto_completo = ""
        
        # Abre o PDF com pdfplumber
        with pdfplumber.open(arquivo_pdf) as pdf:
            for pagina_num, pagina in enumerate(pdf.pages, 1):
                # Extrai texto da página
                texto_pagina = pagina.extract_text()
                if texto_pagina:
                    texto_completo += f"\n--- Página {pagina_num} ---\n"
                    texto_completo += texto_pagina
                
                # Tenta extrair tabelas se existirem
                tabelas = pagina.extract_tables()
                if tabelas:
                    texto_completo += f"\n--- Tabelas da Página {pagina_num} ---\n"
                    for tabela_num, tabela in enumerate(tabelas, 1):
                        if tabela:
                            texto_completo += f"\nTabela {tabela_num}:\n"
                            for linha in tabela[:10]:  # Limita a 10 linhas
                                texto_completo += " | ".join([str(cell) if cell else "" for cell in linha]) + "\n"
        
        if not texto_completo.strip():
            return None, "Nenhum texto foi encontrado no PDF. O arquivo pode estar vazio ou ser composto apenas de imagens."
        
        return texto_completo, None
        
    except Exception as e:
        return None, f"Erro ao processar PDF: {str(e)}"

def extrair_tabelas_pdf(arquivo_pdf) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Tenta extrair tabelas do PDF e converter para DataFrame
    Com tratamento robusto para índices duplicados
    """
    try:
        todas_tabelas = []
        
        with pdfplumber.open(arquivo_pdf) as pdf:
            for pagina_num, pagina in enumerate(pdf.pages, 1):
                # Tenta extrair tabelas da página
                tabelas = pagina.extract_tables()
                for tabela_num, tabela in enumerate(tabelas, 1):
                    if tabela and len(tabela) > 1:
                        try:
                            # Verifica se a tabela tem cabeçalho
                            cabecalho = tabela[0] if tabela[0] else None
                            
                            # Se tem cabeçalho, usa como colunas
                            if cabecalho and any(cabecalho):
                                # Remove cabeçalho vazio ou nulo
                                cabecalho_limpo = []
                                for col in cabecalho:
                                    if col and str(col).strip():
                                        cabecalho_limpo.append(str(col).strip())
                                    else:
                                        cabecalho_limpo.append(f"Coluna_{len(cabecalho_limpo) + 1}")
                                
                                # Cria DataFrame com os dados (ignorando cabeçalho)
                                dados = tabela[1:]
                                df_tabela = pd.DataFrame(dados)
                                
                                # Renomeia colunas
                                df_tabela.columns = cabecalho_limpo[:len(df_tabela.columns)]
                                
                            else:
                                # Sem cabeçalho, cria colunas genéricas
                                df_tabela = pd.DataFrame(tabela)
                                df_tabela.columns = [f"Coluna_{i+1}" for i in range(len(df_tabela.columns))]
                            
                            # Limpa valores None/None
                            df_tabela = df_tabela.replace([None, ''], pd.NA)
                            
                            # Remove linhas totalmente vazias
                            df_tabela = df_tabela.dropna(how='all')
                            
                            # Se não estiver vazia, adiciona à lista
                            if not df_tabela.empty:
                                # Adiciona informações de origem
                                df_tabela['_origem_pagina'] = pagina_num
                                df_tabela['_origem_tabela'] = tabela_num
                                todas_tabelas.append(df_tabela)
                                
                        except Exception as e:
                            st.warning(f"Erro ao processar tabela {tabela_num} da página {pagina_num}: {str(e)}")
                            continue
        
        if todas_tabelas:
            # Combina todas as tabelas encontradas com reset de índice
            try:
                # Tenta concatenar todas as tabelas
                df_combinado = pd.concat(todas_tabelas, ignore_index=True, sort=False)
                return df_combinado, None
            except Exception as e:
                # Se falhar ao concatenar, retorna a primeira tabela
                st.warning(f"Erro ao combinar tabelas: {str(e)}. Retornando apenas a primeira tabela.")
                return todas_tabelas[0], None
        else:
            return None, "Nenhuma tabela válida foi encontrada no PDF."
            
    except Exception as e:
        return None, f"Erro ao extrair tabelas do PDF: {str(e)}"

def validar_e_limpar_dados_pdf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida e limpa dados extraídos de PDF
    """
    if df is None or df.empty:
        return df
    
    try:
        # Cria uma cópia para não modificar o original
        df_limpo = df.copy()
        
        # Remove colunas completamente vazias
        df_limpo = df_limpo.dropna(axis=1, how='all')
        
        # Remove linhas duplicadas
        df_limpo = df_limpo.drop_duplicates()
        
        # Limpa nomes das colunas
        df_limpo.columns = [str(col).strip().replace('\n', ' ').replace('\r', '') 
                           for col in df_limpo.columns]
        
        # Remove colunas de origem se existirem (opcional)
        # if '_origem_pagina' in df_limpo.columns and '_origem_tabela' in df_limpo.columns:
        #     df_limpo = df_limpo.drop(['_origem_pagina', '_origem_tabela'], axis=1)
        
        # Tenta converter colunas para tipos apropriados
        for col in df_limpo.columns:
            try:
                # Tenta converter para numérico
                df_limpo[col] = pd.to_numeric(df_limpo[col], errors='ignore')
            except:
                pass
        
        return df_limpo
        
    except Exception as e:
        st.warning(f"Erro na limpeza dos dados do PDF: {str(e)}")
        return df

def processar_pdf_para_analise(arquivo_pdf) -> Dict[str, Any]:
    """
    Processa PDF e retorna dados para análise
    Com tratamento robusto para erros
    """
    resultado = {
        "texto": None,
        "tabelas": None,
        "metadados": {
            "nome_arquivo": arquivo_pdf.name,
            "tamanho": arquivo_pdf.size,
            "tempo_processamento": datetime.now()
        },
        "mensagem_erro": None,
        "tabelas_encontradas": 0,
        "paginas_processadas": 0
    }
    
    try:
        # Extrai texto primeiro
        texto, erro_texto = extrair_texto_pdf(arquivo_pdf)
        if texto:
            resultado["texto"] = texto
            resultado["paginas_processadas"] = texto.count("--- Página")
        elif erro_texto:
            resultado["mensagem_erro"] = erro_texto
        
        # Tenta extrair tabelas com tratamento de erro
        try:
            tabelas, erro_tabelas = extrair_tabelas_pdf(arquivo_pdf)
            if tabelas is not None:
                resultado["tabelas"] = tabelas
                resultado["tabelas_encontradas"] = 1
            elif erro_tabelas:
                # Não considera erro de tabelas como crítico
                if not resultado["mensagem_erro"]:
                    resultado["mensagem_erro"] = erro_tabelas
        except Exception as e:
            if not resultado["mensagem_erro"]:
                resultado["mensagem_erro"] = f"Aviso: {str(e)}"
        
        # Analisa o texto extraído
        if resultado["texto"]:
            try:
                resultado["analise"] = analisar_texto_pdf(resultado["texto"])
            except Exception as e:
                resultado["analise"] = {"mensagem": f"Erro na análise: {str(e)}"}
        else:
            resultado["analise"] = {"mensagem": "PDF sem texto extraível"}
        
        return resultado
        
    except Exception as e:
        return {
            **resultado,
            "mensagem_erro": f"Erro ao processar PDF: {str(e)}"
        }

# ==================== FUNÇÕES DE LIMPEZA DE DADOS ====================

def limpar_valor_numerico(valor):
    """
    Limpa e converte um valor para numérico, independente do formato
    """
    if pd.isna(valor) or valor is None or valor == '':
        return None
    
    try:
        # Se já for numérico, retorna direto
        if isinstance(valor, (int, float)):
            return float(valor)
        
        # Converte para string e remove espaços
        str_valor = str(valor).strip()
        
        # Se estiver vazio após strip
        if not str_valor:
            return None
        
        # Remove símbolos de moeda e outros caracteres não numéricos
        str_valor = re.sub(r'[^\d,\-\.]', '', str_valor)
        
        # Substitui vírgula por ponto
        str_valor = str_valor.replace(',', '.')
        
        # Remove pontos extras (separadores de milhar)
        partes = str_valor.split('.')
        if len(partes) > 2:
            str_valor = ''.join(partes[:-1]) + '.' + partes[-1]
        
        # Converte para float
        if str_valor and str_valor != '.':
            return float(str_valor)
        else:
            return None
            
    except Exception as e:
        return None

def limpar_coluna_numerica(serie):
    """
    Aplica limpeza em uma série/coluna inteira
    """
    return serie.apply(limpar_valor_numerico)

def identificar_colunas_por_tipo(df):
    """
    Identifica automaticamente o tipo de cada coluna
    """
    colunas_numericas = []
    colunas_datas = []
    colunas_texto = []
    colunas_id = []
    
    for coluna in df.columns:
        try:
            # Amostra para análise
            amostra = df[coluna].dropna().head(5)
            
            if amostra.empty:
                colunas_texto.append(coluna)
                continue
            
            # Verifica se é data
            if df[coluna].dtype == 'datetime64[ns]':
                colunas_datas.append(coluna)
                continue
            
            # Tenta identificar por nome da coluna
            nome_lower = str(coluna).lower()
            
            if any(palavra in nome_lower for palavra in ['data', 'date', 'dia']):
                colunas_datas.append(coluna)
            elif any(palavra in nome_lower for palavra in ['valor', 'preco', 'preço', 'price', 'amount', 'total', 'quant', 'qtde', 'qtd']):
                colunas_numericas.append(coluna)
            elif any(palavra in nome_lower for palavra in ['id', 'codigo', 'código', 'code', 'sku']):
                colunas_id.append(coluna)
            else:
                # Verifica pelo tipo de dado
                if df[coluna].dtype in ['int64', 'float64']:
                    colunas_numericas.append(coluna)
                elif df[coluna].dtype == 'object':
                    # Tenta converter para número
                    try:
                        teste_conv = limpar_coluna_numerica(amostra)
                        if teste_conv.notna().any():
                            colunas_numericas.append(coluna)
                        else:
                            colunas_texto.append(coluna)
                    except:
                        colunas_texto.append(coluna)
                else:
                    colunas_texto.append(coluna)
        except Exception as e:
            # Em caso de erro, classifica como texto
            colunas_texto.append(coluna)
    
    return {
        'numericas': list(set(colunas_numericas)),
        'datas': list(set(colunas_datas)),
        'texto': list(set(colunas_texto)),
        'ids': list(set(colunas_id))
    }

def limpar_dataframe_completo(df):
    """
    Função principal de limpeza do DataFrame
    """
    df_limpo = df.copy()
    
    try:
        # 1. Identifica tipos de colunas
        tipos_colunas = identificar_colunas_por_tipo(df_limpo)
        
        # 2. Limpa colunas numéricas
        for coluna in tipos_colunas['numericas']:
            try:
                df_limpo[coluna] = limpar_coluna_numerica(df_limpo[coluna])
            except:
                pass
        
        # 3. Converte colunas de data
        for coluna in tipos_colunas['datas']:
            try:
                df_limpo[coluna] = pd.to_datetime(df_limpo[coluna], dayfirst=True, errors='coerce')
            except:
                pass
        
        # 4. Limpa colunas de texto
        for coluna in tipos_colunas['texto']:
            try:
                df_limpo[coluna] = df_limpo[coluna].astype(str).str.strip()
                df_limpo[coluna] = df_limpo[coluna].replace(['nan', 'None', 'null', ''], 'NÃO INFORMADO')
            except:
                df_limpo[coluna] = 'ERRO_PROCESSAMENTO'
        
        # 5. Limpa colunas de ID
        for coluna in tipos_colunas['ids']:
            try:
                df_limpo[coluna] = df_limpo[coluna].astype(str).str.strip()
                df_limpo[coluna] = df_limpo[coluna].replace(['nan', 'None', 'null', ''], 'ID_NÃO_INFORMADO')
            except:
                df_limpo[coluna] = 'ID_ERRO'
        
        return df_limpo, tipos_colunas
        
    except Exception as e:
        # Se houver erro, retorna DataFrame original com tipos padrão
        tipos_padrao = {
            'numericas': [],
            'datas': [],
            'texto': list(df.columns),
            'ids': []
        }
        return df, tipos_padrao

# ==================== FUNÇÕES PARA ANÁLISE DE TEXTO DE PDF ====================

def analisar_texto_pdf(texto: str) -> Dict[str, Any]:
    """
    Analisa o texto extraído do PDF e extrai informações relevantes
    """
    analise = {
        "tamanho_texto": len(texto),
        "num_paginas": texto.count("--- Página") if texto else 0,
        "palavras_chave": [],
        "possiveis_tabelas": [],
        "resumo": ""
    }
    
    if not texto:
        return analise
    
    # Identifica palavras-chave comuns em relatórios
    palavras_chave_comuns = [
        "total", "média", "soma", "percentual", "relatório", "análise",
        "vendas", "receita", "lucro", "despesa", "custo", "cliente",
        "produto", "serviço", "mês", "ano", "trimestre", "fatura", "nota fiscal",
        "demonstrativo", "balanço", "patrimonial", "resultado"
    ]
    
    texto_lower = texto.lower()
    for palavra in palavras_chave_comuns:
        if palavra in texto_lower:
            analise["palavras_chave"].append(palavra)
    
    # Cria um resumo (primeiros 500 caracteres)
    analise["resumo"] = texto[:500] + "..." if len(texto) > 500 else texto
    
    return analise

# ==================== FUNÇÕES PARA ACESSO AOS DADOS ====================

def gerar_contexto_dados(df, tipos_colunas, analise, pergunta_usuario, dados_pdf=None):
    """
    Gera um contexto seguro dos dados para a LLM
    """
    try:
        # Informações básicas sempre disponíveis
        contexto = f"""
ARQUIVO CARREGADO: {st.session_state.get('nome_arquivo', 'Desconhecido')}
TIPO DE ARQUIVO: {st.session_state.get('tipo_arquivo', 'CSV').upper()}

ESTRUTURA DO ARQUIVO:"""
        
        if df is not None:
            contexto += f"""
- Colunas ({len(analise['colunas'])}): {', '.join(analise['colunas'][:10])}{'...' if len(analise['colunas']) > 10 else ''}
- Total de registros: {analise['total_linhas']}
- Colunas numéricas ({len(tipos_colunas['numericas'])}): {', '.join(tipos_colunas['numericas'][:5]) if tipos_colunas['numericas'] else 'Nenhuma'}
- Colunas de data ({len(tipos_colunas['datas'])}): {', '.join(tipos_colunas['datas'][:5]) if tipos_colunas['datas'] else 'Nenhuma'}
- Colunas de texto ({len(tipos_colunas['texto'])}): {', '.join(tipos_colunas['texto'][:5]) if tipos_colunas['texto'] else 'Nenhuma'}
"""
        
        # Adiciona informações do PDF se disponíveis
        if dados_pdf and dados_pdf.get("texto"):
            contexto += f"""
\n\nCONTEÚDO DO PDF:
- Tamanho do texto: {dados_pdf['analise']['tamanho_texto']} caracteres
- Páginas processadas: {dados_pdf['analise']['num_paginas']}
- Palavras-chave identificadas: {', '.join(dados_pdf['analise']['palavras_chave'][:10])}
- Resumo: {dados_pdf['analise']['resumo']}
"""
            
            # Se o PDF tem texto completo, inclui (limitado)
            if len(dados_pdf['texto']) < 5000:
                contexto += f"\n\nTEXTO COMPLETO DO PDF:\n{dados_pdf['texto']}"
            else:
                contexto += f"\n\nTEXTO DO PDF (primeiros 3000 caracteres):\n{dados_pdf['texto'][:3000]}..."
        
        # Estatísticas das colunas numéricas se houver DataFrame
        if df is not None and analise.get('estatisticas'):
            contexto += "\n\nESTATÍSTICAS DAS COLUNAS NUMÉRICAS:"
            for col, stats in list(analise['estatisticas'].items())[:5]:  # Limita a 5 colunas
                contexto += f"\n- {col}:"
                if stats.get('count', 0) > 0:
                    contexto += f" Média={stats['mean']:.2f}, Soma={stats['sum']:.2f}"
                    contexto += f", Mín={stats['min']:.2f}, Máx={stats['max']:.2f}"
                else:
                    contexto += " Sem dados válidos"
        
        # Amostra dos dados se houver DataFrame
        if df is not None:
            try:
                amostra_df = df.head(10).copy()
                amostra_texto = ""
                for idx, row in amostra_df.iterrows():
                    amostra_texto += f"\nLinha {idx+1}: "
                    for col in amostra_df.columns:
                        valor = row[col]
                        if pd.isna(valor):
                            amostra_texto += f"{col}=NULO, "
                        else:
                            amostra_texto += f"{col}={str(valor)[:50]}, "
                contexto += f"\n\nAMOSTRA DOS DADOS (primeiras 10 linhas):{amostra_texto}"
            except:
                contexto += "\n\nAMOSTRA DOS DADOS: Não disponível"
        
        # Pergunta do usuário
        contexto += f"\n\nPERGUNTA DO USUÁRIO: {pergunta_usuario}"
        
        return contexto
        
    except Exception as e:
        # Em caso de erro, retorna contexto mínimo
        return f"""
ERRO AO GERAR CONTEXTO COMPLETO: {str(e)}

INFORMAÇÕES BÁSICAS:
- Arquivo: {st.session_state.get('nome_arquivo', 'Desconhecido')}
- Tipo: {st.session_state.get('tipo_arquivo', 'Desconhecido')}
- Registros: {len(df) if df is not None else 0}

PERGUNTA: {pergunta_usuario}
"""

def executar_consulta_simples(df, pergunta):
    """
    Executa consultas simples baseadas em palavras-chave
    """
    if df is None:
        return None
    
    try:
        resultados = []
        pergunta_lower = pergunta.lower()
        
        # Verifica se é uma pergunta de total/soma
        if any(p in pergunta_lower for p in ['total', 'soma', 'sum']):
            for col in df.select_dtypes(include=[np.number]).columns:
                col_lower = col.lower()
                if any(p in pergunta_lower for p in [col_lower, 'valor', 'preco', 'quantidade']):
                    soma = df[col].sum()
                    if pd.notna(soma):
                        resultados.append(f"Soma de {col}: {soma:,.2f}")
        
        # Verifica se é uma pergunta de média
        if any(p in pergunta_lower for p in ['média', 'media', 'average', 'mean']):
            for col in df.select_dtypes(include=[np.number]).columns:
                col_lower = col.lower()
                if any(p in pergunta_lower for p in [col_lower, 'valor', 'preco', 'quantidade']):
                    media = df[col].mean()
                    if pd.notna(media):
                        resultados.append(f"Média de {col}: {media:,.2f}")
        
        # Verifica se é uma pergunta de contagem
        if any(p in pergunta_lower for p in ['quantos', 'quantas', 'contagem', 'count']):
            for col in df.select_dtypes(include=['object']).columns:
                col_lower = col.lower()
                if any(p in pergunta_lower for p in [col_lower, 'categoria', 'tipo', 'produto']):
                    contagem = len(df[col].unique())
                    resultados.append(f"Número de valores únicos em {col}: {contagem}")
        
        return resultados if resultados else None
        
    except Exception as e:
        return [f"Erro na consulta: {str(e)}"]

# ==================== FUNÇÕES DE ANÁLISE ====================

def analisar_estrutura_arquivo(df, tipos_colunas):
    """
    Analisa automaticamente a estrutura do DataFrame
    """
    analise = {
        'total_linhas': len(df) if df is not None else 0,
        'total_colunas': len(df.columns) if df is not None else 0,
        'colunas': list(df.columns) if df is not None else [],
        'tipos_colunas': tipos_colunas,
        'estatisticas': {},
        'tem_datas_validas': False,
        'periodo_inicio': None,
        'periodo_fim': None
    }
    
    if df is None:
        return analise
    
    try:
        # Estatísticas para colunas numéricas
        for coluna in tipos_colunas['numericas']:
            try:
                valores_validos = df[coluna].dropna()
                if not valores_validos.empty:
                    analise['estatisticas'][coluna] = {
                        'min': float(valores_validos.min()) if not pd.isna(valores_validos.min()) else None,
                        'max': float(valores_validos.max()) if not pd.isna(valores_validos.max()) else None,
                        'mean': float(valores_validos.mean()) if not pd.isna(valores_validos.mean()) else None,
                        'sum': float(valores_validos.sum()) if not pd.isna(valores_validos.sum()) else None,
                        'count': int(len(valores_validos))
                    }
            except:
                pass
        
        # Período para colunas de data
        for coluna in tipos_colunas['datas']:
            try:
                datas_validas = df[coluna].dropna()
                if not datas_validas.empty:
                    analise['tem_datas_validas'] = True
                    if analise['periodo_inicio'] is None or datas_validas.min() < analise['periodo_inicio']:
                        analise['periodo_inicio'] = datas_validas.min()
                    if analise['periodo_fim'] is None or datas_validas.max() > analise['periodo_fim']:
                        analise['periodo_fim'] = datas_validas.max()
            except:
                pass
                
    except Exception as e:
        st.warning(f"Aviso na análise: {str(e)}")
    
    return analise

def diagnosticar_dados(df_original, df_limpo, tipos_colunas):
    """
    Mostra diagnóstico detalhado da limpeza dos dados
    """
    if df_original is None or df_limpo is None:
        return
    
    st.subheader("🔍 Diagnóstico de Limpeza de Dados")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Linhas originais", len(df_original))
    with col2:
        st.metric("Linhas após limpeza", len(df_limpo))
    
    # Mostra diagnóstico apenas para as primeiras 5 colunas para não sobrecarregar
    for i, coluna in enumerate(df_original.columns[:5]):
        with st.expander(f"📊 Coluna: {coluna}"):
            # Tipo identificado
            tipo_detectado = "Não identificado"
            for tipo, cols in tipos_colunas.items():
                if coluna in cols:
                    tipo_detectado = tipo.upper()
                    break
            
            st.write(f"**Tipo identificado:** {tipo_detectado}")
            
            # Amostra antes/depois
            col_antes, col_depois = st.columns(2)
            
            with col_antes:
                st.write("**Antes da limpeza:**")
                amostra_antes = df_original[coluna].head(5).tolist()
                st.write(amostra_antes)
                
                nulos_antes = df_original[coluna].isna().sum()
                st.write(f"Nulos: {nulos_antes} ({nulos_antes/len(df_original)*100:.1f}%)")
            
            with col_depois:
                st.write("**Depois da limpeza:**")
                amostra_depois = df_limpo[coluna].head(5).tolist()
                st.write(amostra_depois)
                
                nulos_depois = df_limpo[coluna].isna().sum()
                st.write(f"Nulos: {nulos_depois} ({nulos_depois/len(df_limpo)*100:.1f}%)")
    
    if len(df_original.columns) > 5:
        st.info(f"Mostrando diagnóstico das primeiras 5 colunas. Total: {len(df_original.columns)} colunas.")

# ==================== FUNÇÃO DE CARREGAMENTO ====================

@st.cache_data
def carregar_arquivo_csv(arquivo):
    """
    Carrega qualquer arquivo CSV com separador ; e aplica limpeza automática
    """
    try:
        # Tenta diferentes encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df_original = None
        
        for encoding in encodings:
            try:
                # Volta ao início do arquivo
                arquivo.seek(0)
                df_original = pd.read_csv(arquivo, sep=';', encoding=encoding)
                break
            except:
                continue
        
        if df_original is None:
            return None, None, None, "Erro: Não foi possível ler o arquivo com os encodings disponíveis"
        
        # Aplica limpeza completa
        df_limpo, tipos_colunas = limpar_dataframe_completo(df_original)
        
        return df_limpo, df_original, tipos_colunas, "Arquivo carregado e limpo com sucesso!"
        
    except Exception as e:
        return None, None, None, f"Erro ao carregar arquivo: {str(e)}"

def carregar_arquivo_pdf(arquivo):
    """
    Carrega e processa arquivo PDF com melhor tratamento de erros
    """
    try:
        with st.spinner("Processando PDF..."):
            dados_pdf = processar_pdf_para_analise(arquivo)
            
            if dados_pdf["mensagem_erro"] and "Nenhuma tabela" in dados_pdf["mensagem_erro"]:
                # Apenas aviso, não erro crítico
                st.warning(dados_pdf["mensagem_erro"])
                dados_pdf["mensagem_erro"] = None
            
            # Se encontrou tabelas, processa
            if dados_pdf.get("tabelas") is not None:
                try:
                    # Limpa os dados da tabela
                    df_limpo = validar_e_limpar_dados_pdf(dados_pdf["tabelas"])
                    
                    if not df_limpo.empty:
                        # Aplica limpeza completa
                        df_limpo_final, tipos_colunas = limpar_dataframe_completo(df_limpo)
                        dados_pdf["dataframe"] = df_limpo_final
                        dados_pdf["tipos_colunas"] = tipos_colunas
                        
                        # Remove colunas de origem se desejar
                        if '_origem_pagina' in dados_pdf["dataframe"].columns:
                            dados_pdf["dataframe"] = dados_pdf["dataframe"].drop(['_origem_pagina', '_origem_tabela'], axis=1, errors='ignore')
                    else:
                        st.warning("As tabelas extraídas estão vazias após limpeza.")
                except Exception as e:
                    st.warning(f"Erro ao processar tabelas do PDF: {str(e)}")
            
            return dados_pdf, None, "PDF processado com sucesso!"
            
    except Exception as e:
        return None, None, f"Erro ao processar PDF: {str(e)}"

# ==================== PROMPT DO SISTEMA ====================

CUSTOM_PROMPT = """
Você é o "FD Consultoria de Dados", um assistente de IA especialista em análise de dados. 
Sua função é ajudar usuários a extrair insights dos dados fornecidos.

DIRETRIZES PARA RESPOSTA:
1. **BASEIE-SE NOS DADOS**: Use apenas as informações fornecidas no contexto
2. **SEJA ESPECÍFICO**: Cite valores reais dos dados quando possível
3. **SEJA CLARO**: Use linguagem simples e direta
4. **FORMATE BEM**: Use **negrito** para números importantes
5. **SEJA HONESTO**: Se não souber, diga que precisa de mais informações

ANALISE OS DADOS FORNECIDOS E RESPONDA À PERGUNTA DO USUÁRIO!
"""

# ==================== INTERFACE STREAMLIT ====================

# Barra lateral
with st.sidebar:
    st.title("📊 FD Consultoria de Dados")
    st.markdown("### Análise Inteligente de Dados")
    
    # Campo para API Key
    groq_api_key = st.text_input(
        "🔑 Sua API Key Groq", 
        type="password",
        help="Obtenha sua chave em https://console.groq.com/keys"
    )
    
    st.divider()
    
    # Upload do arquivo
    st.subheader("📁 Upload de Arquivo")
    st.markdown("**Formatos suportados:** CSV (separador ;) ou PDF")
    
    arquivo_dados = st.file_uploader(
        "Selecione o arquivo CSV ou PDF",
        type=['csv', 'pdf'],
        help="Faça upload de arquivos CSV com separador ; ou arquivos PDF"
    )
    
    if arquivo_dados:
        st.success(f"✅ Arquivo selecionado: {arquivo_dados.name}")
        st.info(f"Tipo: {arquivo_dados.type}")
    
    st.divider()
    
    # Informações de formato
    with st.expander("📋 Exemplos de formatos"):
        st.code("""
# CSV - Exemplo de vendas
Codigo;Data;Produto;Quantidade;Preco
V001;01/01/2024;Produto A;2;50,00

# PDF - Documentos com texto e tabelas
- Relatórios financeiros
- Faturas e notas fiscais
- Planilhas exportadas
        """, language="text")
    
    st.divider()
    st.caption("Versão 7.1 - Suporte a PDF com Tratamento Robusto")

# Área principal
st.title("📈 FD Consultoria de Dados")
st.title("Assistente Inteligente para Análise de Dados")

# Inicializa variáveis de sessão
if "df_limpo" not in st.session_state:
    st.session_state.df_limpo = None
if "df_original" not in st.session_state:
    st.session_state.df_original = None
if "tipos_colunas" not in st.session_state:
    st.session_state.tipos_colunas = None
if "analise" not in st.session_state:
    st.session_state.analise = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "nome_arquivo" not in st.session_state:
    st.session_state.nome_arquivo = None
if "tipo_arquivo" not in st.session_state:
    st.session_state.tipo_arquivo = None
if "dados_pdf" not in st.session_state:
    st.session_state.dados_pdf = None

# Processa o arquivo quando carregado
if arquivo_dados is not None:
    # Verifica se é um arquivo novo
    if st.session_state.nome_arquivo != arquivo_dados.name:
        tipo_arquivo = arquivo_dados.type
        
        if tipo_arquivo == "application/pdf":
            # Processa PDF
            dados_pdf, _, mensagem = carregar_arquivo_pdf(arquivo_dados)
            
            if dados_pdf:
                st.session_state.dados_pdf = dados_pdf
                st.session_state.nome_arquivo = arquivo_dados.name
                st.session_state.tipo_arquivo = "pdf"
                
                # Se encontrou tabelas no PDF, define como DataFrame
                if dados_pdf.get("dataframe") is not None:
                    st.session_state.df_limpo = dados_pdf["dataframe"]
                    st.session_state.tipos_colunas = dados_pdf["tipos_colunas"]
                    st.session_state.analise = analisar_estrutura_arquivo(
                        dados_pdf["dataframe"], 
                        dados_pdf["tipos_colunas"]
                    )
                else:
                    st.session_state.df_limpo = None
                    st.session_state.tipos_colunas = None
                    st.session_state.analise = {'total_linhas': 0, 'total_colunas': 0, 'colunas': []}
                
                st.session_state.messages = []  # Limpa histórico
                st.success(f"✅ Arquivo PDF '{arquivo_dados.name}' processado com sucesso!")
                
                # Mostra informações do PDF
                st.subheader("📄 Informações do PDF")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tamanho do texto", f"{dados_pdf['analise']['tamanho_texto']:,} caracteres")
                with col2:
                    st.metric("Páginas", dados_pdf['analise']['num_paginas'])
                with col3:
                    st.metric("Palavras-chave", len(dados_pdf['analise']['palavras_chave']))
                
                if dados_pdf['analise']['palavras_chave']:
                    st.write("**Palavras-chave encontradas:**", ", ".join(dados_pdf['analise']['palavras_chave'][:10]))
                
                # Se encontrou tabelas, mostra preview
                if dados_pdf.get("dataframe") is not None and not dados_pdf["dataframe"].empty:
                    st.subheader("📊 Tabelas extraídas do PDF")
                    st.dataframe(dados_pdf["dataframe"].head(100), use_container_width=True)
                elif dados_pdf.get("texto"):
                    with st.expander("📝 Visualizar texto extraído"):
                        st.text_area("Conteúdo do PDF", dados_pdf["texto"][:5000], height=300)
            else:
                st.error(f"❌ {mensagem}")
        
        else:
            # Processa CSV
            with st.spinner("Carregando e limpando dados..."):
                df_limpo, df_original, tipos_colunas, mensagem = carregar_arquivo_csv(arquivo_dados)
            
            if df_limpo is not None:
                # Salva na sessão
                st.session_state.df_limpo = df_limpo
                st.session_state.df_original = df_original
                st.session_state.tipos_colunas = tipos_colunas
                st.session_state.nome_arquivo = arquivo_dados.name
                st.session_state.tipo_arquivo = "csv"
                st.session_state.analise = analisar_estrutura_arquivo(df_limpo, tipos_colunas)
                st.session_state.dados_pdf = None
                st.session_state.messages = []  # Limpa histórico
                
                st.success(f"✅ Arquivo CSV '{arquivo_dados.name}' carregado com sucesso!")
                
                # Mostra diagnóstico
                diagnosticar_dados(df_original, df_limpo, tipos_colunas)
            else:
                st.error(f"❌ {mensagem}")
    
    # Se já tem dados carregados
    if st.session_state.df_limpo is not None or st.session_state.dados_pdf:
        
        # Métricas principais
        st.subheader("📊 Resumo dos Dados")
        
        if st.session_state.tipo_arquivo == "pdf" and st.session_state.dados_pdf:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tipo", "PDF")
            with col2:
                st.metric("Tamanho texto", f"{st.session_state.dados_pdf['analise']['tamanho_texto']:,} caracteres")
            with col3:
                st.metric("Páginas", st.session_state.dados_pdf['analise']['num_paginas'])
            
            if st.session_state.df_limpo is not None:
                col4, col5 = st.columns(2)
                with col4:
                    st.metric("Linhas", len(st.session_state.df_limpo))
                with col5:
                    st.metric("Colunas", len(st.session_state.df_limpo.columns))
        
        elif st.session_state.df_limpo is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Tipo", st.session_state.tipo_arquivo.upper())
            with col2:
                st.metric("Total de Linhas", f"{st.session_state.analise['total_linhas']:,}")
            with col3:
                st.metric("Total de Colunas", st.session_state.analise['total_colunas'])
            with col4:
                if st.session_state.analise.get('tem_datas_validas', False):
                    st.metric("Período", f"{formatar_data_segura(st.session_state.analise['periodo_inicio'])}")
        
        # Abas para visualização
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dados", "📈 Análise", "🔍 Estrutura", "💬 Chat"])
        
        with tab1:
            st.subheader("Visualização dos Dados")
            if st.session_state.tipo_arquivo == "pdf" and st.session_state.dados_pdf.get("texto"):
                st.subheader("📝 Texto Extraído")
                with st.expander("Visualizar texto completo", expanded=False):
                    st.text_area("Conteúdo do PDF", st.session_state.dados_pdf["texto"], height=400)
            
            if st.session_state.df_limpo is not None:
                st.subheader("📊 Dados Estruturados")
                st.dataframe(st.session_state.df_limpo.head(100), use_container_width=True)
        
        with tab2:
            st.subheader("Análise dos Dados")
            
            if st.session_state.tipo_arquivo == "pdf" and st.session_state.dados_pdf.get("analise"):
                st.write("**Palavras-chave identificadas:**")
                if st.session_state.dados_pdf['analise']['palavras_chave']:
                    cols = st.columns(4)
                    for i, palavra in enumerate(st.session_state.dados_pdf['analise']['palavras_chave'][:12]):
                        with cols[i % 4]:
                            st.write(f"- {palavra}")
                else:
                    st.write("Nenhuma palavra-chave identificada")
                
                st.write("**Resumo do conteúdo:**")
                st.info(st.session_state.dados_pdf['analise']['resumo'])
            
            # Gráficos automáticos se houver DataFrame
            if st.session_state.df_limpo is not None and st.session_state.tipos_colunas:
                tipos = st.session_state.tipos_colunas
                if tipos['numericas']:
                    coluna_graf = st.selectbox("Selecione uma coluna para visualizar:", tipos['numericas'])
                    if coluna_graf:
                        try:
                            # Prepara dados para o gráfico
                            dados_graf = st.session_state.df_limpo[coluna_graf].value_counts().head(20)
                            if not dados_graf.empty:
                                st.bar_chart(dados_graf)
                            else:
                                # Tenta gráfico de linha
                                st.line_chart(st.session_state.df_limpo[coluna_graf].head(100))
                        except Exception as e:
                            st.warning(f"Não foi possível gerar o gráfico: {str(e)}")
        
        with tab3:
            st.subheader("Estrutura Detectada")
            
            if st.session_state.tipo_arquivo == "pdf":
                st.write("**📄 Metadados do PDF:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"- Nome: {st.session_state.dados_pdf['metadados']['nome_arquivo']}")
                    st.write(f"- Tamanho: {st.session_state.dados_pdf['metadados']['tamanho']:,} bytes")
                with col2:
                    st.write(f"- Tabelas encontradas: {st.session_state.dados_pdf.get('tabelas_encontradas', 0)}")
                    st.write(f"- Páginas: {st.session_state.dados_pdf['analise']['num_paginas']}")
            
            if st.session_state.df_limpo is not None and st.session_state.tipos_colunas:
                tipos = st.session_state.tipos_colunas
                
                # Colunas por tipo
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**📊 Colunas Numéricas**")
                    if tipos['numericas']:
                        for col in tipos['numericas'][:10]:
                            st.write(f"- {col}")
                        if len(tipos['numericas']) > 10:
                            st.write(f"... e mais {len(tipos['numericas'])-10} colunas")
                    else:
                        st.write("Nenhuma detectada")
                
                with col2:
                    st.write("**📅 Colunas de Data**")
                    if tipos['datas']:
                        for col in tipos['datas']:
                            st.write(f"- {col}")
                    else:
                        st.write("Nenhuma detectada")
                
                st.write("**📝 Colunas de Texto**")
                if tipos['texto']:
                    for col in tipos['texto'][:10]:  # Limita a 10
                        st.write(f"- {col}")
                    if len(tipos['texto']) > 10:
                        st.write(f"... e mais {len(tipos['texto'])-10} colunas")
        
        with tab4:
            st.subheader("💬 Chat com Assistente")
            st.markdown("*Faça perguntas sobre seus dados!*")
            
            # Exibe mensagens anteriores
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Campo de pergunta
            if prompt := st.chat_input("Faça sua pergunta sobre os dados..."):
                
                # Verifica API Key
                if not groq_api_key:
                    st.warning("⚠️ Insira sua API Key da Groq na barra lateral")
                    st.stop()
                
                try:
                    # Cria cliente Groq
                    client = Groq(api_key=groq_api_key)
                    
                    # Adiciona mensagem do usuário
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Prepara dados para análise
                    df = st.session_state.df_limpo
                    tipos = st.session_state.tipos_colunas
                    analise = st.session_state.analise
                    dados_pdf = st.session_state.dados_pdf
                    
                    # Tenta executar consulta simples primeiro (se houver DataFrame)
                    with st.spinner("Analisando dados..."):
                        resultados_consulta = None
                        if df is not None:
                            resultados_consulta = executar_consulta_simples(df, prompt)
                        
                        # Gera contexto seguro
                        contexto = gerar_contexto_dados(df, tipos, analise, prompt, dados_pdf)
                        
                        # Se encontrou resultados de consulta, adiciona ao contexto
                        if resultados_consulta:
                            contexto += "\n\nRESULTADOS DE CONSULTA ESPECÍFICA:\n" + "\n".join(resultados_consulta)
                    
                    # Prepara mensagens para API
                    system_prompt = CUSTOM_PROMPT
                    if st.session_state.tipo_arquivo == "pdf":
                        system_prompt += "\n\nNOTA SOBRE PDF: O usuário forneceu um arquivo PDF. Analise o conteúdo extraído para responder à pergunta."
                    
                    messages_for_api = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": contexto}
                    ]
                    
                    # Gera resposta
                    with st.chat_message("assistant"):
                        with st.spinner("Gerando resposta..."):
                            try:
                                chat_completion = client.chat.completions.create(
                                    messages=messages_for_api,
                                    model="llama-3.3-70b-versatile",
                                    temperature=0.2,
                                    max_tokens=2048,
                                )
                                
                                resposta = chat_completion.choices[0].message.content
                                st.markdown(resposta)
                                
                                st.session_state.messages.append({"role": "assistant", "content": resposta})
                            except Exception as e:
                                erro_msg = f"Erro na API: {str(e)}"
                                st.error(erro_msg)
                                st.session_state.messages.append({"role": "assistant", "content": f"❌ {erro_msg}"})
                
                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")
                    st.code(traceback.format_exc())

else:
    # Mensagem inicial
    st.info("👋 Olá! Sou a FD Consultoria de Dados - Versão com Suporte a PDF!")
    st.markdown("""
    ### 📋 Para começar:
    1. 🔑 Insira sua API Key Groq na barra lateral
    2. 📁 Faça upload de um arquivo **CSV** (separador ;) ou **PDF**
    3. 💬 Faça perguntas sobre seus dados
    
    ### 🎯 Exemplos de perguntas:
    **Para CSV:**
    - "Qual é o total de vendas?"
    - "Mostre a média de valores"
    - "Quantos registros existem?"
    
    **Para PDF:**
    - "Resuma o conteúdo deste PDF"
    - "Quais são os principais tópicos abordados?"
    - "Extraia as informações financeiras"
    - "O que diz sobre vendas no documento?"
    """)

# Rodapé
st.markdown(
    """
    <div style="text-align: center; color: gray; margin-top: 50px;">
        <hr>
        <p>FD Consultoria de Dados - Versão 7.1 (Suporte a PDF com Tratamento Robusto)</p>
    </div>
    """,
    unsafe_allow_html=True
)