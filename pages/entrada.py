import streamlit as st
import pandas as pd
from st_supabase_connection import SupabaseConnection
from PIL import Image
from datetime import datetime, timedelta
import plotly.express as px
import hashlib
import secrets
import random
import string
import re
from io import BytesIO

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Gestão de Backlog Pro", layout="wide")

# 2. INICIALIZAÇÃO DO ESTADO DA SESSÃO
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_nome' not in st.session_state:
    st.session_state.user_nome = None
if 'login_attempts' not in st.session_state:
    st.session_state.login_attempts = {}
if 'reset_step' not in st.session_state:
    st.session_state.reset_step = None          # 'request' ou 'verify'
if 'reset_email' not in st.session_state:
    st.session_state.reset_email = None
if 'reset_code' not in st.session_state:
    st.session_state.reset_code = None

# 3. CONEXÃO SUPABASE
try:
    conn = st.connection(
        "supabase",
        type=SupabaseConnection
    )
except Exception as e:
    st.error(f"Erro na conexão com Supabase: {str(e)}")
    st.stop()

# ==========================================
# 4. FUNÇÕES DE HASH DE SENHA
# ==========================================
def hash_senha(senha: str, salt: str = None) -> tuple:
    if salt is None:
        salt = secrets.token_hex(16)
    senha_com_salt = senha + salt
    hash_obj = hashlib.pbkdf2_hmac('sha256', senha_com_salt.encode(), salt.encode(), 100000)
    return hash_obj.hex(), salt

def verificar_senha(senha: str, hash_armazenado: str, salt: str) -> bool:
    if not salt:
        return False
    novo_hash, _ = hash_senha(senha, salt)
    return novo_hash == hash_armazenado

# ==========================================
# 5. FUNÇÕES DE RECUPERAÇÃO DE SENHA (SEM 2FA)
# ==========================================
def solicitar_reset_senha(email: str):
    """Gera código de 6 dígitos e armazena na tabela (simula envio por e-mail)"""
    try:
        resp = conn.table("usuarios").select("id").eq("email", email).execute()
        if not resp.data:
            return False, "E-mail não cadastrado"

        user_id = resp.data[0]['id']
        reset_code = ''.join(random.choices(string.digits, k=6))
        expires_at = datetime.now() + timedelta(minutes=15)

        # Atualiza o código e expiração na tabela (assumindo que as colunas existem)
        conn.table("usuarios").update({
            "reset_code": reset_code,
            "reset_expires": expires_at.isoformat()
        }).eq("id", user_id).execute()

        # Em ambiente real, enviar e-mail. Aqui apenas exibimos o código.
        st.info(f"Código de recuperação (simulado): **{reset_code}** - válido por 15 minutos")
        return True, reset_code
    except Exception as e:
        return False, f"Erro: {str(e)}"

def verificar_reset_code(email: str, code: str):
    """Verifica se o código é válido e não expirou"""
    try:
        resp = conn.table("usuarios").select("reset_code", "reset_expires").eq("email", email).execute()
        if not resp.data:
            return False, "E-mail não encontrado"
        usuario = resp.data[0]
        stored_code = usuario.get('reset_code')
        expires_str = usuario.get('reset_expires')
        if not stored_code or stored_code != code:
            return False, "Código inválido"
        if expires_str:
            exp = datetime.fromisoformat(expires_str.replace('Z', '+00:00'))
            if datetime.now() > exp:
                return False, "Código expirado. Solicite um novo."
        return True, "Código válido"
    except Exception as e:
        return False, f"Erro: {str(e)}"

def redefinir_senha(email: str, nova_senha: str):
    """Redefine a senha e limpa os campos de reset"""
    try:
        if len(nova_senha) < 6:
            return False, "A senha deve ter no mínimo 6 caracteres"
        novo_hash, novo_salt = hash_senha(nova_senha)
        conn.table("usuarios").update({
            "senha_hash": novo_hash,
            "salt": novo_salt,
            "reset_code": None,
            "reset_expires": None
        }).eq("email", email).execute()
        return True, "Senha redefinida com sucesso! Faça login."
    except Exception as e:
        return False, f"Erro: {str(e)}"

def fazer_login(email: str, senha: str):
    try:
        response = conn.table("usuarios").select("*").eq("email", email).execute()
        if not response.data:
            return False, "E-mail não encontrado"
        usuario = response.data[0]
        tentativas = st.session_state.login_attempts.get(email, 0)
        if tentativas >= 5:
            return False, "Muitas tentativas. Tente novamente mais tarde."
        if not verificar_senha(senha, usuario['senha_hash'], usuario.get('salt', '')):
            st.session_state.login_attempts[email] = tentativas + 1
            return False, "Senha incorreta"
        st.session_state.login_attempts[email] = 0
        conn.table("usuarios").update({
            "last_login": datetime.now().isoformat()
        }).eq("id", usuario['id']).execute()
        st.session_state.authenticated = True
        st.session_state.user_email = usuario['email']
        st.session_state.user_id = usuario['id']
        st.session_state.user_nome = usuario.get('nome_completo', 'Usuário')
        return True, "Login realizado com sucesso!"
    except Exception as e:
        return False, f"Erro: {str(e)}"

def fazer_logout():
    st.session_state.authenticated = False
    st.session_state.user_email = None
    st.session_state.user_id = None
    st.session_state.user_nome = None
    st.session_state.reset_step = None
    st.session_state.reset_email = None
    return True

def alterar_senha(senha_atual: str, nova_senha: str):
    try:
        response = conn.table("usuarios").select("*").eq("id", st.session_state.user_id).execute()
        if not response.data:
            return False, "Usuário não encontrado"
        usuario = response.data[0]
        if not verificar_senha(senha_atual, usuario['senha_hash'], usuario.get('salt', '')):
            return False, "Senha atual incorreta"
        if len(nova_senha) < 6:
            return False, "A nova senha deve ter no mínimo 6 caracteres"
        novo_hash, novo_salt = hash_senha(nova_senha)
        conn.table("usuarios").update({
            "senha_hash": novo_hash,
            "salt": novo_salt
        }).eq("id", st.session_state.user_id).execute()
        return True, "Senha alterada com sucesso!"
    except Exception as e:
        return False, f"Erro: {str(e)}"

# ==========================================
# 6. TELA DE AUTENTICAÇÃO (APENAS LOGIN + ESQUECI SENHA)
# ==========================================
def tela_autenticacao():
    try:
        img_logo = Image.open("logo.png")
        st.sidebar.image(img_logo, use_container_width=True)
    except:
        st.sidebar.warning("Arquivo logo.png não encontrado na pasta.")

    st.title("🔐 Sistema de Gestão de Backlog")
    st.markdown("### Faça login para acessar o sistema")

    # Etapa de redefinição de senha
    if st.session_state.reset_step == "verify":
        st.subheader("🔑 Redefinir senha")
        with st.form("form_reset_verify"):
            email = st.text_input("E-mail", value=st.session_state.reset_email, disabled=True)
            code = st.text_input("Código de verificação", placeholder="Digite o código recebido")
            nova_senha = st.text_input("Nova senha", type="password", placeholder="Mínimo 6 caracteres")
            confirmar = st.text_input("Confirmar nova senha", type="password")
            if st.form_submit_button("Redefinir senha", use_container_width=True):
                if code and nova_senha and confirmar:
                    if nova_senha != confirmar:
                        st.error("As senhas não coincidem")
                    elif len(nova_senha) < 6:
                        st.error("A senha deve ter no mínimo 6 caracteres")
                    else:
                        valido, msg = verificar_reset_code(email, code)
                        if valido:
                            sucesso, msg2 = redefinir_senha(email, nova_senha)
                            if sucesso:
                                st.success(msg2)
                                st.session_state.reset_step = None
                                st.session_state.reset_email = None
                                st.rerun()
                            else:
                                st.error(msg2)
                        else:
                            st.error(msg)
                else:
                    st.warning("Preencha todos os campos")
        if st.button("Voltar ao login"):
            st.session_state.reset_step = None
            st.session_state.reset_email = None
            st.rerun()
        return

    # Login normal
    with st.form("form_login"):
        email = st.text_input("E-mail", placeholder="seu@email.com")
        senha = st.text_input("Senha", type="password", placeholder="••••••••")
        submit_login = st.form_submit_button("Entrar", use_container_width=True)

        if submit_login:
            if email and senha:
                with st.spinner("Autenticando..."):
                    sucesso, mensagem = fazer_login(email, senha)
                    if sucesso:
                        st.success(mensagem)
                        st.rerun()
                    else:
                        st.error(mensagem)
            else:
                st.warning("Preencha todos os campos")

    # Link "Esqueci minha senha"
    with st.expander("🔒 Esqueci minha senha"):
        with st.form("form_reset_request"):
            reset_email = st.text_input("Digite seu e-mail cadastrado")
            if st.form_submit_button("Enviar código de recuperação"):
                if reset_email:
                    with st.spinner("Enviando código..."):
                        sucesso, msg = solicitar_reset_senha(reset_email)
                        if sucesso:
                            st.success("Código gerado! (verifique a mensagem informativa acima)")
                            st.session_state.reset_step = "verify"
                            st.session_state.reset_email = reset_email
                            st.rerun()
                        else:
                            st.error(msg)
                else:
                    st.warning("Informe seu e-mail")

# ==========================================
# 7. FUNÇÕES AUXILIARES DO SISTEMA (CRUD, GRÁFICOS)
# ==========================================
def calcular_metricas(r, e, i, g, u, t):
    m_rei = r * e * i
    m_gut = g * u * t
    return m_rei, m_gut, (m_rei + m_gut)

def cor_status(val):
    if val == "Atrasada":
        color = "#ff4b4b"
    elif val == "Em Andamento":
        color = "#faca2b"
    elif val == "Concluida":
        color = "#00c853"
    else:
        color = "#333333"
    return f'background-color: {color}; color: white; font-weight: bold'

def limpar_dados(df):
    if df.empty:
        return df
    colunas_texto = [
        'tarefa_product_backlog', 'setor_grooming_detalhamento',
        'pessoa_responsavel', 'origem_nao_conformidade',
        'status', 'detalhamento_acoes'
    ]
    for coluna in colunas_texto:
        if coluna in df.columns:
            df[coluna] = df[coluna].astype(str)
            df[coluna] = df[coluna].str.strip()
            df[coluna] = df[coluna].str.replace('\n', ' ')
            df[coluna] = df[coluna].str.replace('\r', ' ')
            df[coluna] = df[coluna].str.replace(r'\s+', ' ', regex=True)
            df[coluna] = df[coluna].replace(['nan', 'null', 'None', 'NaN'], '')
            df[coluna] = df[coluna].replace('', 'Não Atribuído')
    if 'status' in df.columns:
        df['status'] = df['status'].replace({
            'finalizada': 'Concluida', 'Finalizada': 'Concluida',
            'concluida': 'Concluida', 'concluído': 'Concluida',
            'em andamento': 'Em Andamento', 'Em andamento': 'Em Andamento',
            'atrasada': 'Atrasada', 'atrasado': 'Atrasada'
        })
    colunas_inteiras = [
        'resultado_rei', 'execucao_rei', 'investimento_rei', 'matriz_rei',
        'gravidade_gut', 'urgencia_gut', 'tendencia_gut', 'matriz_gut', 'soma_gut_rei'
    ]
    for col in colunas_inteiras:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')
    colunas_data = ['data_previsao_conclusao', 'data_conclusao']
    for col in colunas_data:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    hoje = pd.Timestamp(datetime.now().date())
    if 'data_previsao_conclusao' in df.columns and 'data_conclusao' in df.columns and 'status' in df.columns:
        mascara_atraso = (
            df['data_conclusao'].isna() &
            df['data_previsao_conclusao'].notna() &
            (df['data_previsao_conclusao'] < hoje) &
            (df['status'] != 'Concluida')
        )
        df.loc[mascara_atraso, 'status'] = 'Atrasada'
    return df

def carregar_dados():
    if not st.session_state.authenticated:
        return pd.DataFrame()
    try:
        query = conn.table("tarefasqrz").select("*").execute()
        df = pd.DataFrame(query.data)
        if not df.empty:
            df = limpar_dados(df)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()

# ==========================================
# 8. INTERFACE PRINCIPAL (APÓS LOGIN - SEM 2FA)
# ==========================================
def interface_principal():
    try:
        img_logo = Image.open("logo.png")
        st.sidebar.image(img_logo, use_container_width=True)
    except:
        st.sidebar.warning("Arquivo logo.png não encontrado na pasta.")

    st.sidebar.divider()
    st.sidebar.markdown(f"### 👤 {st.session_state.user_nome}")
    st.sidebar.markdown(f"📧 {st.session_state.user_email}")

    # Menu simplificado (apenas Dashboard e Alterar Senha)
    menu_usuario = st.sidebar.selectbox("Menu", ["📊 Dashboard", "🔑 Alterar Senha"])

    if st.sidebar.button("🚪 Sair do Sistema", use_container_width=True):
        fazer_logout()
        st.rerun()
    st.sidebar.divider()

    if menu_usuario == "🔑 Alterar Senha":
        st.title("🔑 Alterar Senha")
        with st.form("form_alterar_senha"):
            senha_atual = st.text_input("Senha atual", type="password")
            nova_senha = st.text_input("Nova senha", type="password", help="Mínimo 6 caracteres")
            confirmar_senha = st.text_input("Confirmar nova senha", type="password")
            if st.form_submit_button("Alterar Senha", use_container_width=True):
                if nova_senha == confirmar_senha:
                    if len(nova_senha) >= 6:
                        with st.spinner("Alterando senha..."):
                            sucesso, mensagem = alterar_senha(senha_atual, nova_senha)
                            if sucesso:
                                st.success(mensagem)
                                st.balloons()
                            else:
                                st.error(mensagem)
                    else:
                        st.error("A nova senha deve ter no mínimo 6 caracteres")
                else:
                    st.error("As senhas não coincidem")
        return

    # DASHBOARD (restante do código original, inalterado)
    st.title("📋 Sistema de Controle de Tarefas")
    df = carregar_dados()

    if not df.empty:
        if 'user_id' in df.columns:
            usuarios_response = conn.table("usuarios").select("id, nome_completo, email").execute()
            usuarios_dict = {u['id']: u['nome_completo'] for u in usuarios_response.data}
            df['criado_por'] = df['user_id'].map(usuarios_dict).fillna('Sistema')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Tarefas", len(df))
        with col2:
            st.metric("Concluídas", len(df[df['status'] == 'Concluida']))
        with col3:
            st.metric("Em Andamento", len(df[df['status'] == 'Em Andamento']))
        with col4:
            st.metric("Atrasadas", len(df[df['status'] == 'Atrasada']))
        st.divider()

    aba1, aba2 = st.tabs(["🔍 Consulta e Edição", "➕ Novo Registro"])

    with aba1:
        if not df.empty:
            st.subheader("📊 Distribuição de Tarefas por Responsável e Status")
            df_agrupado = df.groupby(['pessoa_responsavel', 'status']).size().reset_index(name='quantidade')
            ordem = df.groupby('pessoa_responsavel').size().sort_values(ascending=False).index
            df_agrupado['pessoa_responsavel'] = pd.Categorical(df_agrupado['pessoa_responsavel'], categories=ordem, ordered=True)
            df_agrupado = df_agrupado.sort_values('pessoa_responsavel')
            fig = px.bar(
                df_agrupado, x='pessoa_responsavel', y='quantidade', color='status',
                title="Tarefas por Responsável e Status",
                labels={'pessoa_responsavel': 'Responsável', 'quantidade': 'Número de Tarefas', 'status': 'Status'},
                color_discrete_map={'Concluida': '#00c853', 'Em Andamento': '#faca2b', 'Atrasada': '#ff4b4b'},
                barmode='group', text='quantidade'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400, legend_title="Status", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_title="", margin=dict(t=50, b=100))
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            st.divider()

            termo = st.text_input("🔍 Localizar tarefa (ID, Nome, Responsável, Setor ou Origem)")
            df_filtrado = df[df.apply(lambda row: termo.lower() in row.astype(str).str.lower().values, axis=1)] if termo else df
            st.subheader(f"Registros Encontrados: {len(df_filtrado)}")
            colunas_exibir = ['id_registro', 'tarefa_product_backlog', 'pessoa_responsavel', 'status', 'criado_por', 'data_previsao_conclusao']
            colunas_disponiveis = [col for col in colunas_exibir if col in df_filtrado.columns]
            st.dataframe(df_filtrado[colunas_disponiveis].style.map(cor_status, subset=['status']), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("📝 Editar Registro")
            lista_ids = df_filtrado['id_registro'].tolist()
            if lista_ids:
                id_selecionado = st.selectbox("Selecione o ID para editar", options=lista_ids)
                if id_selecionado:
                    dados_atuais = df[df['id_registro'] == id_selecionado].iloc[0]
                    data_previsao = dados_atuais.get('data_previsao_conclusao')
                    if data_previsao and pd.notna(data_previsao):
                        data_previsao = data_previsao.date() if hasattr(data_previsao, 'date') else data_previsao
                    else:
                        data_previsao = None
                    data_conclusao = dados_atuais.get('data_conclusao')
                    if data_conclusao and pd.notna(data_conclusao):
                        data_conclusao = data_conclusao.date() if hasattr(data_conclusao, 'date') else data_conclusao
                        tarefa_ja_concluida = True
                    else:
                        data_conclusao = None
                        tarefa_ja_concluida = False
                    status_atual = dados_atuais.get('status', 'Em Andamento')
                    if status_atual == "Finalizada":
                        status_atual = "Concluida"
                    if 'criado_por' in dados_atuais:
                        st.info(f"📝 Criado por: {dados_atuais['criado_por']}")

                    with st.form("form_edicao"):
                        col_e1, col_e2 = st.columns(2)
                        with col_e1:
                            novo_titulo = st.text_input("Tarefa", value=dados_atuais.get('tarefa_product_backlog', ''))
                            novo_setor = st.text_input("Setor (Grooming Detalhamento)", value=dados_atuais.get('setor_grooming_detalhamento', ''))
                            novo_resp = st.text_input("Responsável", value=dados_atuais.get('pessoa_responsavel', ''))
                            status_lista = ["Em Andamento", "Atrasada", "Concluida"]
                            idx_status = status_lista.index(status_atual) if status_atual in status_lista else 0
                            novo_status = st.selectbox("Status", status_lista, index=idx_status)
                        with col_e2:
                            nova_origem = st.text_input("Origem (Não Conformidade)", value=dados_atuais.get('origem_nao_conformidade', ''))
                            novas_acoes = st.text_area("Ações", value=dados_atuais.get('detalhamento_acoes', ''))
                        st.divider()
                        st.write("### 🧮 Prioridade (Editar REI e GUT)")
                        val_r = int(dados_atuais['resultado_rei']) if pd.notna(dados_atuais.get('resultado_rei')) else 3
                        val_e = int(dados_atuais['execucao_rei']) if pd.notna(dados_atuais.get('execucao_rei')) else 3
                        val_i = int(dados_atuais['investimento_rei']) if pd.notna(dados_atuais.get('investimento_rei')) else 3
                        val_g = int(dados_atuais['gravidade_gut']) if pd.notna(dados_atuais.get('gravidade_gut')) else 3
                        val_u = int(dados_atuais['urgencia_gut']) if pd.notna(dados_atuais.get('urgencia_gut')) else 3
                        val_t = int(dados_atuais['tendencia_gut']) if pd.notna(dados_atuais.get('tendencia_gut')) else 3
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**REI**")
                            novo_r = st.slider("Resultado", 1, 5, val_r, key="edit_r")
                            novo_e = st.slider("Execução", 1, 5, val_e, key="edit_e")
                            novo_i = st.slider("Investimento", 1, 5, val_i, key="edit_i")
                        with c2:
                            st.markdown("**GUT**")
                            novo_g = st.slider("Gravidade", 1, 5, val_g, key="edit_g")
                            novo_u = st.slider("Urgência", 1, 5, val_u, key="edit_u")
                            novo_t = st.slider("Tendência", 1, 5, val_t, key="edit_t")
                        st.divider()
                        st.write("### 📅 Datas")
                        col_data1, col_data2 = st.columns(2)
                        with col_data1:
                            nova_data_previsao = st.date_input("Data Previsão de Conclusão", value=data_previsao, format="DD/MM/YYYY")
                        with col_data2:
                            if tarefa_ja_concluida:
                                st.info(f"📅 Tarefa concluída em: {data_conclusao.strftime('%d/%m/%Y') if data_conclusao else 'N/A'}")
                                nova_data_conclusao = data_conclusao
                            else:
                                nova_data_conclusao = st.date_input("Data de Conclusão", value=None, format="DD/MM/YYYY", help="Selecione uma data para concluir a tarefa ou deixe em branco")
                        if st.form_submit_button("💾 Atualizar Dados"):
                            m_rei, m_gut, soma = calcular_metricas(novo_r, novo_e, novo_i, novo_g, novo_u, novo_t)
                            update_payload = {
                                "tarefa_product_backlog": novo_titulo, "setor_grooming_detalhamento": novo_setor,
                                "pessoa_responsavel": novo_resp, "origem_nao_conformidade": nova_origem,
                                "status": novo_status, "detalhamento_acoes": novas_acoes,
                                "resultado_rei": novo_r, "execucao_rei": novo_e, "investimento_rei": novo_i,
                                "gravidade_gut": novo_g, "urgencia_gut": novo_u, "tendencia_gut": novo_t,
                                "matriz_rei": m_rei, "matriz_gut": m_gut, "soma_gut_rei": soma,
                                "data_previsao_conclusao": nova_data_previsao.isoformat() if nova_data_previsao else None,
                                "data_conclusao": nova_data_conclusao.isoformat() if nova_data_conclusao else None,
                                "ultimo_usuario_edicao": st.session_state.user_nome,
                                "data_ultima_edicao": datetime.now().isoformat()
                            }
                            if 'user_id' in dados_atuais and pd.notna(dados_atuais['user_id']):
                                update_payload["user_id"] = dados_atuais['user_id']
                            else:
                                update_payload["user_id"] = st.session_state.user_id
                            if nova_data_conclusao and not tarefa_ja_concluida:
                                update_payload["status"] = "Concluida"
                            try:
                                conn.table("tarefasqrz").update(update_payload).eq("id_registro", id_selecionado).execute()
                                st.success(f"✅ Registro {id_selecionado} atualizado com sucesso!")
                                st.rerun()
                            except Exception as err:
                                st.error(f"Erro ao atualizar: {err}")
            else:
                st.info("Nenhum registro disponível para edição")

            st.divider()
            with st.expander("🗑️ Excluir Registro"):
                if not df.empty:
                    opcoes = df.apply(lambda row: f"{row['id_registro']} - {row['tarefa_product_backlog'][:40]} | Setor: {row.get('setor_grooming_detalhamento', '')} | Origem: {row.get('origem_nao_conformidade', '')} | Resp: {row.get('pessoa_responsavel', '')}", axis=1).tolist()
                    id_selecionado_excluir = st.selectbox("Selecione o registro para remoção", options=opcoes, key="excluir_select")
                    id_excluir = int(id_selecionado_excluir.split(" - ")[0])
                    registro = df[df['id_registro'] == id_excluir].iloc[0]
                    st.info(f"**Tarefa:** {registro['tarefa_product_backlog']}\n**Setor:** {registro.get('setor_grooming_detalhamento', 'N/A')}\n**Origem:** {registro.get('origem_nao_conformidade', 'N/A')}\n**Responsável:** {registro.get('pessoa_responsavel', 'N/A')}\n**Status:** {registro.get('status', 'N/A')}")
                    if st.button("Remover Definitivamente", type="primary", key="excluir_btn"):
                        try:
                            conn.table("tarefasqrz").delete().eq("id_registro", id_excluir).execute()
                            st.success(f"ID {id_excluir} removido com sucesso!")
                            st.rerun()
                        except Exception as err:
                            st.error(f"Erro ao remover: {err}")
                else:
                    st.info("Nenhum registro disponível para exclusão")
        else:
            st.info("Nenhuma tarefa encontrada no banco de dados.")

    with aba2:
        st.subheader("➕ Cadastrar Nova Tarefa")
        with st.form("form_novo", clear_on_submit=True):
            col_inf1, col_inf2 = st.columns(2)
            with col_inf1:
                t_backlog = st.text_input("Tarefa Product Backlog")
                setor = st.text_input("Setor (Grooming Detalhamento)")
                resp = st.text_input("Pessoa Responsável")
            with col_inf2:
                origem = st.text_input("Origem (Não Conformidade)")
                status_cad = st.selectbox("Status Inicial", ["Em Andamento", "Atrasada", "Concluida"])
                acoes_init = st.text_area("Detalhamento de Ações")
            st.divider()
            st.write("### 📅 Datas")
            col_data_n1, col_data_n2 = st.columns(2)
            with col_data_n1:
                data_previsao_nova = st.date_input("Data Previsão de Conclusão", value=None, format="DD/MM/YYYY")
            with col_data_n2:
                data_conclusao_nova = st.date_input("Data de Conclusão", value=None, format="DD/MM/YYYY", help="Preencha apenas se a tarefa já estiver concluída")
            st.divider()
            st.write("### 🧮 Definição de Prioridade (1 a 5)")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**REI**")
                r = st.slider("Resultado", 1, 5, 3, key="r_new")
                e = st.slider("Execução", 1, 5, 3, key="e_new")
                i = st.slider("Investimento", 1, 5, 3, key="i_new")
            with c2:
                st.markdown("**GUT**")
                g = st.slider("Gravidade", 1, 5, 3, key="g_new")
                u = st.slider("Urgência", 1, 5, 3, key="u_new")
                t = st.slider("Tendência", 1, 5, 3, key="t_new")
            if st.form_submit_button("🚀 Salvar no Supabase"):
                if not t_backlog.strip():
                    st.error("O campo Tarefa é obrigatório")
                else:
                    m_rei, m_gut, soma = calcular_metricas(r, e, i, g, u, t)
                    if status_cad == "Concluida" and not data_conclusao_nova:
                        data_conclusao_nova = datetime.now().date()
                    payload = {
                        "tarefa_product_backlog": t_backlog.strip(), "setor_grooming_detalhamento": setor.strip() if setor else '',
                        "pessoa_responsavel": resp.strip() if resp else '', "origem_nao_conformidade": origem.strip() if origem else '',
                        "status": status_cad, "detalhamento_acoes": acoes_init.strip() if acoes_init else '',
                        "resultado_rei": r, "execucao_rei": e, "investimento_rei": i,
                        "gravidade_gut": g, "urgencia_gut": u, "tendencia_gut": t,
                        "matriz_rei": m_rei, "matriz_gut": m_gut, "soma_gut_rei": soma,
                        "data_previsao_conclusao": data_previsao_nova.isoformat() if data_previsao_nova else None,
                        "data_conclusao": data_conclusao_nova.isoformat() if data_conclusao_nova else None,
                        "user_id": st.session_state.user_id,
                        "ultimo_usuario_edicao": st.session_state.user_nome,
                        "data_ultima_edicao": datetime.now().isoformat()
                    }
                    try:
                        conn.table("tarefasqrz").insert(payload).execute()
                        st.success(f"✅ Tarefa cadastrada com sucesso! Prioridade Total: {soma}")
                        st.balloons()
                        st.rerun()
                    except Exception as err:
                        st.error(f"Erro ao salvar: {err}")

# ==========================================
# 9. FLUXO PRINCIPAL
# ==========================================
def main():
    if not st.session_state.authenticated:
        tela_autenticacao()
    else:
        interface_principal()

if __name__ == "__main__":
    main()

# dentro de usuarios.py, no final da interface
if st.sidebar.button("← Voltar ao menu principal"):
    st.switch_page("main.py")