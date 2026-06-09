import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from st_supabase_connection import SupabaseConnection
from groq import Groq
from datetime import datetime, timedelta
import warnings
import hashlib
import secrets
import smtplib
import random
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PIL import Image  # <-- Adicionado para corrigir erro de importação

# ==========================================
# 1. CONFIGURAÇÃO E CONEXÃO
# ==========================================
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="Controle Industrial - QRZ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# INICIALIZAÇÃO DO ESTADO DA SESSÃO
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
    st.session_state.reset_step = None  # 'request', 'verify'
if 'reset_email' not in st.session_state:
    st.session_state.reset_email = None
if 'reset_code' not in st.session_state:
    st.session_state.reset_code = None

# --- CONEXÃO SUPABASE ---
try:
    conn = st.connection(
        "supabase",
        type=SupabaseConnection
    )
except Exception as e:
    st.error(f"Erro na conexão com Supabase: {str(e)}")
    st.stop()

# --- CONFIGURAÇÃO GROQ ---
try:
    API_KEY_GROQ = st.secrets["GROQ_API_KEY"]
except Exception as e:
    API_KEY_GROQ = ""
    st.error("Chave API do Groq não configurada nos segredos (secrets.toml).")

# --- CONFIGURAÇÃO SMTP (lida do secrets.toml) ---
# Estrutura esperada no secrets.toml:
# [email]
# smtp_server = "smtp.gmail.com"
# smtp_port = 587
# smtp_user = "seu-email@gmail.com"
# smtp_password = "sua-senha-de-app"
try:
    email_config = st.secrets.get("email", {})
    SMTP_SERVER = email_config.get("smtp_server")
    SMTP_PORT = email_config.get("smtp_port", 587)  # padrão 587 se não informado
    SMTP_USER = email_config.get("smtp_user")
    SMTP_PASSWORD = email_config.get("smtp_password")
    
    # Verificação básica para evitar erro silencioso
    if not all([SMTP_SERVER, SMTP_USER, SMTP_PASSWORD]):
        st.warning("Configurações de e-mail incompletas no secrets.toml. O envio de e-mails pode falhar.")
except Exception:
    SMTP_SERVER = None
    SMTP_USER = None
    SMTP_PASSWORD = None
    st.warning("Seção [email] não encontrada no secrets.toml. Configure para habilitar envio de e-mails.")

# ==========================================
# 2. FUNÇÕES DE AUTENTICAÇÃO (SEM REGISTRO E SEM 2FA)
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

def enviar_email(destinatario, assunto, corpo):
    """Envia email usando SMTP com credenciais do secrets.toml"""
    # Verifica se as credenciais foram carregadas corretamente
    if not all([SMTP_SERVER, SMTP_USER, SMTP_PASSWORD]):
        print("Credenciais SMTP não configuradas. E-mail não enviado.")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = destinatario
        msg['Subject'] = assunto
        msg.attach(MIMEText(corpo, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Erro ao enviar email: {e}")
        return False

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

def solicitar_reset_senha(email: str):
    """Gera código de 6 dígitos e envia por email, armazena na tabela"""
    try:
        # Verificar se email existe
        response = conn.table("usuarios").select("id").eq("email", email).execute()
        if not response.data:
            return False, "E-mail não cadastrado"
        
        usuario_id = response.data[0]['id']
        # Gerar código de 6 dígitos
        reset_code = ''.join(random.choices(string.digits, k=6))
        expires_at = datetime.now() + timedelta(minutes=15)
        
        # Atualizar ou inserir código na tabela (assumindo colunas reset_code, reset_expires)
        conn.table("usuarios").update({
            "reset_code": reset_code,
            "reset_expires": expires_at.isoformat()
        }).eq("id", usuario_id).execute()
        
        # Enviar email
        assunto = "Recuperação de senha - QRZ Industrial"
        corpo = f"""
        Olá,

        Você solicitou a recuperação de senha para sua conta no QRZ Industrial.

        Seu código de verificação é: {reset_code}

        Este código é válido por 15 minutos.

        Se não foi você quem solicitou, ignore este e-mail.

        Atenciosamente,
        Equipe QRZ
        """
        enviado = enviar_email(email, assunto, corpo)
        if not enviado:
            # Fallback: exibir código na tela (para desenvolvimento)
            st.warning(f"⚠️ Não foi possível enviar e-mail. Código de recuperação: **{reset_code}** (válido por 15 min)")
        return True, "Código enviado para seu e-mail."
    except Exception as e:
        return False, f"Erro: {str(e)}"

def verificar_reset_code(email: str, code: str):
    """Verifica se o código é válido e não expirou"""
    try:
        response = conn.table("usuarios").select("reset_code", "reset_expires").eq("email", email).execute()
        if not response.data:
            return False, "E-mail não encontrado"
        
        usuario = response.data[0]
        stored_code = usuario.get('reset_code')
        expires_str = usuario.get('reset_expires')
        
        if not stored_code or stored_code != code:
            return False, "Código inválido"
        
        if expires_str:
            expires = datetime.fromisoformat(expires_str.replace('Z', '+00:00'))
            if datetime.now() > expires:
                return False, "Código expirado. Solicite um novo."
        
        return True, "Código válido"
    except Exception as e:
        return False, f"Erro: {str(e)}"

def redefinir_senha(email: str, nova_senha: str):
    """Redefine a senha e limpa o código de reset"""
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

def fazer_logout():
    st.session_state.authenticated = False
    st.session_state.user_email = None
    st.session_state.user_id = None
    st.session_state.user_nome = None
    st.session_state.reset_step = None
    st.session_state.reset_email = None
    st.session_state.reset_code = None
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
# 3. TELA DE LOGIN (APENAS LOGIN + ESQUECI SENHA)
# ==========================================
def tela_login():
    """Tela de login com opção de recuperar senha"""
    try:
        img_logo = Image.open("logo.png")
        st.sidebar.image(img_logo, use_container_width=True)
    except:
        st.sidebar.title("QRZ Analytics")
    
    st.title("🔐 Sistema de Visualização de Dados")
    st.markdown("### Faça login para acessar o dashboard")
    
    # Se estiver em etapa de reset de senha
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
                        # Verificar código
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
        
        col1, col2 = st.columns([1, 1])
        with col1:
            submit_login = st.form_submit_button("Entrar", use_container_width=True)
        with col2:
            pass
        
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
                            st.success(msg)
                            st.session_state.reset_step = "verify"
                            st.session_state.reset_email = reset_email
                            st.rerun()
                        else:
                            st.error(msg)
                else:
                    st.warning("Informe seu e-mail")

# ==========================================
# 4. CARGA E TRATAMENTO DE DADOS (IGUAL AO ORIGINAL)
# ==========================================
@st.cache_data(ttl=60)
def carregar_dados():
    try:
        response = conn.table("tarefasqrz").select("*").execute()
        df = pd.DataFrame(response.data)
        if df.empty:
            return pd.DataFrame()
        cols_num = ['resultado_rei', 'execucao_rei', 'investimento_rei', 
                    'gravidade_gut', 'urgencia_gut', 'tendencia_gut', 
                    'matriz_rei', 'matriz_gut', 'soma_gut_rei', 'percentual_ok']
        for col in cols_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        cols_data = ['data_previsao_conclusao', 'data_conclusao', 'data_criacao']
        for col in cols_data:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Erro de conexão: {e}")
        return pd.DataFrame()

def criar_gauge(valor, titulo, max_val=100, cor_hex="#4CAF50"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor,
        title={'text': titulo, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [None, max_val], 'tickwidth': 1},
            'bar': {'color': cor_hex},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [{'range': [0, max_val], 'color': '#262730'}],
        }
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=20), 
                      paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig

# ==========================================
# 5. DASHBOARD PRINCIPAL (APÓS LOGIN) - SEM 2FA
# ==========================================
def interface_principal():
    # Sidebar
    try:
        st.sidebar.image("logo.png", use_container_width=True)
    except:
        st.sidebar.title("QRZ Analytics")
    
    st.sidebar.markdown(f"### 👤 {st.session_state.user_nome}")
    st.sidebar.markdown(f"📧 {st.session_state.user_email}")
    
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
    
    # Dashboard principal (igual ao original)
    st.title("📊 Dashboard de Performance GRC")
    
    st.sidebar.header("🔎 Filtros Globais")
    df = carregar_dados()
    if df.empty:
        st.info("Sem dados para exibir.")
        return
    
    df['pessoa_responsavel'] = df['pessoa_responsavel'].fillna("Não atribuído").astype(str)
    df['status'] = df['status'].fillna("Sem status").astype(str)
    
    lista_resp = ["Todos"] + sorted(df['pessoa_responsavel'].unique().tolist())
    filtro_resp = st.sidebar.selectbox("Responsável", lista_resp)
    lista_status = ["Todos"] + sorted(df['status'].unique().tolist())
    filtro_status = st.sidebar.selectbox("Status", lista_status)
    
    df_filtered = df.copy()
    if filtro_resp != "Todos":
        df_filtered = df_filtered[df_filtered['pessoa_responsavel'] == filtro_resp]
    if filtro_status != "Todos":
        df_filtered = df_filtered[df_filtered['status'] == filtro_status]
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Visão Geral e Métricas", "🔥 Priorização GUT", 
        "💎 Análise REI", "⚠️ Tarefas Críticas", "🤖 Chat IA"
    ])
    
    with tab1:
        st.subheader("Visão Geral Detalhada")
        df_display = df_filtered.copy()
        if 'percentual_ok' in df_display.columns:
            if df_display['percentual_ok'].max() <= 1.0 and df_display['percentual_ok'].max() > 0:
                df_display['percentual_ok'] = df_display['percentual_ok'] * 100
            df_display['percentual_ok'] = df_display['percentual_ok'].fillna(0).astype(int).clip(0, 100)
        
        mapa_colunas = {
            'id_registro': 'ID',
            'tarefa_product_backlog': 'Tarefa',
            'pessoa_responsavel': 'Responsável',
            'percentual_ok': 'Progresso',
            'status': 'Status',
            'data_previsao_conclusao': 'Previsão'
        }
        cols_para_mostrar = [c for c in mapa_colunas.keys() if c in df_display.columns]
        df_final = df_display[cols_para_mostrar].rename(columns=mapa_colunas)
        st.dataframe(
            df_final,
            column_config={
                "ID": st.column_config.NumberColumn("ID", format="%d", width="small"),
                "Tarefa": st.column_config.TextColumn("Tarefa", width="large"),
                "Progresso": st.column_config.ProgressColumn("Progresso", format="%d%%", min_value=0, max_value=100),
                "Previsão": st.column_config.DateColumn("Previsão", format="YYYY-MM-DD")
            },
            use_container_width=True,
            hide_index=True
        )
        st.markdown("---")
        st.subheader("📈 Métricas de Performance")
        total = len(df_filtered)
        concluidas = len(df_filtered[(df_filtered['percentual_ok'] >= 100) | (df_filtered['status'] == 'Finalizada')])
        perc_geral = (concluidas / total * 100) if total > 0 else 0
        prog_medio = df_display['percentual_ok'].mean()
        med_gut = df_filtered['matriz_gut'].mean() if 'matriz_gut' in df_filtered.columns else 0
        med_rei = df_filtered['matriz_rei'].mean() if 'matriz_rei' in df_filtered.columns else 0
        
        g1, g2, g3, g4 = st.columns(4)
        with g1: st.plotly_chart(criar_gauge(perc_geral, "Conclusão Geral (%)", 100, "#4CAF50"), use_container_width=True)
        with g2: st.plotly_chart(criar_gauge(prog_medio, "Progresso Médio (%)", 100, "#2196F3"), use_container_width=True)
        with g3: st.plotly_chart(criar_gauge(med_gut, "Média GUT", 125, "#FFC107"), use_container_width=True)
        with g4: st.plotly_chart(criar_gauge(med_rei, "Média REI", 125, "#E91E63"), use_container_width=True)
        
        st.subheader("📊 Visualizações Avançadas")
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.markdown("**Top Tarefas por Percentual**")
            df_top = df_display.nlargest(10, 'percentual_ok').sort_values('percentual_ok', ascending=True)
            fig_bar = px.bar(df_top, x="percentual_ok", y="tarefa_product_backlog", orientation='h', 
                             text="percentual_ok", color="percentual_ok", color_continuous_scale="Viridis")
            fig_bar.update_layout(showlegend=False, yaxis={'title': ''})
            fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)
        with c2:
            st.markdown("**Distribuição de Ocorrências**")
            if 'soma_gut_rei' not in df_filtered.columns: 
                df_filtered['soma_gut_rei'] = 1
            fig_tree = px.treemap(df_filtered, path=[px.Constant("Todas"), 'status', 'tarefa_product_backlog'], 
                                  values='soma_gut_rei', color='percentual_ok', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_tree, use_container_width=True)
    
    with tab2:
        st.subheader("🎯 Priorização GUT")
        fig_gut = px.scatter(df_filtered, x="gravidade_gut", y="urgencia_gut", size="tendencia_gut", 
                             color="status", hover_data=["tarefa_product_backlog"], title="Dispersão GUT")
        st.plotly_chart(fig_gut, use_container_width=True)
        st.subheader("Top 10 Tarefas por Prioridade GUT")
        top_gut = df_filtered.nlargest(10, 'matriz_gut')[['tarefa_product_backlog', 'pessoa_responsavel', 'matriz_gut', 'status']]
        st.dataframe(top_gut, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("📊 Análise REI")
        fig_rei = px.bar(df_filtered.head(15), x="tarefa_product_backlog", y=["resultado_rei", "execucao_rei", "investimento_rei"], 
                         title="Composição REI", barmode='group')
        st.plotly_chart(fig_rei, use_container_width=True)
        st.subheader("Top 10 Tarefas por Prioridade REI")
        top_rei = df_filtered.nlargest(10, 'matriz_rei')[['tarefa_product_backlog', 'pessoa_responsavel', 'matriz_rei', 'status']]
        st.dataframe(top_rei, use_container_width=True, hide_index=True)
    
    with tab4:
        st.subheader("🚨 Atenção Imediata")
        criticas = df_filtered[(df_filtered['soma_gut_rei'] > 80) | (df_filtered['status'] == 'Atrasada')]
        if not criticas.empty:
            st.warning(f"⚠️ {len(criticas)} tarefas requerem atenção imediata!")
            st.dataframe(criticas[['tarefa_product_backlog', 'pessoa_responsavel', 'status', 'soma_gut_rei']], 
                        use_container_width=True, hide_index=True)
        else:
            st.success("✅ Nenhuma tarefa crítica encontrada!")
    
    with tab5:
        st.subheader("🤖 Assistente IA - Análise de Dados")
        p = st.chat_input("Pergunte sobre os dados...")
        if p:
            st.chat_message("user").write(p)
            try:
                client = Groq(api_key=API_KEY_GROQ)
                csv_mini = df_filtered[['tarefa_product_backlog', 'status', 'percentual_ok', 'pessoa_responsavel']].head(50).to_csv(index=False)
                system_prompt = """
## INSTRUÇÕES PARA RESPOSTA
Com base nos dados acima, responda à pergunta de forma clara e objetiva.
- Use as metodologias ISO 9001, Matriz GUT e Matriz REI como referência
- Use os números exatos fornecidos
- Sugira priorizações baseadas na matriz GUT
- Considere o impacto/esforço pela matriz REI
- Seja específico nas recomendações para a certificação ISO 9001
- Se perguntar sobre outro assunto, responda: "Desculpe, somente posso responder sobre ISO 9001, Matriz GUT e REI", a não ser  ser pedir o diagrama de pareto ou similar.
- Se perguntar sobre quem é você, responda: "Sou a IA de apoio à gestão para Implementação da Gestão Industrial - QRZ Consultoria"
- Responder sobre o diagrama de pareto das tarefas
Seja sempre cordial!
"""
                msg = f"Dados resumidos do dashboard:\n{csv_mini}\n\nPergunta do usuário: {p}"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": msg}
                ]
                r = client.chat.completions.create(
                    messages=messages, 
                    model="openai/gpt-oss-120b",
                    temperature=0.7,
                    max_tokens=2000
                )
                st.chat_message("assistant").write(r.choices[0].message.content)
            except Exception as e:
                st.error(f"Erro na IA: {e}")

# ==========================================
# 6. FLUXO PRINCIPAL
# ==========================================
def main():
    if not st.session_state.authenticated:
        tela_login()
    else:
        interface_principal()

if __name__ == "__main__":
    main()

# dentro de usuarios.py, no final da interface
if st.sidebar.button("← Voltar ao menu principal"):
    st.switch_page("main.py")