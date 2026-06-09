import streamlit as st
import pandas as pd
from st_supabase_connection import SupabaseConnection
from PIL import Image
from datetime import datetime
import hashlib
import secrets
import re

# ==========================================
# 1. CONFIGURAÇÃO INICIAL
# ==========================================
st.set_page_config(
    page_title="Admin - Gestão de Usuários",
    page_icon="👥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Estados de autenticação ---
if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False
if 'admin_user' not in st.session_state:
    st.session_state.admin_user = None
if 'reset_step' not in st.session_state:
    st.session_state.reset_step = None   # 'request', 'verify'
if 'reset_email' not in st.session_state:
    st.session_state.reset_email = None
if 'reset_code' not in st.session_state:
    st.session_state.reset_code = None

# --- Conexão Supabase ---
try:
    conn = st.connection(
        "supabase",
        type=SupabaseConnection
    )
except Exception as e:
    st.error(f"Erro de conexão: {e}")
    st.stop()

# ==========================================
# 2. FUNÇÕES DE HASH E UTILITÁRIOS
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

def criar_admin_se_nao_existir():
    """Cria o usuário administrador padrão (admin / fdsj#2026) se não existir"""
    try:
        # Verifica se já existe algum admin
        resp = conn.table("usuarios").select("id").eq("is_admin", True).execute()
        if not resp.data:
            # Cria o admin
            hash_senha_result, salt = hash_senha("fdsj#2026")
            novo_admin = {
                "email": "admin@qrz.com",
                "senha_hash": hash_senha_result,
                "salt": salt,
                "nome_completo": "Administrador",
                "is_admin": True,
                "two_factor_enabled": False
            }
            conn.table("usuarios").insert(novo_admin).execute()
            st.success("✅ Usuário administrador criado com sucesso! Login: admin@qrz.com / senha: fdsj#2026")
    except Exception as e:
        st.warning(f"Admin já existe ou erro: {e}")

# ==========================================
# 3. FUNÇÕES ADMINISTRATIVAS
# ==========================================
def listar_usuarios():
    """Retorna lista de todos os usuários (exceto admin)"""
    resp = conn.table("usuarios").select("id, email, nome_completo, is_admin").execute()
    df = pd.DataFrame(resp.data)
    # Filtra para mostrar apenas não-admins (opcional, pode mostrar todos)
    return df

def criar_usuario(email: str, nome: str, senha: str):
    """Cria um novo usuário comum (não admin)"""
    try:
        # Validações
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return False, "E-mail inválido"
        if len(senha) < 6:
            return False, "Senha deve ter no mínimo 6 caracteres"
        if len(nome.strip()) < 3:
            return False, "Nome deve ter no mínimo 3 caracteres"

        # Verifica duplicidade
        resp = conn.table("usuarios").select("id").eq("email", email).execute()
        if resp.data:
            return False, "E-mail já cadastrado"

        hash_s, salt = hash_senha(senha)
        novo = {
            "email": email.strip().lower(),
            "senha_hash": hash_s,
            "salt": salt,
            "nome_completo": nome.strip(),
            "is_admin": False,
            "two_factor_enabled": False
        }
        conn.table("usuarios").insert(novo).execute()
        return True, "Usuário criado com sucesso!"
    except Exception as e:
        return False, f"Erro: {e}"

def alterar_senha_usuario(user_id: int, nova_senha: str):
    """Altera a senha de qualquer usuário (admin)"""
    try:
        if len(nova_senha) < 6:
            return False, "Senha deve ter no mínimo 6 caracteres"
        novo_hash, novo_salt = hash_senha(nova_senha)
        conn.table("usuarios").update({
            "senha_hash": novo_hash,
            "salt": novo_salt
        }).eq("id", user_id).execute()
        return True, "Senha alterada com sucesso!"
    except Exception as e:
        return False, f"Erro: {e}"

def alterar_propria_senha(user_id: int, senha_atual: str, nova_senha: str):
    """Altera a senha do próprio administrador (com validação da atual)"""
    try:
        resp = conn.table("usuarios").select("senha_hash, salt").eq("id", user_id).execute()
        if not resp.data:
            return False, "Usuário não encontrado"
        usuario = resp.data[0]
        if not verificar_senha(senha_atual, usuario['senha_hash'], usuario.get('salt', '')):
            return False, "Senha atual incorreta"
        if len(nova_senha) < 6:
            return False, "Nova senha deve ter no mínimo 6 caracteres"
        novo_hash, novo_salt = hash_senha(nova_senha)
        conn.table("usuarios").update({
            "senha_hash": novo_hash,
            "salt": novo_salt
        }).eq("id", user_id).execute()
        return True, "Sua senha foi alterada!"
    except Exception as e:
        return False, f"Erro: {e}"

def solicitar_reset_senha_admin(email: str):
    """Gera código de recuperação para o administrador (simula envio por e-mail)"""
    try:
        resp = conn.table("usuarios").select("id").eq("email", email).eq("is_admin", True).execute()
        if not resp.data:
            return False, "E-mail de administrador não encontrado"
        user_id = resp.data[0]['id']
        reset_code = ''.join(secrets.choice('0123456789') for _ in range(6))
        expires_at = datetime.now() + timedelta(minutes=15)
        conn.table("usuarios").update({
            "reset_code": reset_code,
            "reset_expires": expires_at.isoformat()
        }).eq("id", user_id).execute()
        # Em ambiente real, enviar e-mail. Aqui apenas exibimos o código.
        st.info(f"Código de recuperação (simulado): **{reset_code}** - válido por 15 min")
        return True, reset_code
    except Exception as e:
        return False, f"Erro: {e}"

def redefinir_senha_admin(email: str, code: str, nova_senha: str):
    """Redefine senha do admin usando código"""
    try:
        resp = conn.table("usuarios").select("reset_code, reset_expires").eq("email", email).eq("is_admin", True).execute()
        if not resp.data:
            return False, "Administrador não encontrado"
        usuario = resp.data[0]
        if usuario['reset_code'] != code:
            return False, "Código inválido"
        if usuario['reset_expires']:
            exp = datetime.fromisoformat(usuario['reset_expires'].replace('Z', '+00:00'))
            if datetime.now() > exp:
                return False, "Código expirado"
        if len(nova_senha) < 6:
            return False, "Senha deve ter no mínimo 6 caracteres"
        novo_hash, novo_salt = hash_senha(nova_senha)
        conn.table("usuarios").update({
            "senha_hash": novo_hash,
            "salt": novo_salt,
            "reset_code": None,
            "reset_expires": None
        }).eq("email", email).execute()
        return True, "Senha redefinida com sucesso!"
    except Exception as e:
        return False, f"Erro: {e}"

# ==========================================
# 4. TELA DE LOGIN DO ADMIN
# ==========================================
def tela_login():
    st.title("👥 Painel Administrativo")
    try:
        img = Image.open("logo.png")
        st.sidebar.image(img, use_container_width=True)
    except:
        st.sidebar.write("(Logo não encontrada)")

    # Se estiver em processo de recuperação de senha do admin
    if st.session_state.reset_step == "verify":
        st.subheader("🔐 Redefinir senha do administrador")
        with st.form("form_reset_verify"):
            email = st.text_input("E-mail", value=st.session_state.reset_email, disabled=True)
            code = st.text_input("Código de verificação")
            nova_senha = st.text_input("Nova senha", type="password")
            confirmar = st.text_input("Confirmar nova senha", type="password")
            if st.form_submit_button("Redefinir senha"):
                if code and nova_senha == confirmar:
                    sucesso, msg = redefinir_senha_admin(email, code, nova_senha)
                    if sucesso:
                        st.success(msg)
                        st.session_state.reset_step = None
                        st.session_state.reset_email = None
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Preencha corretamente os campos")
        if st.button("Voltar ao login"):
            st.session_state.reset_step = None
            st.rerun()
        return

    # Login normal
    with st.form("admin_login"):
        email = st.text_input("E-mail do administrador", placeholder="admin@qrz.com")
        senha = st.text_input("Senha", type="password")
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Entrar", use_container_width=True)
        with col2:
            # Botão "Esqueci minha senha" abre etapa de recuperação
            if st.form_submit_button("Esqueci minha senha", use_container_width=True):
                st.session_state.reset_step = "request"
                st.rerun()

        if submitted:
            if email and senha:
                # Verificar se é admin no banco
                resp = conn.table("usuarios").select("id, senha_hash, salt, nome_completo, is_admin").eq("email", email).execute()
                if resp.data and resp.data[0].get('is_admin', False):
                    user = resp.data[0]
                    if verificar_senha(senha, user['senha_hash'], user.get('salt', '')):
                        st.session_state.admin_authenticated = True
                        st.session_state.admin_user = {
                            "id": user['id'],
                            "email": email,
                            "nome": user['nome_completo']
                        }
                        st.success("Login realizado!")
                        st.rerun()
                    else:
                        st.error("Senha incorreta")
                else:
                    st.error("Acesso permitido apenas para administradores")
            else:
                st.warning("Preencha e-mail e senha")

    # Etapa de solicitação de código de recuperação
    if st.session_state.reset_step == "request":
        st.subheader("🔑 Recuperar acesso de administrador")
        with st.form("form_reset_request"):
            reset_email = st.text_input("E-mail do administrador")
            if st.form_submit_button("Enviar código de recuperação"):
                if reset_email:
                    sucesso, codigo = solicitar_reset_senha_admin(reset_email)
                    if sucesso:
                        st.success("Código enviado (simulado no console/info)")
                        st.session_state.reset_step = "verify"
                        st.session_state.reset_email = reset_email
                        st.rerun()
                    else:
                        st.error(sucesso)  # sucesso é a mensagem de erro
                else:
                    st.warning("Informe o e-mail")

# ==========================================
# 5. PAINEL ADMINISTRATIVO (APÓS LOGIN)
# ==========================================
def painel_admin():
    st.title(f"👋 Olá, {st.session_state.admin_user['nome']}!")
    try:
        img = Image.open("logo.png")
        st.sidebar.image(img, use_container_width=True)
    except:
        pass
    st.sidebar.write(f"**Admin:** {st.session_state.admin_user['email']}")

    menu = st.sidebar.radio(
        "Navegação",
        ["Criar novo usuário", "Listar / Alterar senha de usuários", "Alterar minha senha", "Sair"]
    )

    if menu == "Sair":
        st.session_state.admin_authenticated = False
        st.session_state.admin_user = None
        st.rerun()

    if menu == "Criar novo usuário":
        st.header("➕ Criar novo usuário")
        with st.form("criar_usuario"):
            nome = st.text_input("Nome completo")
            email = st.text_input("E-mail")
            senha = st.text_input("Senha", type="password", help="Mínimo 6 caracteres")
            if st.form_submit_button("Criar usuário"):
                sucesso, msg = criar_usuario(email, nome, senha)
                if sucesso:
                    st.success(msg)
                    st.balloons()
                else:
                    st.error(msg)

    elif menu == "Listar / Alterar senha de usuários":
        st.header("📋 Gerenciar usuários")
        df_usuarios = listar_usuarios()
        if df_usuarios.empty:
            st.info("Nenhum usuário comum cadastrado ainda.")
        else:
            # Exibe tabela com todos os usuários (exceto administradores, opcional)
            st.dataframe(df_usuarios[['id', 'email', 'nome_completo']], use_container_width=True, hide_index=True)
            st.divider()
            st.subheader("🔧 Alterar senha de um usuário")
            # Selectbox com nomes
            usuario_opcoes = {f"{row['nome_completo']} ({row['email']})": row['id'] for _, row in df_usuarios.iterrows()}
            escolha = st.selectbox("Selecione o usuário", list(usuario_opcoes.keys()))
            user_id = usuario_opcoes[escolha]
            nova_senha = st.text_input("Nova senha", type="password", key="nova_senha_admin")
            if st.button("Alterar senha"):
                if nova_senha:
                    sucesso, msg = alterar_senha_usuario(user_id, nova_senha)
                    if sucesso:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning("Digite a nova senha")

    elif menu == "Alterar minha senha":
        st.header("🔑 Alterar sua própria senha")
        with st.form("alterar_minha_senha"):
            atual = st.text_input("Senha atual", type="password")
            nova = st.text_input("Nova senha", type="password", help="Mínimo 6 caracteres")
            confirmar = st.text_input("Confirmar nova senha", type="password")
            if st.form_submit_button("Alterar"):
                if nova == confirmar:
                    if nova:
                        sucesso, msg = alterar_propria_senha(
                            st.session_state.admin_user['id'],
                            atual,
                            nova
                        )
                        if sucesso:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.warning("Digite a nova senha")
                else:
                    st.error("As senhas não coincidem")

# ==========================================
# 6. FLUXO PRINCIPAL
# ==========================================
def main():
    # Garantir que o admin existe no banco
    criar_admin_se_nao_existir()

    if not st.session_state.admin_authenticated:
        tela_login()
    else:
        painel_admin()

if __name__ == "__main__":
    main()

# dentro de usuarios.py, no final da interface
if st.sidebar.button("← Voltar ao menu principal"):
    st.switch_page("main.py")