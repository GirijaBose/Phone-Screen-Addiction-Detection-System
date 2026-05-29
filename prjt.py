# ====================== IMPORT PACKAGES ==============
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import cv2
import mediapipe as mp
from groq import Groq
import datetime
import time
import base64
import warnings
import json
import os
import sqlite3


warnings.filterwarnings("ignore")


def init_db():
    conn = sqlite3.connect('wellness.db')
    c = conn.cursor()
    # User table with streak column
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, streak INTEGER, last_login TEXT)''')
    conn.commit()
    conn.close()

init_db() 

# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="MindGuard AI", page_icon="🧠", layout="wide")

# ====================== GROQ CONFIG ======================
client = None
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.sidebar.warning("Groq API Key missing in secrets!")

# ====================== AUTH SYSTEM ======================
USER_FILE = "users.json"

# Load users
def load_users():
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, "w") as f:
            json.dump({}, f)
    with open(USER_FILE, "r") as f:
        return json.load(f)

# Save users
def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

# ================= SESSION =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "main"

# ================= MAIN AUTH PAGE =================
def main_auth():
    # Function to convert image to base64
    def get_base64(file):
        try:
            with open(file, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            # Return a default gradient if image not found
            return ""

    # Try to load image, if not available use gradient background
    try:
        img = get_base64("1.avif")
        background_style = f"""
        .stApp {{
            background: url("data:image/avif;base64,{img}") no-repeat center center fixed;
            background-size: cover;
        }}
        """
    except:
        background_style = """
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        """

    st.markdown(f"""
    <style>
    {background_style}
    
    /* Remove default Streamlit padding */
    .main .block-container {{
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-bottom: 80px; /* Add padding for footer */
    }}
    
    /* Dark overlay for better text readability */
    .overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.6);
        z-index: 0;
    }}
    
    /* Main content container */
    .content-wrapper {{
        position: relative;
        z-index: 1;
    }}
    
    /* TOP RIGHT BUTTONS */
    .top-right {{
        position: fixed;
        top: 65px;
        right: 30px;
        display: flex;
        gap: 15px;
        z-index: 1000;
    }}
    
    .btn {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 24px;
        border-radius: 50px;
        text-decoration: none;
        font-weight: bold;
        font-size: 16px;
        border: none;
        cursor: pointer;
        transition: transform 0.3s, box-shadow 0.3s;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }}
    
    .btn:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }}
    
    /* CENTER TEXT */
    .main-title {{
        text-align: center;
        color: white;
        font-size: 70px;
        font-weight: bold;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: fadeInUp 1s ease;
    }}
    
    .subtitle {{
        text-align: center;
        color: white;
        font-size: 24px;
        margin-bottom: 40px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        animation: fadeInUp 1s ease 0.2s both;
    }}
    
    /* Feature Cards */
    .features-container {{
        display: flex;
        justify-content: center;
        gap: 30px;
        flex-wrap: wrap;
        margin-top: 50px;
        animation: fadeInUp 1s ease 0.4s both;
    }}
    
    .feature-card {{
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 30px;
        width: 250px;
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
        cursor: pointer;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }}
    
    .feature-card:hover {{
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }}
    
    .feature-icon {{
        font-size: 50px;
        margin-bottom: 15px;
    }}
    
    .feature-title {{
        font-size: 20px;
        font-weight: bold;
        color: #333;
        margin-bottom: 10px;
    }}
    
    .feature-desc {{
        font-size: 14px;
        color: #666;
    }}
    
    /* Footer Styling */
    .footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 12px;
        background: rgba(0,0,0,0.7);
        backdrop-filter: blur(10px);
        z-index: 1000;
        border-top: 1px solid rgba(255,255,255,0.1);
    }}
    
    .footer-content {{
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        flex-wrap: wrap;
    }}
    
    .footer-link {{
        color: rgba(255,255,255,0.8);
        text-decoration: none;
        font-size: 12px;
        transition: all 0.3s;
        cursor: pointer;
    }}
    
    .footer-link:hover {{
        color: white;
        text-decoration: underline;
    }}
    
    .footer-separator {{
        color: rgba(255,255,255,0.5);
        font-size: 12px;
    }}
    
    .footer-copyright {{
        color: rgba(255,255,255,0.6);
        font-size: 11px;
    }}
    
    /* Animations */
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    /* Center container */
    .center-container {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        text-align: center;
    }}
    
    </style>
    """, unsafe_allow_html=True)

    # Add overlay
    st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)

    # ================= NAVBAR BUTTONS =================
    st.markdown("""
    <style>
    .top-buttons {
        position: fixed;
        top: 65px;
        right: 40px;
        display: flex;
        gap: 0px;
        z-index: 1000;
    }

    .top-buttons button {
        background: white;
        color: black;
        border: none;
        padding: 8px 18px;
        border-radius: 12px;
        font-weight: 500;
        cursor: pointer;
        transition: 0.3s;
    }

    .top-buttons button:hover {
        background: #e6e6e6;
    }

    button {
        margin: 0px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ================= HANDLE NAVIGATION =================
    col_space, col1, col2 = st.columns([10, 0.8, 0.8])

    with col1:
        if st.button("Login"):
            st.session_state.page = "login"
            st.rerun()

    with col2:
        if st.button("Sign Up"):
            st.session_state.page = "signup"
            st.rerun()

    # ================= MAIN CONTENT =================
    st.markdown("""
        <div class="center-container">
            <div class="content-wrapper">
                <div class="main-title">
                    🧠 MindGuard AI
                </div>
                <div class="subtitle">
                    Your Intelligent Digital Wellness Companion
                </div>
                <div class="features-container">
                    <div class="feature-card">
                        <div class="feature-icon">📊</div>
                        <div class="feature-title">Smart Assessment</div>
                        <div class="feature-desc">AI-powered addiction detection</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🤖</div>
                        <div class="feature-title">AI Wellness Coach</div>
                        <div class="feature-desc">24/7 personalized guidance</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">👁️</div>
                        <div class="feature-title">Eye Strain Analysis</div>
                        <div class="feature-desc">Real-time eye health monitoring</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">📈</div>
                        <div class="feature-title">Analytics Dashboard</div>
                        <div class="feature-desc">Track your digital wellness</div>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # ================= ADD FOOTER WITH TERMS LINKS =================
    st.markdown("""
    <div class="footer">
        <div class="footer-content">
            <span class="footer-link" id="terms_footer_link">📜 Terms of Service</span>
            <span class="footer-separator">|</span>
            <span class="footer-link" id="privacy_footer_link">🔒 Privacy Policy</span>
            <span class="footer-separator">|</span>
            <span class="footer-copyright">© 2026 MindGuard AI. All rights reserved.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
  
# ================= LOGIN PAGE (Enhanced Version) =================
def update_streak_on_login(username):
    """Database-la user streak-ai track panni update pannum."""
    conn = sqlite3.connect('wellness.db')
    c = conn.cursor()
    
    # Table illana create pannum
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, streak INTEGER, last_login TEXT)''')
    
    today = datetime.date.today()
    today_str = str(today)
    yesterday_str = str(today - datetime.timedelta(days=1))
    
    c.execute("SELECT streak, last_login FROM users WHERE username=?", (username,))
    row = c.fetchone()
    
    if row:
        streak, last_login = row
        if last_login == today_str:
            # Innaiku munaadiye login panni iruntha streak-ai matha vendam
            pass
        elif last_login == yesterday_str:
            # Nethu login panni iruntha, innaiku streak + 1
            new_streak = streak + 1  # ITHU THAAN MISS AAGI IRUKKUM
            c.execute("UPDATE users SET streak=?, last_login=? WHERE username=?", (new_streak, today_str, username))
        else:
            # Romba naal gap vittu vantha streak-ai 1-nu reset pannum
            c.execute("UPDATE users SET streak=?, last_login=? WHERE username=?", (1, today_str, username))
    else:
        # Puthiya user-ku streak 1-nu start pannum
        c.execute("INSERT INTO users VALUES (?, ?, ?)", (username, 1, today_str))
    
    conn.commit()
    conn.close()
#-----login---
def login_page():
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    /* Center container */
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        padding: 20px;
    }
    
    /* Login card styling */
    .login-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 40px;
        max-width: 450px;
        width: 100%;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        animation: fadeInUp 0.6s ease;
    }
    
    .login-icon {
        text-align: center;
        font-size: 60px;
        margin-bottom: 20px;
    }
    
    .login-title {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #333;
        margin-bottom: 10px;
    }
    
    .login-subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
        font-size: 14px;
    }
    
    /* Style Streamlit inputs */
    .stTextInput > div > div > input {
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 16px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        outline: none;
    }
    
    /* Style buttons */
    .stButton > button {
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        width: 100%;
        border: none;
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Secondary button */
    .stButton > button:not([kind="primary"]) {
        background: #f5f5f5;
        color: #333;
        border: 1px solid #e0e0e0;
    }
    
    .stButton > button:not([kind="primary"]):hover {
        background: #e8e8e8;
    }
    
    /* Success/Error message styling */
    .stAlert {
        border-radius: 12px;
        margin-top: 20px;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Divider */
    .divider {
        text-align: center;
        margin: 20px 0;
        position: relative;
    }
    
    .divider::before,
    .divider::after {
        content: "";
        position: absolute;
        top: 50%;
        width: 45%;
        height: 1px;
        background: #e0e0e0;
    }
    
    .divider::before {
        left: 0;
    }
    
    .divider::after {
        right: 0;
    }
    
    .divider span {
        background: white;
        padding: 0 10px;
        color: #999;
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create centered container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Add a container for the login card
        with st.container():
            st.markdown("""
            <div class="login-icon">🔐</div>
            <div class="login-title">Welcome Back!</div>
            <div class="login-subtitle">Sign in to continue to MindGuard AI</div>
            """, unsafe_allow_html=True)
            
            # Load users
            users = load_users()
            
            # Input fields with better spacing
            username = st.text_input(
                "Username", 
                placeholder="Enter your username", 
                key="login_username_input",
                label_visibility="collapsed"
            )
            
            password = st.text_input(
                "Password", 
                type="password", 
                placeholder="Enter your password", 
                key="login_password_input",
                label_visibility="collapsed"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Buttons row
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("Login", type="primary", use_container_width=True, key="do_login_btn"):
                    if not username or not password:
                        st.warning("⚠️ Please enter both username and password")
                    elif username in users and users[username] == password:
                        update_streak_on_login(username)
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("✅ Login successful! Redirecting...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
            
            with col_btn2:
                if st.button("Back", use_container_width=True, key="back_from_login"):
                    st.session_state.page = "main"
                    st.rerun()
            
           
# ================= SIGNUP PAGE =================
def signup_page():
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    /* Center container */
    .signup-container-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        padding: 20px;
    }
    
    /* Signup card styling */
    .signup-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 40px;
        max-width: 500px;
        width: 100%;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        animation: fadeInUp 0.6s ease;
    }
    
    .signup-icon {
        text-align: center;
        font-size: 60px;
        margin-bottom: 20px;
    }
    
    .signup-title {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #333;
        margin-bottom: 10px;
    }
    
    .signup-subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
        font-size: 14px;
    }
    
    /* Style Streamlit inputs */
    .stTextInput > div > div > input {
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 16px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        outline: none;
    }
    
    /* Style buttons */
    .stButton > button {
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        width: 100%;
        border: none;
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Secondary button */
    .stButton > button:not([kind="primary"]) {
        background: #f5f5f5;
        color: #333;
        border: 1px solid #e0e0e0;
    }
    
    .stButton > button:not([kind="primary"]):hover {
        background: #e8e8e8;
    }
    
    /* Success/Error message styling */
    .stAlert {
        border-radius: 12px;
        margin-top: 20px;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Password requirements styling */
    .password-requirements {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 12px;
        margin-top: 10px;
        font-size: 12px;
        color: #666;
    }
    
    /* Divider */
    .divider {
        text-align: center;
        margin: 20px 0;
        position: relative;
    }
    
    .divider::before,
    .divider::after {
        content: "";
        position: absolute;
        top: 50%;
        width: 45%;
        height: 1px;
        background: #e0e0e0;
    }
    
    .divider::before {
        left: 0;
    }
    
    .divider::after {
        right: 0;
    }
    
    .divider span {
        background: white;
        padding: 0 10px;
        color: #999;
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create centered container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container():
            st.markdown("""
            <div class="signup-icon">🚀</div>
            <div class="signup-title">Create Account</div>
            <div class="signup-subtitle">Join MindGuard AI and start your wellness journey</div>
            """, unsafe_allow_html=True)
            
            # Load users
            users = load_users()
            
            # Input fields
            new_user = st.text_input(
                "Username", 
                placeholder="Choose a username (min. 3 characters)", 
                key="signup_username",
                label_visibility="collapsed"
            )
            
            new_pass = st.text_input(
                "Password", 
                type="password", 
                placeholder="Create a password (min. 4 characters)", 
                key="signup_password",
                label_visibility="collapsed"
            )
            
            # Real-time password requirements
            if new_pass:
                with st.container():
                    st.markdown('<div class="password-requirements">', unsafe_allow_html=True)
                    st.markdown("**Password Requirements:**")
                    col_req1, col_req2 = st.columns(2)
                    with col_req1:
                        if len(new_pass) >= 4:
                            st.markdown('<span style="color:#28a745;">✓ Minimum 4 characters</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span style="color:#dc3545;">✗ Minimum 4 characters</span>', unsafe_allow_html=True)
                    with col_req2:
                        if any(c.isdigit() for c in new_pass):
                            st.markdown('<span style="color:#28a745;">✓ Contains numbers</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span style="color:#dc3545;">✗ Contains numbers (recommended)</span>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            confirm_pass = st.text_input(
                "Confirm Password", 
                type="password", 
                placeholder="Confirm your password", 
                key="signup_confirm",
                label_visibility="collapsed"
            )
            
            # Show password match status
            if confirm_pass and new_pass:
                if new_pass == confirm_pass:
                    st.markdown('<span style="color:#28a745;">✓ Passwords match</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color:#dc3545;">✗ Passwords do not match</span>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Buttons row
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("Create Account", type="primary", use_container_width=True, key="do_signup_btn"):
                    # Validation checks
                    if not new_user or not new_pass or not confirm_pass:
                        st.warning("⚠️ Please fill all fields")
                    elif new_user in users:
                        st.error("❌ Username already exists! Please choose a different username")
                    elif len(new_user) < 3:
                        st.warning("⚠️ Username must be at least 3 characters")
                    elif len(new_pass) < 4:
                        st.warning("⚠️ Password must be at least 4 characters")
                    elif new_pass != confirm_pass:
                        st.error("❌ Passwords do not match!")
                    else:
                        # Create the account
                        users[new_user] = new_pass
                        save_users(users)
                        st.success("🎉 Account created successfully!")
                        st.balloons()
                        st.info("Redirecting to login page...")
                        time.sleep(1.5)
                        st.session_state.page = "login"
                        st.rerun()
            
            with col_btn2:
                if st.button("Back to Home", use_container_width=True, key="back_from_signup"):
                    st.session_state.page = "main"
                    st.rerun()
            
            # Terms and conditions
            st.markdown("""
            <div class="divider">
                <span>By signing up, you agree to our</span>
            </div>
            <div style="text-align: center; font-size: 12px; color: #666;">
                <a href="#" style="color: #667eea; text-decoration: none;">Terms of Service</a> and 
                <a href="#" style="color: #667eea; text-decoration: none;">Privacy Policy</a>
            </div>
            """, unsafe_allow_html=True)
    
# ================= ROUTING =================
if not st.session_state.logged_in:
    if st.session_state.page == "main":
        main_auth()
    elif st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "signup":
        signup_page()
    st.stop()

# ====================== SIDEBAR & HOME PAGE ======================

# Custom CSS for Sidebar
st.markdown("""
<style>
/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    padding-top: 20px;
}

[data-testid="stSidebar"] .stImage {
    text-align: center;
    margin-bottom: 20px;
}

[data-testid="stSidebar"] .stMarkdown {
    color: #ffffff;
}

/* Sidebar User Welcome */
.sidebar-welcome {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 15px;
    border-radius: 12px;
    margin: 10px 0;
    text-align: center;
}

.sidebar-welcome-emoji {
    font-size: 32px;
    margin-bottom: 8px;
}

.sidebar-welcome-name {
    font-size: 16px;
    font-weight: bold;
    color: white;
}

.sidebar-welcome-text {
    font-size: 12px;
    color: rgba(255,255,255,0.8);
    margin-top: 5px;
}

/* Sidebar Menu Items */
[data-testid="stSidebar"] .stRadio > div {
    gap: 8px;
}

[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.1);
    padding: 10px 15px;
    border-radius: 10px;
    transition: all 0.3s ease;
    color: #ffffff !important;
    font-weight: 500;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.2);
    transform: translateX(5px);
}

[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
    background: rgba(102, 126, 234, 0.3);
}

/* Sidebar Button */
[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 8px 16px;
    font-weight: 500;
    transition: all 0.3s ease;
}

[data-testid="stSidebar"] .stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

/* Sidebar Divider */
.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    margin: 15px 0;
}

/* Home Page Styling */
.home-hero {
    text-align: center;
    padding: 15px !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    margin-bottom: 15px !important;
    color: white;
    animation: fadeInDown 0.8s ease;
}

.home-hero h1 {
    font-size: 28px !important;
    font-weight: 800;
    margin-bottom: 15px;
    background: linear-gradient(135deg, #fff 0%, #e0e7ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.home-hero p {
    font-size: 14px !important;
    opacity: 0.95;
    max-width: 600px;
    margin: 0 auto;
}

/* Stats Cards */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    margin: 30px 0;
}

.stat-card-home {
    background: white;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
    cursor: pointer;
}

.stat-card-home:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.12);
}

.stat-number-home {
    font-size: 32px;
    font-weight: bold;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.stat-label-home {
    font-size: 14px;
    color: #666;
    font-weight: 500;
}

/* Quick Actions */
.quick-actions {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
    margin: 20px 0;
}

/* Welcome Banner */
.welcome-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    padding: 10px 30px;
    margin-bottom: 30px;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
    animation: fadeInUp 0.6s ease;
}

.welcome-text {
    font-size: 20px;
    font-weight: 600;
}

.welcome-subtext {
    font-size: 14px;
    opacity: 0.9;
    margin-top: 5px;
}

.motivation-badge {
    background: rgba(255,255,255,0.2);
    padding: 8px 20px;
    border-radius: 50px;
    font-size: 14px;
}

/* Progress Section */
.progress-section {
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    border-radius: 20px;
    padding: 25px;
    margin: 20px 0;
    border: 1px solid #e0e0e0;
}

/* Feature List */
.feature-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.feature-list li {
    padding: 12px 0;
    border-bottom: 1px solid #e0e0e0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.feature-list li:last-child {
    border-bottom: none;
}

.feature-icon-list {
    font-size: 24px;
}

/* Tip Cards */
.tip-card {
    background: #fff3e0;
    border-left: 4px solid #ff9800;
    padding: 15px;
    border-radius: 10px;
    margin: 15px 0;
}

/* Animations */
@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-up {
    animation: fadeInUp 0.6s ease forwards;
}

/* Metric Cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================

# ✅ ADD CSS HERE (RIGHT BELOW set_page_config)
st.markdown("""
<style>

/* Sidebar background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}

/* Sidebar text */
[data-testid="stSidebar"] label {
    color: white !important;
    font-weight: 500;
}

/* Radio menu items - BRIGHTER STYLING */
[data-testid="stSidebar"] .stRadio > div {
    gap: 8px;
}

[data-testid="stSidebar"] .stRadio label {
    background: rgba(255, 255, 255, 0.15) !important;
    padding: 12px 15px !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    margin: 2px 0 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    transform: translateX(5px) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
}

[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

/* Radio button circle - make it brighter */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    gap: 10px;
}

[data-testid="stSidebar"] .stRadio [role="radio"] {
    background: rgba(255, 255, 255, 0.2) !important;
}

[data-testid="stSidebar"] .stRadio [role="radio"][aria-checked="true"] {
    background: #00c6ff !important;
    box-shadow: 0 0 10px rgba(0, 198, 255, 0.5) !important;
}

/* Sidebar welcome section */
.sidebar-welcome {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 15px;
    border-radius: 12px;
    margin: 10px 0;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.sidebar-welcome-emoji {
    font-size: 32px;
    margin-bottom: 8px;
}

.sidebar-welcome-name {
    font-size: 16px;
    font-weight: bold;
    color: white;
}

.sidebar-welcome-text {
    font-size: 12px;
    color: rgba(255,255,255,0.9);
    margin-top: 5px;
}

/* Sidebar divider */
.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    margin: 15px 0;
}

/* Logout button styling */
[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #ff4b4b 0%, #ff6b6b 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

[data-testid="stSidebar"] .stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4) !important;
}

</style>
""", unsafe_allow_html=True)

with st.sidebar:
    # Logo with background circle
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    width: 80px; 
                    height: 80px; 
                    border-radius: 50%; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center;
                    margin: 0 auto;">
            <span style="font-size: 45px;">🤖</span>
        </div>
        <h2 style="color: white; margin-top: 15px;">MindGuard AI</h2>
    </div>
    """, unsafe_allow_html=True)
    if st.session_state.get('username'):
        st.markdown(f"""
        <div class="sidebar-welcome">
            <div class="sidebar-welcome-emoji">👤</div>
            <div class="sidebar-welcome-name">{st.session_state['username']}</div>
            <div class="sidebar-welcome-text">Your wellness journey continues</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Logout Button
    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.page = "main"
        if 'username' in st.session_state:
            del st.session_state.username
        st.rerun()
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Main Menu - BRIGHTER AND MORE VISIBLE
    st.markdown("<p style='color: white; font-weight: 600; margin-bottom: 10px;'>📋 MAIN MENU</p>", unsafe_allow_html=True)
    
    choice = st.radio("", [
        "🏠 Home",
        "📊 Addiction Test",
        "🤖 AI Wellness Coach",
        "📸 Visual Scan",
        "📈 Analytics & Reports"
    ], label_visibility="collapsed")
    
    # Sidebar Footer
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.6); font-size: 11px; margin-top: 20px;">
        © 2026 MindGuard AI<br>
        Your Digital Wellness Companion
    </div>
    """, unsafe_allow_html=True)
# ====================== HOME PAGE ======================

if choice == "🏠 Home":
    import random

    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 30px;">
        <h1 style="color: white; font-size: 48px; margin-bottom: 10px;">🧠 MindGuard AI</h1>
        <p style="color: white; font-size: 18px;">Your Wellness Companion</p>
    </div>
    """, unsafe_allow_html=True)

    # Welcome Banner
    if st.session_state.get('username'):
        messages = [
            "🌟 Keep pushing forward!",
            "💪 Every step counts!",
            "🎯 You're doing great!",
            "🌈 Your wellness journey is inspiring!",
            "⭐ Keep up the amazing work!"
        ]
        msg = random.choice(messages)

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 20px 30px; margin-bottom: 30px;">
            <p style="color: white; font-size: 18px; margin: 0; text-align: center;">
                👋 Welcome back, <strong>{st.session_state['username']}</strong>!<br>
                {msg}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Stats Section (only if test is done)
    if st.session_state.get('test_done'):
        risk_score = 85 if st.session_state.get('result') == 1 else 35
        wellness_score = 100 - risk_score

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Addiction Risk", f"{risk_score}%")
        with col2:
            st.metric("⏰ Avg Screen Time", "5.2 hrs")
        with col3:
            st.metric("💪 Wellness Score", f"{wellness_score}/100")

    # Features Section - CENTRALIZED
    st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>🌟 Features</h3>", unsafe_allow_html=True)
    
    # Center features using columns
    col_space1, col1, col2, col3, col4, col_space2 = st.columns([1, 2, 2, 2, 2, 1])
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <div style="font-size: 40px;">📊</div>
            <p style="font-weight: bold; margin: 10px 0 0 0;">ML-Based Test</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <div style="font-size: 40px;">🤖</div>
            <p style="font-weight: bold; margin: 10px 0 0 0;">AI Coach</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <div style="font-size: 40px;">👁️</div>
            <p style="font-weight: bold; margin: 10px 0 0 0;">Eye Scan</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <div style="font-size: 40px;">📈</div>
            <p style="font-weight: bold; margin: 10px 0 0 0;">Analytics</p>
        </div>
        """, unsafe_allow_html=True)

    # Daily Tip Section - Below Features
    st.markdown("---")
    st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>💡 Daily Tip</h3>", unsafe_allow_html=True)
    
    tips = [
        "Take a 5-minute break every hour 👁️",
        "Avoid phone during meals 🍽️",
        "Use grayscale mode 🎨",
        "Stop screen 30 mins before sleep 🌙",
        "Follow 20-20-20 rule 👓",
    ]
    
    # Center the tip
    tip_col1, tip_col2, tip_col3 = st.columns([1, 2, 1])
    with tip_col2:
        st.info(random.choice(tips))

    # Progress Section (only if test is done)
    if st.session_state.get('test_done'):
        st.markdown("---")
        st.subheader("📈 Your Progress")
        
        progress = 100 - risk_score
        st.progress(progress / 100)
        st.write(f"Wellness Progress: {progress}%")

        if risk_score > 70:
            st.warning("⚠️ Reduce screen time by 1 hour daily")
        elif risk_score > 40:
            st.info("💡 Try a 3-day detox challenge")
        else:
            st.success("✅ Great job! Maintain your streak!")

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>🌟 MindGuard AI - Digital Wellness Companion</p>", unsafe_allow_html=True)

# ====================== ADDICTION TEST ======================
elif choice == "📊 Addiction Test":

    


    st.title("📊 Smart Addiction Self-Assessment")

    uploaded_file = st.file_uploader("Upload Dataset (Final_csv.csv)", ['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file).fillna(0)

        # Drop unwanted columns
        drop_cols = [
            'Unnamed: 0',
            'Pevious semester mark percentage',
            'Number of Arrear papers',
            ' Using mobile phone for non-academic purposes',
            'Check phone during class',
            'cluster'
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Encode categorical columns
        le = preprocessing.LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = le.fit_transform(df[col].astype(str))

        # Split data
        X = df.drop('Addiction', axis=1)
        y = df['Addiction']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        st.subheader("Answer the following questions")
        st.info("⚠️ Higher levels indicate higher addiction risk")

        col1, col2 = st.columns(2)

        # ================= INPUT SECTION =================
        with col1:

            i1 = 1 if st.selectbox("Gender", ("Male", "Female")) == "Male" else 0

            # Sleep disturbance
            sleep_label = st.selectbox("Sleep disturbance?", ("Low", "Medium", "High"))
            sleep_map = {"Low": 0, "Medium": 1, "High": 2}
            i2 = sleep_map[sleep_label]

            i3 = 1 if st.selectbox("Phone immediately after waking?", ("No", "Yes")) == "Yes" else 0
            i4 = 1 if st.selectbox("Can live without phone?", ("No", "Yes")) == "Yes" else 0

            # Primary usage
            usage_label = st.selectbox(
                "Primary usage",
                ("Education", "Social Media", "Gaming", "Entertainment", "Others")
            )
            usage_map = {
                "Education": 0,
                "Social Media": 1,
                "Gaming": 2,
                "Entertainment": 3,
                "Others": 4
            }
            i5 = usage_map[usage_label]

            # Screen time
            screen_time_label = st.selectbox(
                "Daily Screen Time",
                ("< 1 hr", "1–3 hrs", "3–5 hrs", "More than 5 hrs")
            )
            screen_map = {
                "< 1 hr": 0,
                "1–3 hrs": 1,
                "3–5 hrs": 2,
                "More than 5 hrs": 3
            }
            i6 = screen_map[screen_time_label]

            # Distracted
            dist_label = st.selectbox(
                "Distracted while studying?",
                ("Never", "Sometimes", "Often", "Always")
            )
            dist_map = {
                "Never": 0,
                "Sometimes": 1,
                "Often": 2,
                "Always": 3
            }
            i7 = dist_map[dist_label]

            i8 = 1 if st.selectbox("Late night phone usage?", ("No", "Yes")) == "Yes" else 0
            i9 = 1 if st.selectbox("Sleep loss affects focus?", ("No", "Yes")) == "Yes" else 0

        with col2:

            i10 = 1 if st.checkbox("Eye strain / Headache") else 0
            i11 = 1 if st.checkbox("Anxiety") else 0
            i12 = 1 if st.checkbox("Depression") else 0
            i13 = 1 if st.checkbox("Sleep issues") else 0
            i14 = 1 if st.checkbox("Social isolation") else 0
            i15 = 1 if st.checkbox("Poor concentration") else 0
            i16 = 1 if st.checkbox("Irritability") else 0
            i17 = 1 if st.checkbox("Cognitive decline") else 0
            i18 = 1 if st.checkbox("Low self-esteem") else 0

        # ================= PREDICTION =================
        if st.button("Generate Diagnostic Report"):

            user_data = np.array([
                i1, i2, i3, i4, i5, i6, i7, i8, i9,
                i10, i11, i12, i13, i14, i15, i16, i17, i18
            ]).reshape(1, -1)

            prediction = model.predict(user_data)[0]

            # 🔥 Rule-based override
            if i6 == 3:
                prediction = 1

            # Save session
            st.session_state['result'] = prediction
            st.session_state['test_done'] = True
            st.session_state['symptom_anxiety'] = i11
            st.session_state['symptom_depression'] = i12
            st.session_state['symptom_sleep'] = i13

            # ================= OUTPUT =================
            if prediction == 1:
                st.error("🚨 Result: Smartphone Addiction Detected")
                st.balloons()
            else:
                st.success("✅ Result: Healthy Smartphone Usage")

# ==================== 🤖 AI WELLNESS COACH (SEPARATE OPTION) ====================
elif choice == "🤖 AI Wellness Coach":
    st.title("🤖 MindGuard AI Recovery Coach")
    st.write("Ask for tips or share your doubts here to recover from smartphone addiction.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your digital wellness coach. How can I help you reduce your screen time?”"}
        ]

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("E.g., How to reduce night phone usage?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if client:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {"role": "system", "content": "You are a professional digital wellness coach. Provide concise, helpful advice for overcoming smartphone addiction in a friendly tone."},
                                *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                            ]
                        )
                        res_text = response.choices[0].message.content
                        st.markdown(res_text)
                        st.session_state.messages.append({"role": "assistant", "content": res_text})
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.error("Groq API configuration missing. Please check your Secrets.")


# ====================== VISUAL SCAN ======================
elif choice == "📸 Visual Scan":

    st.title("👁️ Digital Eye Health & Facial Analysis")

    mp_face_mesh = mp.solutions.face_mesh

    # ✅ MULTI PERSON ENABLED
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    LEFT_EYE = [33,160,158,133,153,144]
    RIGHT_EYE = [362,385,387,263,373,380]

    baseline = 0.28

    if "face_data" not in st.session_state:
        st.session_state.face_data = {}

    face_data = st.session_state.face_data

    # ================= EAR =================
    def calculate_ear(landmarks, points, w, h):
        p = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in points]

        v1 = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
        v2 = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
        h1 = np.linalg.norm(np.array(p[0]) - np.array(p[3]))

        return (v1 + v2) / (2*h1 + 1e-6)

    # ================= DARK CIRCLE =================
    def detect_dark_circles(frame, landmarks, w, h):

        pts = [33,160,158,133,153,144,362,385,387,263,373,380]

        coords = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in pts]

        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        y2 = min(h, y2 + 40)

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return 0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        brightness = np.mean(gray)

        if brightness < 100:
            return 1
        else:
            return 0

    # ================= EXPRESSION =================
    def detect_expression(landmarks, w, h):
        try:
            top = landmarks[13]
            bottom = landmarks[14]
            left = landmarks[78]
            right = landmarks[308]

            top = np.array([top.x*w, top.y*h])
            bottom = np.array([bottom.x*w, bottom.y*h])
            left = np.array([left.x*w, left.y*h])
            right = np.array([right.x*w, right.y*h])

            vertical = np.linalg.norm(top - bottom)
            horizontal = np.linalg.norm(left - right)

            ratio = vertical / (horizontal + 1e-6)

            if ratio > 0.50:
                return "SURPRISED"
            elif ratio > 0.38:
                return "STRESS"
            elif ratio > 0.30:
                return "NEUTRAL"
            else:
                return "HAPPY"

        except:
            return "UNKNOWN"

    # ================= BUTTONS =================
    start = st.button("Start Camera")
    stop = st.button("Stop Camera")

    if start:

        start_time = time.time()
        cap = cv2.VideoCapture(0)
        frame_window = st.image([])

        while cap.isOpened():

            if stop:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:

                for i, face in enumerate(results.multi_face_landmarks):

                    if i not in face_data:
                        face_data[i] = {
                            "blink_count": 0,
                            "state": "open",
                            "last_blink_time": 0
                        }

                    # ===== FACE BOX =====
                    x_coords = [int(lm.x * w) for lm in face.landmark]
                    y_coords = [int(lm.y * h) for lm in face.landmark]

                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    # ===== EAR =====
                    left_ear = calculate_ear(face.landmark, LEFT_EYE, w, h)
                    right_ear = calculate_ear(face.landmark, RIGHT_EYE, w, h)
                    ear = (left_ear + right_ear) / 2

                    current_time = time.time()

                    # ===== BLINK DETECTION =====
                    if ear < baseline * 0.70:
                        if face_data[i]["state"] == "open":
                            if current_time - face_data[i]["last_blink_time"] > 0.3:
                                face_data[i]["blink_count"] += 1
                                face_data[i]["last_blink_time"] = current_time
                            face_data[i]["state"] = "closed"
                    else:
                        face_data[i]["state"] = "open"

                    bpm = face_data[i]["blink_count"] / ((time.time()-start_time)/60 + 0.01)

                    # ===== STRAIN (RESPONSIVE 🔥) =====
                    fatigue = max(0, (baseline - ear) * 200)
                    blink_score = max(0, (12 - bpm) * 2)

                    elapsed = (time.time() - start_time) / 60
                    time_score = min(10, elapsed * 0.5)

                    dark_circle = detect_dark_circles(frame, face.landmark, w, h)
                    dark_effect = dark_circle * 15

                    strain = fatigue + blink_score + time_score + dark_effect
                    strain = max(0, min(100, int(strain)))

                    # ===== EXPRESSION =====
                    expression = detect_expression(face.landmark, w, h)

                    # ===== RISK =====
                    if strain < 25:
                        risk = "LOW"
                        color = (0,255,0)
                    elif strain < 50:
                        risk = "MODERATE"
                        color = (0,255,255)
                    elif strain < 75:
                        risk = "HIGH"
                        color = (0,165,255)
                    else:
                        risk = "VERY HIGH"
                        color = (0,0,255)

                    # ===== STORE =====
                    face_data[i]["bpm"] = bpm
                    face_data[i]["strain"] = strain
                    face_data[i]["emotion"] = expression
                    face_data[i]["risk"] = risk
                    face_data[i]["dark"] = dark_circle

                    # ===== DRAW =====
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                    text_y = y_min - 10 if y_min > 30 else y_min + 20
                    dark_status = "YES" if dark_circle == 1 else "NO"

                    cv2.putText(frame, f"P{i+1} | {risk}", (x_min, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    cv2.putText(frame, f"Strain: {strain}%", (x_min, text_y+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                    cv2.putText(frame, f"Blinks: {face_data[i]['blink_count']}", (x_min, text_y+40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

                    cv2.putText(frame, f"Emotion: {expression}", (x_min, text_y+60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                    cv2.putText(frame, f"Dark Circle: {dark_status}", (x_min, text_y+80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)

                    cv2.putText(frame, f"EAR: {ear:.2f}", (x_min, text_y+100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            else:
                cv2.putText(frame, "No face detected", (30,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            frame_window.image(frame, channels="BGR")

        cap.release()

    # ================= FINAL REPORT =================
    if stop:

        st.subheader("📊 Final Eye Health Report")

        if len(face_data) == 0:
            st.warning("No data available")
        else:
            for i, data in face_data.items():

                dark_status = "YES" if data.get("dark",0) == 1 else "NO"

                st.markdown(f"### 👤 Person {i+1}")
                st.write(f" Total Blinks: {data['blink_count']}")
                st.write(f" Blink Rate (BPM): {data.get('bpm',0):.2f}")
                st.write(f" Eye Strain: {data.get('strain',0)}%")
                st.write(f" Emotion: {data.get('emotion','-')}")
                st.write(f" Dark Circle: {dark_status}")
                st.write(f" Risk Level: {data.get('risk','-')}")
                st.markdown("---")

        st.session_state.face_data = {}

    st.info("""
LOW → Healthy  
MODERATE → Take Break  
HIGH → Eye Fatigue  
VERY HIGH → High Digital Eye Damage Risk  
""")
# ==================== 📈 ANALYTICS & REPORTS ====================
elif choice == "📈 Analytics & Reports":
    st.title("📈 Personalized Digital Wellness Insights")
    
    if 'test_done' not in st.session_state:
        st.warning("No data found. Please complete the 'Addiction Test' first.")
    else:
        # Layout columns
        col_a, col_b = st.columns(2)

        with col_a:
            # 1. Addiction Probability Gauge
            risk_score = 85 if st.session_state.get('result') == 1 else 35
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = risk_score,
                title = {'text': "Addiction Probability (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#FF4B4B" if risk_score > 50 else "#00CC96"},
                    'steps': [
                        {'range': [0, 40], 'color': "#E8F5E9"},
                        {'range': [40, 70], 'color': "#FFF3E0"},
                        {'range': [70, 100], 'color': "#FFEBEE"}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_b:
            # 2. Radar Chart (Mental Health Impact)
            # Indha values test-la user kudutha inputs-ai base panni varum
            categories = ['Anxiety', 'Depression', 'Sleep Loss', 'Social Isolation', 'Low Focus']
            
            # Simple logic to generate values based on test results
            m_values = [80 if st.session_state.get('symptom_anxiety') else 20, 
                        70 if st.session_state.get('symptom_depression') else 30,
                        90 if st.session_state.get('symptom_sleep') else 25,
                        60, 85] # Static values for demo, replace with variables if needed

            fig_radar = go.Figure(data=go.Scatterpolar(
                r=m_values,
                theta=categories,
                fill='toself',
                line_color='#4A90E2'
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Psychological Impact Profile")
            st.plotly_chart(fig_radar, use_container_width=True)

        st.write("---")

        # 3. Weekly Usage Heatmap (Visualizing Intensity)
        st.subheader("📅 Predicted Weekly Usage Intensity")
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        times = ['Morning', 'Afternoon', 'Evening', 'Night']
        
        # Simulating data: Addiction irundha red/orange cells adhigama irukkum
        intensity_base = 8 if st.session_state.get('result') == 1 else 3
        z_data = np.random.randint(intensity_base, intensity_base + 5, size=(4, 7))
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=z_data, x=days, y=times,
            colorscale='YlOrRd',
            showscale=True
        ))
        fig_heat.update_layout(title="Peak Usage Heatmap (Based on your Profile)")
        st.plotly_chart(fig_heat, use_container_width=True)

        # 4. Final Recommendations based on Score
        st.subheader("💡 AI Recommendation")
        if risk_score > 70:
            st.error("🚨 **High Priority Intervention Required:** Unga patterns digital addiction-ai indicate pannuthu. 'AI Wellness Coach' kitta 'Digital Detox Plan' kelunga.")
        else:
            st.success("✅ **Healthy Maintenance:** Unga digital usage ippodhaikku control-la dhaan irukku. Idhaiye continue panna evening time phone kurainga.")























