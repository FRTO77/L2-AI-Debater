"""
Haresha - Advanced AI Debate Platform
Main Streamlit Application
"""

import streamlit as st
import asyncio
from datetime import datetime
import json
from pathlib import Path
from typing import Optional, Dict, Any

from debate_engine import DebateEngine, DebateSession, DebateArgument
from argument_analyzer import ArgumentAnalyzer
from components import (
    render_debate_interface,
    render_analytics_dashboard,
    render_session_manager,
    render_settings_panel,
    render_export_options
)
from config import Config, load_config


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "debate_session" not in st.session_state:
        st.session_state.debate_session: Optional[DebateSession] = None
    if "debate_engine" not in st.session_state:
        st.session_state.debate_engine: Optional[DebateEngine] = None
    if "config" not in st.session_state:
        st.session_state.config = load_config()
    if "debate_history" not in st.session_state:
        st.session_state.debate_history = []
    if "analytics_data" not in st.session_state:
        st.session_state.analytics_data = {}


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Haresha - AI Debate Platform",
        page_icon="⚔️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .debate-bubble {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .ai-bubble {
        background: #f3e5f5;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
    .moderator-bubble {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff9800;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚔️ Haresha - Advanced AI Debate Platform</h1>
        <p>Multi-Agent Debate System with Real-time Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    initialize_session_state()

    # Sidebar navigation
    with st.sidebar:
        st.header("🎛️ Navigation")
        page = st.selectbox(
            "Choose Section",
            ["🏠 Debate Arena", "📊 Analytics", "💾 Session Manager", "⚙️ Settings", "📤 Export"]
        )

    # Main content based on navigation
    if page == "🏠 Debate Arena":
        render_debate_arena()
    elif page == "📊 Analytics":
        render_analytics_dashboard(st.session_state.analytics_data)
    elif page == "💾 Session Manager":
        render_session_manager()
    elif page == "⚙️ Settings":
        render_settings_panel(st.session_state.config)
    elif page == "📤 Export":
        render_export_options(st.session_state.debate_session)


def render_debate_arena():
    """Render the main debate interface."""
    st.header("🏠 Debate Arena")
    
    # Initialize debate engine if not exists
    if st.session_state.debate_engine is None:
        st.session_state.debate_engine = DebateEngine(st.session_state.config)

    # Debate setup
    if st.session_state.debate_session is None:
        render_debate_setup()
    else:
        render_active_debate()


def render_debate_setup():
    """Render debate setup interface."""
    st.subheader("🎯 Setup New Debate")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Debate Configuration")
        topic = st.text_area(
            "Debate Topic",
            placeholder="Enter the main topic for debate...",
            height=100
        )
        
        debate_type = st.selectbox(
            "Debate Format",
            ["Classic (Pro vs Con)", "Multi-Perspective", "Role-Play", "Expert Panel"]
        )
        
        rounds_limit = st.slider("Number of Rounds", 3, 15, 8)
        
    with col2:
        st.markdown("### Participant Positions")
        user_position = st.text_input("Your Position", placeholder="Briefly describe your stance")
        
        pro_position = st.text_area(
            "Pro Position",
            value="Supports the main proposition",
            height=80
        )
        
        con_position = st.text_area(
            "Con Position", 
            value="Opposes the main proposition",
            height=80
        )

    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        col_a, col_b = st.columns(2)
        with col_a:
            ai_creativity = st.slider("AI Creativity", 0.1, 1.0, 0.7, 0.1)
            response_length = st.selectbox("Response Length", ["Short", "Medium", "Long"])
        with col_b:
            enable_moderator = st.checkbox("Enable AI Moderator", value=True)
            auto_save = st.checkbox("Auto-save after each round", value=True)

    if st.button("🚀 Start Debate", type="primary", use_container_width=True):
        if topic and user_position:
            # Create new debate session
            st.session_state.debate_session = DebateSession(
                topic=topic,
                user_position=user_position,
                pro_position=pro_position,
                con_position=con_position,
                debate_type=debate_type,
                rounds_limit=rounds_limit,
                config={
                    "ai_creativity": ai_creativity,
                    "response_length": response_length,
                    "enable_moderator": enable_moderator,
                    "auto_save": auto_save
                }
            )
            st.success("Debate session created! Let's begin.")
            st.rerun()
        else:
            st.error("Please provide a topic and your position.")


def render_active_debate():
    """Render active debate interface."""
    session = st.session_state.debate_session
    engine = st.session_state.debate_engine
    
    # Session info
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.info(f"**Topic:** {session.topic}")
    with col2:
        st.metric("Round", f"{session.current_round}/{session.rounds_limit}")
    with col3:
        st.metric("Arguments", len(session.arguments))

    # Progress bar
    progress = session.current_round / session.rounds_limit
    st.progress(progress, text=f"Progress: {progress:.1%}")

    # Debate controls
    st.subheader("🎮 Debate Controls")
    
    col_left, col_center, col_right = st.columns(3)
    
    with col_left:
        if st.button("🤖 PRO Turn", use_container_width=True):
            with st.spinner("PRO is thinking..."):
                response = engine.generate_pro_response(session)
                session.add_argument("pro", response)
                st.success("PRO has responded!")
                st.rerun()
    
    with col_center:
        if st.button("🤖 CON Turn", use_container_width=True):
            with st.spinner("CON is thinking..."):
                response = engine.generate_con_response(session)
                session.add_argument("con", response)
                st.success("CON has responded!")
                st.rerun()
    
    with col_right:
        if st.button("🧑‍⚖️ Moderator", use_container_width=True):
            with st.spinner("Moderator is analyzing..."):
                summary = engine.generate_moderator_summary(session)
                session.add_argument("moderator", summary)
                session.current_round += 1
                st.success("Moderator has spoken!")
                st.rerun()

    # User input
    st.subheader("💬 Your Turn")
    user_argument = st.text_area(
        "Your Argument",
        placeholder="Enter your argument or response...",
        height=120
    )
    
    col_a, col_b = st.columns([3, 1])
    with col_a:
        if st.button("💬 Send Argument", use_container_width=True):
            if user_argument.strip():
                session.add_argument("user", user_argument)
                st.success("Argument added!")
                st.rerun()
            else:
                st.warning("Please enter an argument.")
    with col_b:
        if st.button("🔄 New Debate", use_container_width=True):
            st.session_state.debate_session = None
            st.rerun()

    # Debate history
    st.subheader("📜 Debate History")
    render_debate_history(session)


def render_debate_history(session: DebateSession):
    """Render debate history with enhanced styling."""
    if not session.arguments:
        st.info("No arguments yet. Start the debate!")
        return

    for i, arg in enumerate(session.arguments):
        # Determine bubble style based on speaker
        if arg.speaker == "user":
            bubble_class = "debate-bubble"
            icon = "👤"
        elif arg.speaker == "pro":
            bubble_class = "ai-bubble"
            icon = "🤖"
        elif arg.speaker == "con":
            bubble_class = "ai-bubble"
            icon = "🤖"
        else:  # moderator
            bubble_class = "moderator-bubble"
            icon = "🧑‍⚖️"

        # Render argument
        st.markdown(f"""
        <div class="{bubble_class}">
            <strong>{icon} {arg.speaker.upper()}</strong><br>
            {arg.content}<br>
            <small>📊 Strength: {arg.strength:.2f} | 🏷️ Type: {arg.argument_type} | ⏰ {arg.timestamp.strftime('%H:%M:%S')}</small>
        </div>
        """, unsafe_allow_html=True)

        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

