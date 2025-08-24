"""
Haresha - Advanced AI Debate Platform
UI Components Module (Fixed Version)
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_debate_interface():
    """Render the debate interface - placeholder function."""
    st.info("Debate interface will be rendered here.")


def render_analytics_dashboard(analytics_data: Dict[str, Any]):
    """Render comprehensive analytics dashboard."""
    st.header("📊 Analytics Dashboard")
    
    if not analytics_data:
        st.info("No debate data available. Start a debate to see analytics.")
        return
    
    # Overview metrics
    st.subheader("📈 Overview Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Arguments", analytics_data.get("total_arguments", 0))
    with col2:
        st.metric("Average Strength", f"{analytics_data.get('avg_strength', 0):.2f}")
    with col3:
        st.metric("Debate Duration", analytics_data.get("duration", "N/A"))
    with col4:
        st.metric("Fallacy Count", analytics_data.get("total_fallacies", 0))
    
    # Charts section
    st.subheader("�� Performance Charts")
    
    # Argument strength over time
    if "strength_timeline" in analytics_data:
        fig_strength = px.line(
            analytics_data["strength_timeline"],
            x="timestamp",
            y="strength",
            title="Argument Strength Over Time",
            labels={"strength": "Strength Score", "timestamp": "Time"}
        )
        st.plotly_chart(fig_strength, use_container_width=True)
    
    # Speaker performance comparison
    if "speaker_stats" in analytics_data:
        speaker_data = analytics_data["speaker_stats"]
        fig_speakers = px.bar(
            x=list(speaker_data.keys()),
            y=[data["avg_strength"] for data in speaker_data.values()],
            title="Average Argument Strength by Speaker",
            labels={"x": "Speaker", "y": "Average Strength"}
        )
        st.plotly_chart(fig_speakers, use_container_width=True)
    
    # Argument type distribution
    if "argument_types" in analytics_data:
        type_data = analytics_data["argument_types"]
        fig_types = px.pie(
            values=list(type_data.values()),
            names=list(type_data.keys()),
            title="Argument Type Distribution"
        )
        st.plotly_chart(fig_types, use_container_width=True)
    
    # Sentiment analysis
    if "sentiment_data" in analytics_data:
        sentiment_data = analytics_data["sentiment_data"]
        fig_sentiment = px.scatter(
            sentiment_data,
            x="timestamp",
            y="sentiment",
            color="speaker",
            title="Sentiment Analysis Over Time",
            labels={"sentiment": "Sentiment Score", "timestamp": "Time"}
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Detailed analysis
    st.subheader("🔍 Detailed Analysis")
    
    # Fallacy detection
    if "fallacies" in analytics_data:
        st.write("**Logical Fallacies Detected:**")
        for fallacy in analytics_data["fallacies"]:
            st.warning(f"• {fallacy['type']}: {fallacy['description']}")
    
    # Recommendations
    if "recommendations" in analytics_data:
        st.write("**Improvement Recommendations:**")
        for rec in analytics_data["recommendations"]:
            st.info(f"• {rec}")


def render_session_manager():
    """Render session management interface."""
    st.header("�� Session Manager")
    
    # Session directory
    session_dir = st.text_input(
        "Session Directory",
        value=str(Path.cwd() / "debate_sessions"),
        help="Directory where debate sessions are saved"
    )
    
    # Load existing sessions
    if st.button("🔄 Refresh Sessions"):
        sessions = list_sessions(session_dir)
        st.session_state.available_sessions = sessions
    
    # Display available sessions
    if "available_sessions" in st.session_state:
        st.subheader("�� Available Sessions")
        
        if not st.session_state.available_sessions:
            st.info("No saved sessions found.")
            return
        
        # Session selection
        selected_session = st.selectbox(
            "Select Session to Load",
            options=st.session_state.available_sessions,
            format_func=lambda x: Path(x).stem
        )
        
        if selected_session:
            # Load session info
            session_info = get_session_info(selected_session)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Session Details:**")
                st.write(f"• Topic: {session_info.get('topic', 'N/A')}")
                st.write(f"• Created: {session_info.get('created_at', 'N/A')}")
                st.write(f"• Arguments: {session_info.get('argument_count', 0)}")
                st.write(f"• Rounds: {session_info.get('current_round', 0)}")
            
            with col2:
                if st.button("�� Load Session", use_container_width=True):
                    load_session(selected_session)
                    st.success("Session loaded successfully!")
                    st.rerun()
                
                if st.button("🗑️ Delete Session", use_container_width=True):
                    delete_session(selected_session)
                    st.success("Session deleted!")
                    st.rerun()
    
    # Export all sessions
    st.subheader("📤 Bulk Export")
    if st.button("📦 Export All Sessions"):
        export_all_sessions(session_dir)


def _get_config_value(config, key: str, default=None):
    """Safely get config value whether it's a dict or object."""
    if hasattr(config, key):
        return getattr(config, key)
    elif isinstance(config, dict):
        return config.get(key, default)
    return default


def render_settings_panel(config):
    """Render settings configuration panel."""
    st.header("⚙️ Settings")
    
    # Convert config to dict for editing
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        config_dict = config
    
    # LLM Configuration
    st.subheader("🤖 Language Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_provider = config_dict.get("llm_provider", "OpenAI")
        config_dict["llm_provider"] = st.selectbox(
            "LLM Provider",
            ["OpenAI", "Ollama"],
            index=0 if current_provider == "OpenAI" else 1
        )
        
        if config_dict["llm_provider"] == "OpenAI":
            config_dict["openai_model"] = st.selectbox(
                "OpenAI Model",
                ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                index=0
            )
        else:
            config_dict["ollama_model"] = st.text_input(
                "Ollama Model",
                value=config_dict.get("ollama_model", "llama3:8b-instruct")
            )
    
    with col2:
        config_dict["ai_creativity"] = st.slider(
            "AI Creativity",
            0.1, 1.0, config_dict.get("ai_creativity", 0.7), 0.1
        )
        
        config_dict["max_tokens"] = st.number_input(
            "Max Tokens",
            min_value=50, max_value=2000, value=config_dict.get("max_tokens", 500)
        )
    
    # Debate Settings
    st.subheader("⚔️ Debate Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config_dict["default_rounds"] = st.slider(
            "Default Rounds",
            3, 15, config_dict.get("default_rounds", 8)
        )
        
        config_dict["enable_moderator"] = st.checkbox(
            "Enable AI Moderator",
            value=config_dict.get("enable_moderator", True)
        )
    
    with col2:
        config_dict["auto_save"] = st.checkbox(
            "Auto-save Sessions",
            value=config_dict.get("auto_save", True)
        )
        
        config_dict["real_time_analysis"] = st.checkbox(
            "Real-time Argument Analysis",
            value=config_dict.get("real_time_analysis", True)
        )
    
    # UI Settings
    st.subheader("🎨 UI Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config_dict["theme"] = st.selectbox(
            "Theme",
            ["Light", "Dark", "Auto"],
            index=0
        )
        
        config_dict["animation_speed"] = st.slider(
            "Animation Speed",
            0.01, 0.2, config_dict.get("animation_speed", 0.05), 0.01
        )
    
    with col2:
        config_dict["show_metrics"] = st.checkbox(
            "Show Argument Metrics",
            value=config_dict.get("show_metrics", True)
        )
        
        config_dict["show_fallacies"] = st.checkbox(
            "Show Fallacy Warnings",
            value=config_dict.get("show_fallacies", True)
        )
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        # Convert back to Config object if needed
        if hasattr(config, 'from_dict'):
            new_config = config.from_dict(config_dict)
        else:
            new_config = config_dict
        
        save_config(new_config)
        st.success("Settings saved successfully!")
        
        # Update session state
        if hasattr(st.session_state, 'config'):
            st.session_state.config = new_config


def render_export_options(session):
    """Render export options for debate sessions."""
    st.header("📤 Export Options")
    
    if not session:
        st.info("No active session to export. Start a debate first.")
        return
    
    # Export formats
    st.subheader("📄 Export Formats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON Export
        json_data = session.to_dict()
        st.download_button(
            "📄 Export as JSON",
            data=json.dumps(json_data, indent=2, ensure_ascii=False),
            file_name=f"debate_{session.session_id}.json",
            mime="application/json"
        )
        
        # Markdown Export
        md_content = generate_markdown_export(session)
        st.download_button(
            "�� Export as Markdown",
            data=md_content,
            file_name=f"debate_{session.session_id}.md",
            mime="text/markdown"
        )
    
    with col2:
        # CSV Export
        csv_data = generate_csv_export(session)
        st.download_button(
            "📊 Export as CSV",
            data=csv_data,
            file_name=f"debate_{session.session_id}.csv",
            mime="text/csv"
        )
        
        # PDF Export (placeholder)
        st.button("📋 Export as PDF", disabled=True, help="PDF export coming soon!")
    
    # Custom export options
    st.subheader("🔧 Custom Export")
    
    export_options = st.multiselect(
        "Select Export Components",
        ["Arguments", "Analytics", "Timeline", "Summary", "Recommendations"],
        default=["Arguments", "Analytics", "Summary"]
    )
    
    if st.button("🎯 Generate Custom Export"):
        custom_export = generate_custom_export(session, export_options)
        st.download_button(
            "📦 Download Custom Export",
            data=custom_export,
            file_name=f"debate_custom_{session.session_id}.json",
            mime="application/json"
        )


# Helper functions

def list_sessions(directory: str) -> List[str]:
    """List all saved debate sessions."""
    try:
        path = Path(directory)
        if not path.exists():
            return []
        return [str(p) for p in path.glob("*.json")]
    except Exception:
        return []


def get_session_info(file_path: str) -> Dict[str, Any]:
    """Get basic information about a saved session."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            "topic": data.get("topic", "Unknown"),
            "created_at": data.get("created_at", "Unknown"),
            "argument_count": len(data.get("arguments", [])),
            "current_round": data.get("current_round", 0),
            "debate_type": data.get("debate_type", "Unknown")
        }
    except Exception:
        return {}


def load_session(file_path: str):
    """Load a saved debate session."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        from debate_engine import DebateSession
        st.session_state.debate_session = DebateSession.from_dict(data)
        st.session_state.debate_engine = None  # Will be reinitialized
    except Exception as e:
        st.error(f"Error loading session: {str(e)}")


def delete_session(file_path: str):
    """Delete a saved debate session."""
    try:
        Path(file_path).unlink()
        if "available_sessions" in st.session_state:
            st.session_state.available_sessions.remove(file_path)
    except Exception as e:
        st.error(f"Error deleting session: {str(e)}")


def save_config(config):
    """Save configuration to file."""
    try:
        config_path = Path("config.json")
        
        # Convert to dict if it's a Config object
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = config
            
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving config: {str(e)}")


def generate_markdown_export(session) -> str:
    """Generate markdown export of debate session."""
    lines = []
    lines.append(f"# Debate: {session.topic}")
    lines.append("")
    lines.append(f"**Session ID:** {session.session_id}")
    lines.append(f"**Created:** {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Type:** {session.debate_type}")
    lines.append(f"**Rounds:** {session.current_round}/{session.rounds_limit}")
    lines.append("")
    
    lines.append("## Positions")
    lines.append(f"- **User:** {session.user_position}")
    lines.append(f"- **Pro:** {session.pro_position}")
    lines.append(f"- **Con:** {session.con_position}")
    lines.append("")
    
    lines.append("## Arguments")
    for i, arg in enumerate(session.arguments, 1):
        lines.append(f"### {i}. {arg.speaker.upper()}")
        lines.append(f"**Type:** {arg.argument_type}")
        lines.append(f"**Strength:** {arg.strength:.2f}")
        lines.append(f"**Time:** {arg.timestamp.strftime('%H:%M:%S')}")
        lines.append("")
        lines.append(arg.content)
        lines.append("")
    
    return "\n".join(lines)


def generate_csv_export(session) -> str:
    """Generate CSV export of debate arguments."""
    import io
    import csv
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(["Speaker", "Content", "Type", "Strength", "Timestamp"])
    
    # Data
    for arg in session.arguments:
        writer.writerow([
            arg.speaker,
            arg.content,
            arg.argument_type,
            f"{arg.strength:.2f}",
            arg.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        ])
    
    return output.getvalue()


def generate_custom_export(session, options: List[str]) -> str:
    """Generate custom export based on selected options."""
    export_data = {
        "session_info": {
            "topic": session.topic,
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "debate_type": session.debate_type
        }
    }
    
    if "Arguments" in options:
        export_data["arguments"] = [
            {
                "speaker": arg.speaker,
                "content": arg.content,
                "type": arg.argument_type,
                "strength": arg.strength,
                "timestamp": arg.timestamp.isoformat()
            }
            for arg in session.arguments
        ]
    
    if "Analytics" in options:
        # Add analytics data
        export_data["analytics"] = {
            "total_arguments": len(session.arguments),
            "avg_strength": sum(arg.strength for arg in session.arguments) / max(1, len(session.arguments)),
            "speaker_distribution": {}
        }
        
        for arg in session.arguments:
            if arg.speaker not in export_data["analytics"]["speaker_distribution"]:
                export_data["analytics"]["speaker_distribution"][arg.speaker] = 0
            export_data["analytics"]["speaker_distribution"][arg.speaker] += 1
    
    if "Summary" in options:
        export_data["summary"] = {
            "rounds_completed": session.current_round,
            "total_rounds": session.rounds_limit,
            "completion_percentage": (session.current_round / session.rounds_limit) * 100
        }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def export_all_sessions(directory: str):
    """Export all saved sessions as a single file."""
    sessions = list_sessions(directory)
    if not sessions:
        st.warning("No sessions to export.")
        return
    
    all_sessions = []
    for session_path in sessions:
        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                session_data["file_path"] = session_path
                all_sessions.append(session_data)
        except Exception:
            continue
    
    export_data = {
        "export_date": datetime.now().isoformat(),
        "total_sessions": len(all_sessions),
        "sessions": all_sessions
    }
    
    st.download_button(
        "📦 Download All Sessions",
        data=json.dumps(export_data, indent=2, ensure_ascii=False),
        file_name=f"all_debate_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )