"""
Haresha - Advanced AI Debate Platform
Configuration Management Module
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class Config:
    """Configuration class for Haresha debate platform."""
    
    # LLM Settings
    llm_provider: str = "OpenAI"
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3:8b-instruct"
    ai_creativity: float = 0.7
    max_tokens: int = 500
    
    # Debate Settings
    default_rounds: int = 8
    enable_moderator: bool = True
    auto_save: bool = True
    real_time_analysis: bool = True
    
    # UI Settings
    theme: str = "Light"
    animation_speed: float = 0.05
    show_metrics: bool = True
    show_fallacies: bool = True
    
    # Analytics Settings
    enable_analytics: bool = True
    save_analytics: bool = True
    analytics_retention_days: int = 30
    
    # Export Settings
    default_export_format: str = "JSON"
    include_analytics_in_export: bool = True
    auto_export_on_completion: bool = False
    
    # Session Settings
    session_directory: str = "debate_sessions"
    max_sessions_per_user: int = 100
    session_retention_days: int = 90
    
    # Advanced Settings
    enable_logging: bool = True
    log_level: str = "INFO"
    debug_mode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(**data)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or create default."""
    if config_path is None:
        config_path = "config.json"
    
    try:
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Config.from_dict(data)
        else:
            # Create default config
            config = Config()
            save_config(config, config_path)
            return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return Config()


def save_config(config: Config, config_path: Optional[str] = None) -> bool:
    """Save configuration to file."""
    if config_path is None:
        config_path = "config.json"
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def load_env_config() -> Dict[str, str]:
    """Load configuration from environment variables."""
    env_config = {}
    
    # LLM Settings
    env_config["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    env_config["OLLAMA_BASE_URL"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    env_config["OLLAMA_MODEL"] = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct")
    
    # Application Settings
    env_config["SESSION_DIRECTORY"] = os.getenv("SESSION_DIRECTORY", "debate_sessions")
    env_config["LOG_LEVEL"] = os.getenv("LOG_LEVEL", "INFO")
    env_config["DEBUG_MODE"] = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    return env_config


def validate_config(config: Config) -> Dict[str, Any]:
    """Validate configuration and return validation results."""
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Validate LLM settings
    if config.llm_provider not in ["OpenAI", "Ollama"]:
        validation_results["errors"].append("Invalid LLM provider")
        validation_results["valid"] = False
    
    if config.ai_creativity < 0.0 or config.ai_creativity > 1.0:
        validation_results["errors"].append("AI creativity must be between 0.0 and 1.0")
        validation_results["valid"] = False
    
    if config.max_tokens < 50 or config.max_tokens > 2000:
        validation_results["warnings"].append("Max tokens should be between 50 and 2000")
    
    # Validate debate settings
    if config.default_rounds < 1 or config.default_rounds > 20:
        validation_results["errors"].append("Default rounds must be between 1 and 20")
        validation_results["valid"] = False
    
    # Validate UI settings
    if config.theme not in ["Light", "Dark", "Auto"]:
        validation_results["errors"].append("Invalid theme setting")
        validation_results["valid"] = False
    
    if config.animation_speed < 0.01 or config.animation_speed > 0.5:
        validation_results["warnings"].append("Animation speed should be between 0.01 and 0.5")
    
    # Validate session settings
    if config.max_sessions_per_user < 1:
        validation_results["errors"].append("Max sessions per user must be at least 1")
        validation_results["valid"] = False
    
    if config.session_retention_days < 1:
        validation_results["warnings"].append("Session retention should be at least 1 day")
    
    return validation_results


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
    """Merge base config with override values."""
    base_dict = base_config.to_dict()
    base_dict.update(override_config)
    return Config.from_dict(base_dict)


def create_config_template() -> str:
    """Create a configuration template file."""
    template = {
        "llm_provider": "OpenAI",
        "openai_model": "gpt-4o-mini",
        "ollama_model": "llama3:8b-instruct",
        "ai_creativity": 0.7,
        "max_tokens": 500,
        "default_rounds": 8,
        "enable_moderator": True,
        "auto_save": True,
        "real_time_analysis": True,
        "theme": "Light",
        "animation_speed": 0.05,
        "show_metrics": True,
        "show_fallacies": True,
        "enable_analytics": True,
        "save_analytics": True,
        "analytics_retention_days": 30,
        "default_export_format": "JSON",
        "include_analytics_in_export": True,
        "auto_export_on_completion": False,
        "session_directory": "debate_sessions",
        "max_sessions_per_user": 100,
        "session_retention_days": 90,
        "enable_logging": True,
        "log_level": "INFO",
        "debug_mode": False
    }
    
    return json.dumps(template, indent=2, ensure_ascii=False)


def export_config(config: Config, file_path: str) -> bool:
    """Export configuration to a file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error exporting config: {e}")
        return False


def import_config(file_path: str) -> Optional[Config]:
    """Import configuration from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Config.from_dict(data)
    except Exception as e:
        print(f"Error importing config: {e}")
        return None


def reset_to_defaults(config_path: str = "config.json") -> bool:
    """Reset configuration to default values."""
    try:
        default_config = get_default_config()
        return save_config(default_config, config_path)
    except Exception as e:
        print(f"Error resetting config: {e}")
        return False


# Configuration presets
def get_preset_configs() -> Dict[str, Config]:
    """Get predefined configuration presets."""
    presets = {
        "default": get_default_config(),
        
        "performance": Config(
            llm_provider="OpenAI",
            openai_model="gpt-4o-mini",
            ai_creativity=0.5,
            max_tokens=300,
            default_rounds=6,
            animation_speed=0.02,
            enable_analytics=False,
            auto_save=False
        ),
        
        "quality": Config(
            llm_provider="OpenAI",
            openai_model="gpt-4o",
            ai_creativity=0.8,
            max_tokens=800,
            default_rounds=12,
            enable_moderator=True,
            real_time_analysis=True,
            show_metrics=True,
            show_fallacies=True
        ),
        
        "local": Config(
            llm_provider="Ollama",
            ollama_model="llama3:8b-instruct",
            ai_creativity=0.6,
            max_tokens=400,
            default_rounds=8,
            enable_analytics=True,
            auto_save=True
        ),
        
        "debug": Config(
            llm_provider="OpenAI",
            openai_model="gpt-3.5-turbo",
            ai_creativity=0.3,
            max_tokens=200,
            default_rounds=4,
            debug_mode=True,
            log_level="DEBUG",
            enable_logging=True
        )
    }
    
    return presets


def apply_preset(preset_name: str, config_path: str = "config.json") -> bool:
    """Apply a configuration preset."""
    presets = get_preset_configs()
    
    if preset_name not in presets:
        print(f"Unknown preset: {preset_name}")
        return False
    
    preset_config = presets[preset_name]
    return save_config(preset_config, config_path)


def list_available_presets() -> List[str]:
    """List available configuration presets."""
    return list(get_preset_configs().keys())

