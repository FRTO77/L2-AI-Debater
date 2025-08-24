"""
Haresha - Advanced AI Debate Platform
Core Debate Engine (Fixed Version)
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import json
from pathlib import Path

import streamlit as st


@dataclass
class DebateArgument:
    """Represents a single argument in the debate."""
    speaker: str  # "user", "pro", "con", "moderator"
    content: str
    argument_type: str  # "claim", "evidence", "counter", "rebuttal", "summary"
    strength: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateSession:
    """Represents a complete debate session."""
    topic: str
    user_position: str
    pro_position: str
    con_position: str
    debate_type: str
    rounds_limit: int
    config: Union[Dict[str, Any], Any]  # Can be dict or Config object
    arguments: List[DebateArgument] = field(default_factory=list)
    current_round: int = 1
    session_id: str = field(default_factory=lambda: f"debate_{int(time.time())}")
    created_at: datetime = field(default_factory=datetime.now)

    def add_argument(self, speaker: str, content: str, argument_type: str = None, strength: float = None):
        """Add a new argument to the session."""
        if argument_type is None:
            argument_type = self._classify_argument_type(content)
        if strength is None:
            strength = self._analyze_argument_strength(content)
        
        argument = DebateArgument(
            speaker=speaker,
            content=content,
            argument_type=argument_type,
            strength=strength
        )
        self.arguments.append(argument)

    def _classify_argument_type(self, text: str) -> str:
        """Classify the type of argument."""
        text_lower = text.lower()
        
        # English keywords
        if any(word in text_lower for word in ["however", "but", "nevertheless", "on the other hand"]):
            return "counter"
        elif any(word in text_lower for word in ["because", "since", "as", "for example", "evidence shows"]):
            return "evidence"
        elif any(word in text_lower for word in ["wrong", "incorrect", "false", "disagree", "not true"]):
            return "rebuttal"
        else:
            return "claim"

    def _analyze_argument_strength(self, text: str) -> float:
        """Analyze the strength of an argument."""
        text_lower = text.lower()
        score = 0.5  # Base score
        
        # Strong indicators
        strong_indicators = [
            "because", "therefore", "thus", "consequently", "evidence", "research shows",
            "statistics", "studies", "proven", "logical", "obvious", "clear"
        ]
        
        # Weak indicators
        weak_indicators = [
            "maybe", "perhaps", "possibly", "I think", "I believe", "not sure",
            "don't know", "uncertain", "might be"
        ]
        
        for indicator in strong_indicators:
            if indicator in text_lower:
                score += 0.08
        
        for indicator in weak_indicators:
            if indicator in text_lower:
                score -= 0.08
        
        return max(0.0, min(1.0, score))

    def get_last_argument(self, speaker: str = None) -> Optional[DebateArgument]:
        """Get the last argument, optionally filtered by speaker."""
        if speaker:
            for arg in reversed(self.arguments):
                if arg.speaker == speaker:
                    return arg
        return self.arguments[-1] if self.arguments else None

    def get_speaker_arguments(self, speaker: str) -> List[DebateArgument]:
        """Get all arguments from a specific speaker."""
        return [arg for arg in self.arguments if arg.speaker == speaker]

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "topic": self.topic,
            "user_position": self.user_position,
            "pro_position": self.pro_position,
            "con_position": self.con_position,
            "debate_type": self.debate_type,
            "rounds_limit": self.rounds_limit,
            "config": self.config,
            "arguments": [
                {
                    "speaker": arg.speaker,
                    "content": arg.content,
                    "argument_type": arg.argument_type,
                    "strength": arg.strength,
                    "timestamp": arg.timestamp.isoformat(),
                    "metadata": arg.metadata
                }
                for arg in self.arguments
            ],
            "current_round": self.current_round,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DebateSession':
        """Create session from dictionary."""
        session = cls(
            topic=data["topic"],
            user_position=data["user_position"],
            pro_position=data["pro_position"],
            con_position=data["con_position"],
            debate_type=data["debate_type"],
            rounds_limit=data["rounds_limit"],
            config=data["config"]
        )
        
        session.current_round = data.get("current_round", 1)
        session.session_id = data.get("session_id", f"debate_{int(time.time())}")
        session.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        
        # Load arguments
        for arg_data in data.get("arguments", []):
            argument = DebateArgument(
                speaker=arg_data["speaker"],
                content=arg_data["content"],
                argument_type=arg_data["argument_type"],
                strength=arg_data["strength"],
                timestamp=datetime.fromisoformat(arg_data["timestamp"]),
                metadata=arg_data.get("metadata", {})
            )
            session.arguments.append(argument)
        
        return session


class DebateEngine:
    """Core debate engine that manages AI interactions."""
    
    def __init__(self, config):
        self.config = config
        self.llm = self._initialize_llm()
        self.analyzer = self._initialize_analyzer()

    def _get_config_value(self, key: str, default=None):
        """Safely get config value whether it's a dict or object."""
        if hasattr(self.config, key):
            return getattr(self.config, key)
        elif isinstance(self.config, dict):
            return self.config.get(key, default)
        return default

    def _initialize_llm(self):
        """Initialize the language model."""
        provider = self._get_config_value("llm_provider", "OpenAI")
        
        if provider == "OpenAI":
            try:
                from langchain_openai import ChatOpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    st.error("OPENAI_API_KEY not found. Please set it in your environment.")
                    return None
                return ChatOpenAI(
                    model=self._get_config_value("openai_model", "gpt-4o-mini"),
                    temperature=self._get_config_value("ai_creativity", 0.7),
                    api_key=api_key
                )
            except ImportError:
                st.error("langchain-openai not installed. Please install it.")
                return None
        
        elif provider == "Ollama":
            try:
                from langchain_ollama import ChatOllama
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                return ChatOllama(
                    model=self._get_config_value("ollama_model", "llama3:8b-instruct"),
                    base_url=base_url,
                    temperature=self._get_config_value("ai_creativity", 0.7)
                )
            except ImportError:
                st.error("langchain-ollama not installed. Please install it.")
                return None
        
        return None

    def _initialize_analyzer(self):
        """Initialize the argument analyzer."""
        try:
            from argument_analyzer import ArgumentAnalyzer
            return ArgumentAnalyzer()
        except ImportError:
            return None

    def generate_pro_response(self, session: DebateSession) -> str:
        """Generate a response from the PRO side."""
        if not self.llm:
            return "Error: Language model not initialized."
        
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            # Get context from recent arguments
            recent_args = session.arguments[-6:] if len(session.arguments) > 6 else session.arguments
            context = "\n".join([f"[{arg.speaker}]: {arg.content}" for arg in recent_args])
            
            # Get last opponent argument
            last_con = session.get_last_argument("con")
            last_user = session.get_last_argument("user")
            opponent_last = (last_con.content if last_con else "") or (last_user.content if last_user else "")
            
            prompt = PromptTemplate(
                input_variables=["topic", "pro_position", "context", "opponent_last", "round"],
                template="""
You are an expert debater representing the PRO side in a formal debate.

TOPIC: {topic}
YOUR POSITION: {pro_position}
CURRENT ROUND: {round}

RECENT CONTEXT:
{context}

LAST OPPONENT ARGUMENT: {opponent_last}

Your task is to:
1. Address the opponent's argument directly
2. Present strong evidence and logical reasoning
3. Support your position with facts and examples
4. Be persuasive but respectful
5. Keep your response concise (2-4 sentences)

Your response:"""
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.invoke({
                "topic": session.topic,
                "pro_position": session.pro_position,
                "context": context,
                "opponent_last": opponent_last,
                "round": session.current_round
            })
            
            return response.get("text", "No response generated.")
            
        except Exception as e:
            return f"Error generating PRO response: {str(e)}"

    def generate_con_response(self, session: DebateSession) -> str:
        """Generate a response from the CON side."""
        if not self.llm:
            return "Error: Language model not initialized."
        
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            # Get context from recent arguments
            recent_args = session.arguments[-6:] if len(session.arguments) > 6 else session.arguments
            context = "\n".join([f"[{arg.speaker}]: {arg.content}" for arg in recent_args])
            
            # Get last opponent argument
            last_pro = session.get_last_argument("pro")
            last_user = session.get_last_argument("user")
            opponent_last = (last_pro.content if last_pro else "") or (last_user.content if last_user else "")
            
            prompt = PromptTemplate(
                input_variables=["topic", "con_position", "context", "opponent_last", "round"],
                template="""
You are an expert debater representing the CON side in a formal debate.

TOPIC: {topic}
YOUR POSITION: {con_position}
CURRENT ROUND: {round}

RECENT CONTEXT:
{context}

LAST OPPONENT ARGUMENT: {opponent_last}

Your task is to:
1. Address the opponent's argument directly
2. Present strong evidence and logical reasoning
3. Support your position with facts and examples
4. Be persuasive but respectful
5. Keep your response concise (2-4 sentences)

Your response:"""
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.invoke({
                "topic": session.topic,
                "con_position": session.con_position,
                "context": context,
                "opponent_last": opponent_last,
                "round": session.current_round
            })
            
            return response.get("text", "No response generated.")
            
        except Exception as e:
            return f"Error generating CON response: {str(e)}"

    def generate_moderator_summary(self, session: DebateSession) -> str:
        """Generate a moderator summary of the current round."""
        if not self.llm:
            return "Error: Language model not initialized."
        
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            # Get recent arguments for this round
            recent_args = session.arguments[-4:] if len(session.arguments) >= 4 else session.arguments
            round_context = "\n".join([f"[{arg.speaker}]: {arg.content}" for arg in recent_args])
            
            prompt = PromptTemplate(
                input_variables=["topic", "round", "rounds_limit", "round_context"],
                template="""
You are an impartial debate moderator. Provide a brief summary of Round {round}/{rounds_limit}.

TOPIC: {topic}

ROUND CONTEXT:
{round_context}

Your task is to:
1. Briefly summarize the key points made by each side
2. Identify the strongest arguments presented
3. Note any logical fallacies or weak points
4. Provide 1-2 sentences of constructive feedback
5. Keep your summary concise and objective

Moderator summary:"""
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.invoke({
                "topic": session.topic,
                "round": session.current_round,
                "rounds_limit": session.rounds_limit,
                "round_context": round_context
            })
            
            return response.get("text", "No moderator summary generated.")
            
        except Exception as e:
            return f"Error generating moderator summary: {str(e)}"

    def generate_final_verdict(self, session: DebateSession) -> str:
        """Generate a final verdict for the debate."""
        if not self.llm:
            return "Error: Language model not initialized."
        
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            # Get all arguments
            all_arguments = "\n".join([f"[{arg.speaker}]: {arg.content}" for arg in session.arguments])
            
            prompt = PromptTemplate(
                input_variables=["topic", "user_position", "pro_position", "con_position", "all_arguments"],
                template="""
You are an impartial debate judge providing a final verdict.

TOPIC: {topic}
USER POSITION: {user_position}
PRO POSITION: {pro_position}
CON POSITION: {con_position}

COMPLETE DEBATE:
{all_arguments}

Your task is to:
1. Evaluate the overall strength of each side's arguments
2. Consider evidence, logic, and persuasiveness
3. Identify the most compelling points made
4. Provide a balanced assessment
5. Give a clear verdict on which side was more convincing
6. Keep your verdict concise but thorough (3-5 sentences)

Final verdict:"""
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.invoke({
                "topic": session.topic,
                "user_position": session.user_position,
                "pro_position": session.pro_position,
                "con_position": session.con_position,
                "all_arguments": all_arguments
            })
            
            return response.get("text", "No final verdict generated.")
            
        except Exception as e:
            return f"Error generating final verdict: {str(e)}"