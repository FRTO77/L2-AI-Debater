"""
Haresha - Advanced AI Debate Platform
Argument Analysis Module
"""

import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ArgumentMetrics:
    """Comprehensive metrics for argument analysis."""
    strength: float  # 0.0 to 1.0
    clarity: float  # 0.0 to 1.0
    persuasiveness: float  # 0.0 to 1.0
    evidence_quality: float  # 0.0 to 1.0
    logical_consistency: float  # 0.0 to 1.0
    sentiment_score: float  # -1.0 to 1.0
    fallacy_count: int
    detected_fallacies: List[str]
    word_count: int
    complexity_score: float  # 0.0 to 1.0


class ArgumentAnalyzer:
    """Advanced argument analysis with multiple metrics."""
    
    def __init__(self):
        self.strong_indicators = [
            "because", "therefore", "thus", "consequently", "evidence", "research shows",
            "statistics", "studies", "proven", "logical", "obvious", "clear", "demonstrates",
            "indicates", "suggests", "confirms", "validates", "supports", "establishes"
        ]
        
        self.weak_indicators = [
            "maybe", "perhaps", "possibly", "I think", "I believe", "not sure",
            "don't know", "uncertain", "might be", "could be", "seems like",
            "appears to", "sort of", "kind of", "basically"
        ]
        
        self.evidence_indicators = [
            "study", "research", "data", "statistics", "survey", "experiment",
            "analysis", "report", "finding", "result", "evidence", "proof",
            "demonstration", "example", "case", "instance"
        ]
        
        self.logical_fallacies = {
            "ad_hominem": [
                r"you're (stupid|dumb|ignorant|wrong)",
                r"only (stupid|dumb|ignorant) people",
                r"you don't understand",
                r"you're biased"
            ],
            "straw_man": [
                r"so you're saying",
                r"if that's the case",
                r"that would mean",
                r"you're arguing that"
            ],
            "appeal_to_authority": [
                r"experts say",
                r"scientists agree",
                r"everyone knows",
                r"it's common knowledge"
            ],
            "false_dichotomy": [
                r"either.*or",
                r"you're either.*or",
                r"there are only two options",
                r"it's either.*or nothing"
            ],
            "slippery_slope": [
                r"if we.*then.*will.*",
                r"this will lead to",
                r"next thing you know",
                r"before long"
            ],
            "appeal_to_emotion": [
                r"think of the children",
                r"how would you feel",
                r"imagine if",
                r"it's heartbreaking"
            ]
        }

    def analyze_argument(self, text: str) -> ArgumentMetrics:
        """Comprehensive argument analysis."""
        text_lower = text.lower()
        
        # Basic metrics
        strength = self._calculate_strength(text_lower)
        clarity = self._calculate_clarity(text)
        persuasiveness = self._calculate_persuasiveness(text_lower)
        evidence_quality = self._calculate_evidence_quality(text_lower)
        logical_consistency = self._calculate_logical_consistency(text)
        sentiment_score = self._calculate_sentiment(text_lower)
        
        # Fallacy detection
        fallacies = self._detect_fallacies(text_lower)
        fallacy_count = len(fallacies)
        
        # Additional metrics
        word_count = len(text.split())
        complexity_score = self._calculate_complexity(text)
        
        return ArgumentMetrics(
            strength=strength,
            clarity=clarity,
            persuasiveness=persuasiveness,
            evidence_quality=evidence_quality,
            logical_consistency=logical_consistency,
            sentiment_score=sentiment_score,
            fallacy_count=fallacy_count,
            detected_fallacies=fallacies,
            word_count=word_count,
            complexity_score=complexity_score
        )

    def _calculate_strength(self, text: str) -> float:
        """Calculate argument strength based on indicators."""
        score = 0.5  # Base score
        
        # Add points for strong indicators
        for indicator in self.strong_indicators:
            if indicator in text:
                score += 0.05
        
        # Subtract points for weak indicators
        for indicator in self.weak_indicators:
            if indicator in text:
                score -= 0.05
        
        return max(0.0, min(1.0, score))

    def _calculate_clarity(self, text: str) -> float:
        """Calculate text clarity."""
        sentences = text.split('.')
        if not sentences:
            return 0.0
        
        # Average sentence length (shorter is better for clarity)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Penalize very long or very short sentences
        if avg_sentence_length < 5:
            return 0.3
        elif avg_sentence_length > 25:
            return 0.4
        elif 8 <= avg_sentence_length <= 15:
            return 0.9
        else:
            return 0.7

    def _calculate_persuasiveness(self, text: str) -> float:
        """Calculate persuasiveness score."""
        score = 0.5
        
        # Rhetorical devices
        rhetorical_patterns = [
            r"not only.*but also",
            r"on the one hand.*on the other hand",
            r"in contrast",
            r"however",
            r"nevertheless",
            r"furthermore",
            r"moreover",
            r"additionally"
        ]
        
        for pattern in rhetorical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1
        
        # Question patterns (can be persuasive)
        question_count = text.count('?')
        if question_count > 0:
            score += min(0.2, question_count * 0.05)
        
        return max(0.0, min(1.0, score))

    def _calculate_evidence_quality(self, text: str) -> float:
        """Calculate evidence quality score."""
        score = 0.0
        
        # Count evidence indicators
        evidence_count = sum(1 for indicator in self.evidence_indicators if indicator in text)
        
        if evidence_count == 0:
            score = 0.2
        elif evidence_count == 1:
            score = 0.5
        elif evidence_count == 2:
            score = 0.7
        else:
            score = 0.9
        
        # Bonus for specific numbers/statistics
        if re.search(r'\d+%|\d+ percent|\d+ out of \d+', text):
            score += 0.1
        
        return max(0.0, min(1.0, score))

    def _calculate_logical_consistency(self, text: str) -> float:
        """Calculate logical consistency score."""
        score = 1.0
        
        # Check for contradictions
        contradictions = [
            (r"always", r"never"),
            (r"all", r"none"),
            (r"everyone", r"no one"),
            (r"impossible", r"possible")
        ]
        
        for pos, neg in contradictions:
            if re.search(pos, text, re.IGNORECASE) and re.search(neg, text, re.IGNORECASE):
                score -= 0.3
        
        # Check for logical connectors
        logical_connectors = ["because", "therefore", "thus", "consequently", "as a result"]
        connector_count = sum(1 for connector in logical_connectors if connector in text)
        
        if connector_count > 0:
            score += min(0.2, connector_count * 0.05)
        
        return max(0.0, min(1.0, score))

    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score (-1.0 to 1.0)."""
        positive_words = [
            "good", "great", "excellent", "amazing", "wonderful", "beneficial",
            "positive", "advantageous", "effective", "successful", "proven"
        ]
        
        negative_words = [
            "bad", "terrible", "awful", "horrible", "harmful", "negative",
            "dangerous", "risky", "problematic", "concerning", "worrisome"
        ]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Normalize by text length
        sentiment = (positive_count - negative_count) / max(1, total_words / 10)
        return max(-1.0, min(1.0, sentiment))

    def _detect_fallacies(self, text: str) -> List[str]:
        """Detect logical fallacies in the text."""
        detected_fallacies = []
        
        for fallacy_name, patterns in self.logical_fallacies.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    detected_fallacies.append(fallacy_name)
                    break  # Only count each fallacy once
        
        return detected_fallacies

    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        words = text.split()
        if not words:
            return 0.0
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Sentence complexity
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        
        # Complexity score based on word and sentence length
        complexity = (avg_word_length / 10) + (avg_sentence_length / 30)
        return max(0.0, min(1.0, complexity))

    def get_argument_summary(self, metrics: ArgumentMetrics) -> Dict[str, Any]:
        """Get a summary of argument analysis."""
        return {
            "overall_score": (metrics.strength + metrics.clarity + metrics.persuasiveness) / 3,
            "strength_level": self._get_level_description(metrics.strength),
            "clarity_level": self._get_level_description(metrics.clarity),
            "persuasiveness_level": self._get_level_description(metrics.persuasiveness),
            "evidence_quality": self._get_level_description(metrics.evidence_quality),
            "sentiment": "positive" if metrics.sentiment_score > 0.1 else "negative" if metrics.sentiment_score < -0.1 else "neutral",
            "fallacy_warning": f"Detected {metrics.fallacy_count} potential logical fallacies" if metrics.fallacy_count > 0 else "No obvious fallacies detected",
            "recommendations": self._get_recommendations(metrics)
        }

    def _get_level_description(self, score: float) -> str:
        """Convert score to descriptive level."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"

    def _get_recommendations(self, metrics: ArgumentMetrics) -> List[str]:
        """Get improvement recommendations based on metrics."""
        recommendations = []
        
        if metrics.strength < 0.6:
            recommendations.append("Add more logical connectors (because, therefore, thus)")
        
        if metrics.clarity < 0.6:
            recommendations.append("Use shorter, clearer sentences")
        
        if metrics.evidence_quality < 0.5:
            recommendations.append("Include specific evidence, statistics, or examples")
        
        if metrics.fallacy_count > 0:
            recommendations.append("Review for logical fallacies and strengthen reasoning")
        
        if metrics.sentiment_score < -0.3:
            recommendations.append("Consider more balanced language to avoid overly negative tone")
        
        if not recommendations:
            recommendations.append("Strong argument! Consider adding more specific examples.")
        
        return recommendations

    def compare_arguments(self, arg1: str, arg2: str) -> Dict[str, Any]:
        """Compare two arguments and provide analysis."""
        metrics1 = self.analyze_argument(arg1)
        metrics2 = self.analyze_argument(arg2)
        
        return {
            "argument1": {
                "metrics": metrics1,
                "summary": self.get_argument_summary(metrics1)
            },
            "argument2": {
                "metrics": metrics2,
                "summary": self.get_argument_summary(metrics2)
            },
            "comparison": {
                "strength_winner": "argument1" if metrics1.strength > metrics2.strength else "argument2",
                "clarity_winner": "argument1" if metrics1.clarity > metrics2.clarity else "argument2",
                "persuasiveness_winner": "argument1" if metrics1.persuasiveness > metrics2.persuasiveness else "argument2",
                "overall_winner": "argument1" if (metrics1.strength + metrics1.clarity + metrics1.persuasiveness) / 3 > 
                                (metrics2.strength + metrics2.clarity + metrics2.persuasiveness) / 3 else "argument2"
            }
        }

