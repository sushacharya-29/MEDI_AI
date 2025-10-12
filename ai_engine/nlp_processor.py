
# ============================================================================
# FILE: ai_engine/nlp_processor.py
# Clinical NLP for Symptom Extraction and Analysis
# ============================================================================

"""
This module extracts medical entities from patient descriptions.
Uses regex patterns and medical dictionaries for clinical terminology.

In production: Would use spaCy's medical NER (medspacy) or BioBERT
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from loguru import logger


class ClinicalNLPProcessor:
    """
    Clinical Natural Language Processing for medical text analysis.
    
    Capabilities:
    - Symptom extraction from free text
    - Severity assessment
    - Temporal reasoning (duration)
    - Anatomical location identification
    - Negation detection (important!)
    """
    
    def __init__(self):
        self._compile_patterns()
        self._load_medical_vocabulary()
        logger.info("Clinical NLP Processor initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for medical entity recognition"""
        
        # Symptom categories with patterns
        self.symptom_patterns = {
            'pain': re.compile(
                r'\b(pain|ache|aching|sore|tender|discomfort|hurt|burning|stabbing|sharp|dull)\b',
                re.IGNORECASE
            ),
            'fever': re.compile(
                r'\b(fever|febrile|pyrexia|temperature|hot|chills|sweating)\b',
                re.IGNORECASE
            ),
            'respiratory': re.compile(
                r'\b(cough|coughing|breathing|dyspnea|wheezing|shortness of breath|SOB|breathless)\b',
                re.IGNORECASE
            ),
            'cardiac': re.compile(
                r'\b(chest pain|palpitations|tachycardia|bradycardia|heart|cardiac)\b',
                re.IGNORECASE
            ),
            'gastrointestinal': re.compile(
                r'\b(nausea|vomiting|diarrhea|constipation|abdominal|stomach|gut)\b',
                re.IGNORECASE
            ),
            'neurological': re.compile(
                r'\b(headache|dizziness|dizzy|vertigo|numbness|tingling|weakness|confusion)\b',
                re.IGNORECASE
            ),
            'dermatological': re.compile(
                r'\b(rash|itching|lesion|skin|swelling|redness|inflammation)\b',
                re.IGNORECASE
            )
        }
        
        # Severity indicators
        self.severity_keywords = {
            'severe': 4, 'excruciating': 4, 'unbearable': 4, 'intense': 4,
            'acute': 3, 'strong': 3, 'significant': 3,
            'moderate': 2, 'noticeable': 2,
            'mild': 1, 'slight': 1, 'minor': 1
        }
        
        # Temporal patterns
        self.temporal_patterns = {
            'days': re.compile(r'(\d+)\s*days?', re.IGNORECASE),
            'weeks': re.compile(r'(\d+)\s*weeks?', re.IGNORECASE),
            'months': re.compile(r'(\d+)\s*months?', re.IGNORECASE),
            'years': re.compile(r'(\d+)\s*years?', re.IGNORECASE),
            'hours': re.compile(r'(\d+)\s*hours?', re.IGNORECASE)
        }
        
        self.temporal_multipliers = {
            'hours': 1/24,
            'days': 1,
            'weeks': 7,
            'months': 30,
            'years': 365
        }
        
        # Negation patterns (critical for accuracy!)
        self.negation_pattern = re.compile(
            r'\b(no|not|never|without|absence|deny|denies|negative|ruled out)\b',
            re.IGNORECASE
        )
    
    def _load_medical_vocabulary(self):
        """Load medical terminology dictionary"""
        # In production: Load from UMLS/SNOMED
        # For hackathon: Use curated list
        
        self.anatomical_locations = {
            'head', 'neck', 'chest', 'abdomen', 'back', 'arm', 'leg',
            'stomach', 'throat', 'lung', 'heart', 'brain', 'kidney', 'liver'
        }
        
        self.symptom_synonyms = {
            'dyspnea': 'shortness of breath',
            'pyrexia': 'fever',
            'cephalalgia': 'headache',
            'emesis': 'vomiting',
            'pruritis': 'itching'
        }
    
    def extract_symptoms(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract symptoms from clinical text with context.
        
        Args:
            text: Patient's symptom description
        
        Returns:
            List of extracted symptoms with metadata
        """
        if not text or not text.strip():
            return []
        
        symptoms = []
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check for negation
            if self._is_negated(sentence):
                continue
            
            # Extract symptoms by category
            for category, pattern in self.symptom_patterns.items():
                if pattern.search(sentence):
                    symptom_info = {
                        'symptom': sentence,
                        'category': category,
                        'severity': self._assess_severity(sentence),
                        'duration_days': self._extract_duration(sentence),
                        'location': self._extract_location(sentence),
                        'context': sentence,
                        'importance': self._calculate_importance(sentence, category)
                    }
                    symptoms.append(symptom_info)
        
        # Deduplicate while preserving detail
        symptoms = self._deduplicate_symptoms(symptoms)
        
        return symptoms
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter
        sentences = re.split(r'[.!?;]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_negated(self, sentence: str) -> bool:
        """
        Check if symptom is negated (e.g., "no fever", "denies chest pain").
        
        This is CRITICAL - we must not diagnose based on absent symptoms!
        """
        return bool(self.negation_pattern.search(sentence))
    
    def _assess_severity(self, text: str) -> int:
        """
        Assess symptom severity on 1-4 scale.
        
        1 = mild, 2 = moderate, 3 = significant, 4 = severe/critical
        """
        text_lower = text.lower()
        
        # Check for severity keywords
        for keyword, score in self.severity_keywords.items():
            if keyword in text_lower:
                return score
        
        # Default to moderate if no indicator
        return 2
    
    def _extract_duration(self, text: str) -> Optional[int]:
        """
        Extract symptom duration in days.
        
        Examples:
        - "3 days" → 3
        - "2 weeks" → 14
        - "6 months" → 180
        """
        for unit, pattern in self.temporal_patterns.items():
            match = pattern.search(text)
            if match:
                value = int(match.group(1))
                multiplier = self.temporal_multipliers[unit]
                return int(value * multiplier)
        
        return None
    
    def _extract_location(self, text: str) -> Optional[str]:
        """Extract anatomical location if mentioned"""
        text_lower = text.lower()
        
        for location in self.anatomical_locations:
            if location in text_lower:
                return location
        
        return None
    
    def _calculate_importance(self, text: str, category: str) -> str:
        """
        Calculate symptom importance (low/medium/high).
        
        Critical symptoms get high importance automatically.
        """
        # Critical symptom categories
        critical_categories = {'cardiac', 'neurological'}
        
        # Critical keywords
        critical_keywords = [
            'chest pain', 'severe', 'acute', 'sudden',
            'difficulty breathing', 'confusion', 'unconscious'
        ]
        
        text_lower = text.lower()
        severity = self._assess_severity(text)
        
        # High importance if critical category or keywords
        if category in critical_categories:
            return 'high'
        
        if any(keyword in text_lower for keyword in critical_keywords):
            return 'high'
        
        # High importance if severe
        if severity >= 3:
            return 'high'
        elif severity == 2:
            return 'medium'
        else:
            return 'low'
    
    def _deduplicate_symptoms(self, symptoms: List[Dict]) -> List[Dict]:
        """Remove duplicate symptoms while keeping most detailed version"""
        if not symptoms:
            return []
        
        # Group by category
        by_category = defaultdict(list)
        for symptom in symptoms:
            by_category[symptom['category']].append(symptom)
        
        # Keep most detailed from each category
        deduped = []
        for category_symptoms in by_category.values():
            # Sort by context length (more detail)
            category_symptoms.sort(key=lambda x: len(x['context']), reverse=True)
            deduped.append(category_symptoms[0])
        
        return deduped
    
    def analyze_clinical_significance(
        self,
        symptoms: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze overall clinical significance of symptom cluster.
        
        Returns summary statistics and risk indicators.
        """
        if not symptoms:
            return {
                'total_symptoms': 0,
                'high_importance_count': 0,
                'average_severity': 0,
                'critical_flag': False
            }
        
        high_importance = sum(1 for s in symptoms if s['importance'] == 'high')
        avg_severity = sum(s['severity'] for s in symptoms) / len(symptoms)
        
        # Flag if multiple severe symptoms or any critical category
        critical_flag = (
            high_importance >= 2 or
            any(s['category'] in {'cardiac', 'neurological'} for s in symptoms)
        )
        
        return {
            'total_symptoms': len(symptoms),
            'high_importance_count': high_importance,
            'average_severity': round(avg_severity, 2),
            'critical_flag': critical_flag,
            'categories_present': list(set(s['category'] for s in symptoms))
        }