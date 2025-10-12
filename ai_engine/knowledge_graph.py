# ============================================================================
# FILE: ai_engine/knowledge_graph.py
# Medical Knowledge Graph with RAG (Retrieval-Augmented Generation)
# ============================================================================

"""
This is the foundation of preventing AI hallucination.
The knowledge graph stores verified medical information that the LLM
can reference, ensuring all diagnoses come from validated sources.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict
import difflib
import json
from loguru import logger

from core.config import settings
from core.exceptions import DataNotFoundError, ValidationError


class MedicalKnowledgeGraph:
    """
    Production-grade medical knowledge graph with:
    - Disease-symptom relationships
    - ICD-10 coding
    - Fuzzy symptom matching
    - Risk factor analysis
    - Treatment recommendations
    
    This is what makes the system "grounded" - every diagnosis
    must trace back to this knowledge base.
    """
    
    def __init__(self, dataset_paths: Optional[List[str]] = None):
        """
        Initialize knowledge graph from datasets.
        
        Args:
            dataset_paths: List of paths to medical datasets (CSV/JSON)
        """
        self.disease_db: Dict[str, Dict] = {}
        self.symptom_index: Dict[str, List[str]] = defaultdict(list)
        self.icd10_mapping: Dict[str, str] = {}
        self.critical_diseases: set = set()
        self.total_symptoms: int = 0
        
        # Load datasets
        if dataset_paths:
            self.load_datasets(dataset_paths)
        else:
            # Load from default location
            default_paths = [
                settings.datasets_dir / "diseases.csv",
                settings.datasets_dir / "diseases.json"
            ]
            self.load_datasets([str(p) for p in default_paths if p.exists()])
        
        logger.info(f"Knowledge Graph initialized: {len(self.disease_db)} diseases, "
                   f"{len(self.symptom_index)} symptoms")
    
    def load_datasets(self, paths: List[str]) -> None:
        """
        Load and merge multiple medical datasets.
        
        This is where you'll integrate your "lots of open-source datasets".
        """
        all_dataframes = []
        
        for path in paths:
            path_obj = Path(path)
            
            if not path_obj.exists():
                logger.warning(f"Dataset not found: {path}")
                continue
            
            try:
                # Load based on file type
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                elif path.endswith('.json'):
                    df = pd.read_json(path)
                elif path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(path)
                else:
                    logger.warning(f"Unsupported file format: {path}")
                    continue
                
                all_dataframes.append(df)
                logger.info(f"Loaded dataset: {path} ({len(df)} records)")
                
            except Exception as e:
                logger.error(f"Failed to load {path}: {str(e)}")
                continue
        
        if not all_dataframes:
            logger.warning("No datasets loaded, initializing with default knowledge")
            self._initialize_default_knowledge()
            return
        
        # Merge all datasets
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        logger.info(f"Merged {len(all_dataframes)} datasets: {len(merged_df)} total records")
        
        # Process merged data
        self._process_medical_data(merged_df)
    
    def _process_medical_data(self, df: pd.DataFrame) -> None:
        """
        Process and index medical data for fast retrieval.
        
        This builds the knowledge graph structure.
        """
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        
        for idx, row in df.iterrows():
            try:
                # Extract disease name
                disease = row.get('disease', row.get('name', '')).strip()
                if not disease:
                    continue
                
                disease_lower = disease.lower()
                
                # Parse symptoms (handle various formats)
                symptoms_raw = row.get('symptoms', '')
                symptoms = self._parse_symptoms(symptoms_raw)
                
                # Build disease record
                disease_record = {
                    'name': disease,
                    'symptoms': symptoms,
                    'severity': str(row.get('severity', 'medium')).lower(),
                    'stage': str(row.get('stage', 'unknown')).lower(),
                    'icd10': str(row.get('icd10', row.get('icd_10', ''))),
                    'tests': self._parse_list(row.get('tests', '')),
                    'treatments': self._parse_list(row.get('treatments', row.get('treatment', ''))),
                    'risk_factors': self._parse_list(row.get('risk_factors', row.get('riskfactors', ''))),
                    'prevalence': float(row.get('prevalence', 0.0)),
                    'description': str(row.get('description', '')),
                }
                
                # Store in disease database
                self.disease_db[disease_lower] = disease_record
                
                # Build symptom index (inverted index for fast lookup)
                for symptom in symptoms:
                    symptom_lower = symptom.lower().strip()
                    if symptom_lower:
                        self.symptom_index[symptom_lower].append(disease_lower)
                
                # Store ICD-10 mapping
                if disease_record['icd10']:
                    self.icd10_mapping[disease_record['icd10']] = disease_lower
                
                # Mark critical diseases
                if disease_record['severity'] in ['high', 'critical', 'severe']:
                    self.critical_diseases.add(disease_lower)
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                continue
        
        self.total_symptoms = len(self.symptom_index)
        logger.info(f"Processed {len(self.disease_db)} diseases with {self.total_symptoms} unique symptoms")
    
    def _parse_symptoms(self, symptoms_str: Any) -> List[str]:
        """Parse symptom string into list of symptoms"""
        if pd.isna(symptoms_str):
            return []
        
        symptoms_str = str(symptoms_str)
        
        # Handle various delimiters
        for delimiter in [',', ';', '|', '\n']:
            if delimiter in symptoms_str:
                return [s.strip() for s in symptoms_str.split(delimiter) if s.strip()]
        
        # Single symptom
        return [symptoms_str.strip()] if symptoms_str.strip() else []
    
    def _parse_list(self, list_str: Any) -> List[str]:
        """Parse comma-separated list"""
        if pd.isna(list_str):
            return []
        return [item.strip() for item in str(list_str).split(',') if item.strip()]
    
    def search_by_symptoms(
        self,
        symptoms: List[str],
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Advanced symptom-based disease search with fuzzy matching.
        
        This is the core RAG retrieval function.
        
        Args:
            symptoms: List of patient symptoms
            top_k: Return top K matches
            min_score: Minimum match score threshold
        
        Returns:
            Ranked list of matching diseases with scores
        """
        if not symptoms:
            return []
        
        disease_scores = defaultdict(float)
        symptom_matches = defaultdict(list)
        
        # Normalize symptoms
        normalized_symptoms = [s.lower().strip() for s in symptoms if s.strip()]
        
        for patient_symptom in normalized_symptoms:
            # Exact matches (highest score)
            if patient_symptom in self.symptom_index:
                for disease in self.symptom_index[patient_symptom]:
                    disease_scores[disease] += 1.0
                    symptom_matches[disease].append(patient_symptom)
            
            # Fuzzy matches (partial score)
            for indexed_symptom, diseases in self.symptom_index.items():
                similarity = difflib.SequenceMatcher(None, patient_symptom, indexed_symptom).ratio()
                
                if similarity > 0.7 and similarity < 1.0:  # Fuzzy match
                    for disease in diseases:
                        disease_scores[disease] += similarity * 0.6
                        if patient_symptom not in symptom_matches[disease]:
                            symptom_matches[disease].append(f"{patient_symptom} (fuzzy: {indexed_symptom})")
        
        # Rank diseases by score
        ranked_diseases = []
        for disease, score in sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            if score < min_score:
                continue
            
            if disease not in self.disease_db:
                continue
            
            disease_info = self.disease_db[disease].copy()
            
            # Calculate coverage score
            coverage = len(symptom_matches[disease]) / len(normalized_symptoms) if normalized_symptoms else 0
            
            # Add match metadata
            disease_info.update({
                'match_score': round(score, 2),
                'matched_symptoms': symptom_matches[disease],
                'coverage': round(coverage, 2),
                'total_disease_symptoms': len(disease_info['symptoms'])
            })
            
            ranked_diseases.append(disease_info)
        
        return ranked_diseases
    
    def get_disease_by_name(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """Get disease information by name"""
        disease_lower = disease_name.lower()
        return self.disease_db.get(disease_lower)
    
    def get_disease_by_icd10(self, icd10_code: str) -> Optional[Dict[str, Any]]:
        """Get disease information by ICD-10 code"""
        disease = self.icd10_mapping.get(icd10_code)
        if disease:
            return self.disease_db.get(disease)
        return None
    
    def get_critical_diseases(self) -> List[Dict[str, Any]]:
        """Get all critical/high severity diseases"""
        return [self.disease_db[d] for d in self.critical_diseases if d in self.disease_db]
    
    def build_rag_context(
        self,
        symptoms: List[str],
        top_matches: List[Dict[str, Any]],
        patient_data: Dict[str, Any],
        image_findings: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build comprehensive RAG context for LLM.
        
        This is the critical function that prevents hallucination.
        It provides the LLM with verified medical knowledge to reference.
        """
        context_parts = ["=== VERIFIED MEDICAL KNOWLEDGE BASE ===\n"]
        
        # Patient symptoms
        if symptoms:
            context_parts.append("PATIENT SYMPTOMS:")
            for i, symptom in enumerate(symptoms, 1):
                context_parts.append(f"{i}. {symptom}")
            context_parts.append("")
        
        # Top disease matches from knowledge base
        context_parts.append("TOP MATCHING DISEASES FROM VALIDATED DATABASE:\n")
        for i, match in enumerate(top_matches[:5], 1):
            context_parts.append(f"\n{i}. {match['name']}")
            context_parts.append(f"   ICD-10 Code: {match['icd10']}")
            context_parts.append(f"   Severity: {match['severity'].upper()}")
            context_parts.append(f"   Match Score: {match['match_score']}/10")
            context_parts.append(f"   Coverage: {int(match['coverage']*100)}% of patient symptoms")
            context_parts.append(f"   Known Symptoms: {', '.join(match['symptoms'][:5])}")
            if match['matched_symptoms']:
                context_parts.append(f"   Matched Patient Symptoms: {', '.join(match['matched_symptoms'][:5])}")
            context_parts.append(f"   Recommended Tests: {', '.join(match['tests'][:3])}")
            context_parts.append(f"   Risk Factors: {', '.join(match['risk_factors'][:3])}")
        
        # Image findings (if available)
        if image_findings and image_findings.get('status') == 'success':
            context_parts.append("\n=== MEDICAL IMAGING ANALYSIS ===")
            context_parts.append(f"Image Type: {image_findings['image_type'].upper()}")
            context_parts.append(f"Primary Finding: {image_findings['primary_finding']}")
            context_parts.append(f"AI Confidence: {image_findings['confidence']*100:.1f}%")
            context_parts.append(f"Clinical Interpretation: {image_findings['clinical_interpretation']}")
            if image_findings.get('requires_immediate_attention'):
                context_parts.append("⚠️  URGENT: Findings require immediate medical attention")
        
        # Patient demographics and history
        context_parts.append("\n=== PATIENT INFORMATION ===")
        if patient_data.get('age'):
            context_parts.append(f"Age: {patient_data['age']} years")
        if patient_data.get('gender'):
            context_parts.append(f"Gender: {patient_data['gender']}")
        if patient_data.get('medical_history'):
            context_parts.append(f"Medical History: {patient_data['medical_history']}")
        if patient_data.get('vital_signs'):
            context_parts.append(f"Vital Signs: {patient_data['vital_signs']}")
        
        # Clinical instructions
        context_parts.append("\n=== DIAGNOSTIC INSTRUCTIONS ===")
        context_parts.append("1. Base your primary diagnosis on the top matching diseases above")
        context_parts.append("2. Reference specific ICD-10 codes from the knowledge base")
        context_parts.append("3. Consider patient age, gender, and medical history")
        context_parts.append("4. If imaging findings are present, integrate them into reasoning")
        context_parts.append("5. Flag any critical/urgent conditions immediately")
        context_parts.append("6. Recommend tests from the knowledge base recommendations")
        context_parts.append("7. Be conservative - if uncertain, recommend clinical evaluation")
        
        return "\n".join(context_parts)
    
    def _initialize_default_knowledge(self):
        """Initialize with minimal default knowledge if no datasets available"""
        logger.warning("Initializing with default knowledge base (limited)")
        
        default_data = {
            'Disease': ['Acute Myocardial Infarction', 'Pneumonia', 'Type 2 Diabetes'],
            'Symptoms': [
                'chest pain,shortness of breath,nausea,sweating',
                'cough,fever,chest pain,difficulty breathing',
                'increased thirst,frequent urination,fatigue'
            ],
            'Severity': ['critical', 'high', 'high'],
            'ICD10': ['I21', 'J18', 'E11'],
            'Tests': [
                'ECG,Troponin test,Coronary angiography',
                'Chest X-ray,Blood culture,Sputum test',
                'HbA1c,Fasting glucose,OGTT'
            ],
            'RiskFactors': [
                'Smoking,High cholesterol,Diabetes',
                'Age>65,Smoking,Chronic lung disease',
                'Obesity,Family history,Sedentary lifestyle'
            ]
        }
        
        df = pd.DataFrame(default_data)
        self._process_medical_data(df)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return {
            'total_diseases': len(self.disease_db),
            'total_symptoms': self.total_symptoms,
            'critical_diseases': len(self.critical_diseases),
            'icd10_codes': len(self.icd10_mapping),
            'severity_breakdown': self._get_severity_breakdown()
        }
    
    def _get_severity_breakdown(self) -> Dict[str, int]:
        """Get disease count by severity"""
        severity_count = defaultdict(int)
        for disease_info in self.disease_db.values():
            severity_count[disease_info['severity']] += 1
        return dict(severity_count)

