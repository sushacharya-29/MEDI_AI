
# ============================================================================
# FILE: ai_engine/diagnostic_engine.py  
# Main Orchestrator - Combines All AI Components
# ============================================================================

"""
This is the BRAIN of the system.
It orchestrates all AI components into a cohesive diagnostic pipeline.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger

from ai_engine.knowledge_graph import MedicalKnowledgeGraph
from ai_engine.image_classifier import MedicalImageClassifier
from ai_engine.nlp_processor import ClinicalNLPProcessor
from ai_engine.llm_interface import GrokLLMInterface
from ai_engine.rag_engine import HyperRAGEngine
from core.exceptions import ValidationError

class HybridDiagnosticEngine:
    """
    Production-grade hybrid diagnostic engine with HyperRAG integration.
    
    Combines:
    1. Medical knowledge graph (rule-based RAG)
    2. HyperRAG (vector-based retrieval)
    3. Clinical NLP (symptom extraction)
    4. Deep learning (image analysis)
    5. LLM reasoning (Grok)
    6. Clinical validation rules
    """
    
    def __init__(
        self,
        knowledge_graph: MedicalKnowledgeGraph,
        image_classifier: MedicalImageClassifier,
        nlp_processor: ClinicalNLPProcessor,
        llm_interface: GrokLLMInterface,
        rag_engine: HyperRAGEngine
    ):
        self.kg = knowledge_graph
        self.image_clf = image_classifier
        self.nlp = nlp_processor
        self.llm = llm_interface
        self.rag_engine = rag_engine
        logger.info("Hybrid Diagnostic Engine initialized with HyperRAG")
    
    async def diagnose(
        self,
        symptoms_text: Optional[str],
        patient_data: Dict[str, Any],
        image_data: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Complete multi-modal diagnostic pipeline with HyperRAG.
        
        Args:
            symptoms_text: Patient's symptom description
            patient_data: Demographics, history, vitals
            image_data: Medical image bytes (optional)
        
        Returns:
            Complete diagnostic report
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Extract and analyze symptoms with NLP
            logger.info("Step 1: Extracting symptoms...")
            extracted_symptoms = []
            symptom_list = []
            
            if symptoms_text:
                extracted_symptoms = self.nlp.extract_symptoms(symptoms_text)
                symptom_list = [s['symptom'] for s in extracted_symptoms]
                logger.info(f"Extracted {len(extracted_symptoms)} symptoms")
            
            # Step 2: Search knowledge graph for rule-based matches
            logger.info("Step 2: Searching knowledge base...")
            kg_matches = self.kg.search_by_symptoms(symptom_list, top_k=10)
            logger.info(f"Found {len(kg_matches)} disease matches")
            
            # Step 3: HyperRAG vector search
            logger.info("Step 3: HyperRAG vector search...")
            query = ", ".join(symptom_list)
            if image_data:
                image_findings = await self.image_clf.analyze_image(image_data)
                query += f", Imaging: {image_findings.get('primary_finding', '')}"
            rag_context = self.rag_engine.retrieve_context(query, top_k=5)
            logger.info(f"Retrieved {len(rag_context)} HyperRAG matches")
            
            # Step 4: Analyze medical image (if provided)
            logger.info("Step 4: Analyzing medical image...")
            image_findings = None
            if image_data:
                image_findings = await self.image_clf.analyze_image(image_data)
                logger.info(f"Image analysis: {image_findings['primary_finding']}")
            
            # Step 5: Build combined RAG context
            logger.info("Step 5: Building combined RAG context...")
            rag_context_str = self.kg.build_rag_context(
                symptom_list,
                kg_matches,
                patient_data,
                image_findings
            )
            rag_context_str += "\nHyperRAG Context:\n"
            for item in rag_context[:3]:
                rag_context_str += f"- {item['disease']} (ICD-10: {item['icd10']}, Score: {item['score']:.2f})\n{item['content']}\n"
            
            # Step 6: Get LLM diagnosis with combined context
            logger.info("Step 6: Getting LLM diagnosis...")
            llm_diagnosis = await self.llm.get_medical_diagnosis(rag_context_str, image_findings=image_findings)
            logger.info(f"LLM diagnosis: {llm_diagnosis['primary_diagnosis']}")
            
            # Step 7: Ensemble and validate predictions
            logger.info("Step 7: Ensemble validation...")
            final_diagnosis = self._ensemble_predictions(
                kg_matches,
                llm_diagnosis,
                image_findings,
                extracted_symptoms,
                rag_context
            )
            
            # Step 8: Apply clinical safety rules
            logger.info("Step 8: Applying clinical rules...")
            final_diagnosis = self._apply_clinical_rules(
                final_diagnosis,
                patient_data,
                extracted_symptoms
            )
            
            # Step 9: Add metadata and explainability
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            final_diagnosis.update({
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round(processing_time, 2),
                'highlighted_symptoms': extracted_symptoms,
                'image_findings': image_findings,
                'hyper_rag_matches': rag_context,
                'confidence_level': self._get_confidence_label(
                    final_diagnosis['confidence_score']
                ),
                'data_sources': self._get_data_sources(kg_matches, image_findings, rag_context),
                'system_version': '4.0.0',
                'model_versions': {
                    'knowledge_base': f"{len(self.kg.disease_db)} diseases",
                    'hyper_rag': 'LangChain/FAISS',
                    'image_classifier': 'ResNet50/EfficientNet',
                    'nlp': 'Clinical NLP v1.0',
                    'llm': self.llm.model
                }
            })
            
            logger.info(f"Diagnosis complete in {processing_time:.0f}ms")
            return final_diagnosis
            
        except Exception as e:
            logger.error(f"Diagnosis pipeline error: {str(e)}", exc_info=True)
            raise
    
    def _ensemble_predictions(
        self,
        kg_matches: List[Dict],
        llm_diagnosis: Dict,
        image_findings: Optional[Dict],
        extracted_symptoms: List[Dict],
        rag_context: List[Dict]
    ) -> Dict:
        """
        Ensemble predictions from knowledge graph, HyperRAG, LLM, and images.
        """
        diagnosis = llm_diagnosis.copy()
        
        # Boost confidence if KG, RAG, and LLM agree
        if kg_matches and rag_context:
            top_kg_disease = kg_matches[0]['name'].lower()
            top_rag_disease = rag_context[0]['disease'].lower()
            llm_disease = diagnosis['primary_diagnosis'].lower()
            
            if (top_kg_disease in llm_disease or llm_disease in top_kg_disease) and \
               (top_rag_disease in llm_disease or llm_disease in top_rag_disease):
                original_conf = diagnosis['confidence_score']
                boost = min(20, 96 - original_conf)
                diagnosis['confidence_score'] = original_conf + boost
                diagnosis['knowledge_base_alignment'] = "Strong agreement with knowledge base and HyperRAG"
                logger.info(f"KG-RAG-LLM agreement, confidence boosted by {boost}")
            else:
                diagnosis['knowledge_base_alignment'] = "Partial alignment with knowledge base and HyperRAG"
        
        # Integrate image findings
        if image_findings and image_findings.get('status') == 'success':
            image_finding = image_findings['primary_finding']
            image_conf = image_findings['confidence']
            
            if image_conf > 0.7:
                if image_finding.lower() in diagnosis['primary_diagnosis'].lower():
                    diagnosis['confidence_score'] = min(96, diagnosis['confidence_score'] + 8)
                    diagnosis['clinical_notes'] += f"\n\nImaging confirms clinical diagnosis ({image_finding})."
                else:
                    diff_diseases = [d['disease'] for d in diagnosis.get('differential_diagnoses', [])]
                    if image_finding not in diff_diseases:
                        diagnosis.setdefault('differential_diagnoses', []).insert(0, {
                            'disease': image_finding,
                            'probability': int(image_conf * 100),
                            'rationale': 'Detected from medical imaging analysis',
                            'icd10': 'Imaging-based'
                        })
        
        # Symptom importance weighting
        high_importance_symptoms = [
            s for s in extracted_symptoms 
            if s.get('importance') == 'high'
        ]
        
        if len(high_importance_symptoms) >= 3:
            if diagnosis['risk_level'] in ['HIGH', 'CRITICAL']:
                diagnosis['confidence_score'] = min(94, diagnosis['confidence_score'] + 5)
        
        return diagnosis
    
    def _are_similar_diseases(self, disease1: str, disease2: str) -> bool:
        """Check if two disease names are similar"""
        common_words = set(disease1.split()) & set(disease2.split())
        return len(common_words) >= 2
    
    def _apply_clinical_rules(
        self,
        diagnosis: Dict,
        patient_data: Dict,
        extracted_symptoms: List[Dict]
    ) -> Dict:
        """
        Apply clinical decision rules and safety checks.
        """
        # Rule 1: Age-based risk adjustment
        age = patient_data.get('age', 0)
        if age and (age < 5 or age > 65):
            if diagnosis['risk_level'] == 'MEDIUM':
                diagnosis['risk_level'] = 'HIGH'
                diagnosis['clinical_notes'] += "\n\nRisk elevated due to age (vulnerable population)."
        
        # Rule 2: Critical symptom detection
        critical_symptoms = [
            'chest pain', 'severe headache', 'difficulty breathing',
            'confusion', 'loss of consciousness', 'severe bleeding'
        ]
        
        for symptom_info in extracted_symptoms:
            symptom_text = symptom_info['symptom'].lower()
            
            for critical in critical_symptoms:
                if critical in symptom_text and symptom_info.get('severity', 0) >= 3:
                    diagnosis['risk_level'] = 'CRITICAL'
                    emergency_action = 'SEEK IMMEDIATE EMERGENCY CARE (CALL 911)'
                    if emergency_action not in diagnosis.get('immediate_actions', []):
                        diagnosis.setdefault('immediate_actions', []).insert(0, emergency_action)
                    flag = f"Critical symptom detected: {critical}"
                    if flag not in diagnosis.get('red_flags', []):
                        diagnosis.setdefault('red_flags', []).append(flag)
        
        # Rule 3: Confidence calibration for safety
        if diagnosis['confidence_score'] > 85:
            num_differentials = len(diagnosis.get('differential_diagnoses', []))
            if num_differentials < 2:
                diagnosis['confidence_score'] = min(85, diagnosis['confidence_score'])
                diagnosis['clinical_notes'] += "\n\nConfidence calibrated for medical safety (limited differential analysis)."
        
        # Rule 4: Duration-based severity
        long_duration_symptoms = [
            s for s in extracted_symptoms 
            if s.get('duration_days', 0) > 30
        ]
        
        if len(long_duration_symptoms) >= 2:
            diagnosis['clinical_notes'] += "\n\nChronic symptoms present (>30 days). Comprehensive evaluation recommended."
            diagnosis.setdefault('recommended_tests', []).append('Comprehensive chronic disease workup')
        
        # Rule 5: Cap maximum confidence at 96% (medical humility)
        diagnosis['confidence_score'] = min(96, diagnosis['confidence_score'])
        
        return diagnosis
    
    def _get_confidence_label(self, score: float) -> str:
        """Convert numeric confidence to human-readable label"""
        if score >= 90:
            return "Very High"
        elif score >= 75:
            return "High"
        elif score >= 60:
            return "Moderate"
        elif score >= 40:
            return "Low"
        else:
            return "Very Low"
    
    def _get_data_sources(
        self,
        kg_matches: List[Dict],
        image_findings: Optional[Dict],
        rag_context: List[Dict]
    ) -> List[str]:
        """List all data sources used in diagnosis"""
        sources = [
            "Medical Knowledge Base (RAG)",
            "HyperRAG (Vector Search)",
            "Clinical NLP Processor"
        ]
        
        if kg_matches:
            sources.append(f"Disease Database ({len(kg_matches)} matches)")
        if rag_context:
            sources.append(f"HyperRAG Database ({len(rag_context)} matches)")
        if image_findings and image_findings.get('status') == 'success':
            sources.append(f"Medical Imaging AI ({image_findings['image_type']})")
        
        sources.append("Grok-4 Medical Reasoning")
        sources.append("Clinical Decision Rules")
        
        return sources