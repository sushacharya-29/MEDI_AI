
# ============================================================================
# FILE: scripts/validate_accuracy.py
# Test accuracy with real medical cases
# ============================================================================

"""
This script validates system accuracy against known medical cases.
Use this to demonstrate >90% accuracy to judges.
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app import app
from ai_engine.knowledge_graph import MedicalKnowledgeGraph
from ai_engine.image_classifier import MedicalImageClassifier
from ai_engine.nlp_processor import ClinicalNLPProcessor
from ai_engine.llm_interface import GrokLLMInterface
from ai_engine.diagnostic_engine import HybridDiagnosticEngine

# Test cases with known diagnoses
TEST_CASES = [
    {
        'case_id': 1,
        'symptoms': 'severe chest pain radiating to left arm, shortness of breath, sweating, nausea',
        'age': 58,
        'gender': 'male',
        'medical_history': 'hypertension, high cholesterol, smoking',
        'expected_diagnosis': 'Acute Myocardial Infarction',
        'expected_risk': 'CRITICAL'
    },
    {
        'case_id': 2,
        'symptoms': 'cough with yellow sputum, fever, chest pain, difficulty breathing',
        'age': 72,
        'gender': 'female',
        'medical_history': 'COPD, smoking history',
        'expected_diagnosis': 'Pneumonia',
        'expected_risk': 'HIGH'
    },
    {
        'case_id': 3,
        'symptoms': 'increased thirst, frequent urination, fatigue, blurred vision, slow healing wounds',
        'age': 45,
        'gender': 'male',
        'medical_history': 'obesity, family history of diabetes',
        'expected_diagnosis': 'Type 2 Diabetes',
        'expected_risk': 'HIGH'
    },
    {
        'case_id': 4,
        'symptoms': 'severe headache, stiff neck, fever, sensitivity to light, confusion',
        'age': 25,
        'gender': 'female',
        'medical_history': 'none',
        'expected_diagnosis': 'Meningitis',
        'expected_risk': 'CRITICAL'
    },
    {
        'case_id': 5,
        'symptoms': 'persistent cough for 3 weeks, night sweats, weight loss, chest pain',
        'age': 38,
        'gender': 'male',
        'medical_history': 'HIV positive, recent travel',
        'expected_diagnosis': 'Tuberculosis',
        'expected_risk': 'HIGH'
    }
]

async def validate_accuracy():
    """Run accuracy validation tests"""
    
    print("="*80)
    print("AI MediScan Pro - Accuracy Validation")
    print("="*80)
    
    # Initialize system
    print("\nInitializing AI system...")
    dataset_paths = ["data/datasets/diseases.csv"]
    
    kg = MedicalKnowledgeGraph(dataset_paths)
    image_clf = MedicalImageClassifier()
    nlp = ClinicalNLPProcessor()
    llm = GrokLLMInterface()
    engine = HybridDiagnosticEngine(kg, image_clf, nlp, llm)
    
    print(f"Knowledge Base: {len(kg.disease_db)} diseases loaded\n")
    
    results = []
    correct_primary = 0
    correct_risk = 0
    
    for test_case in TEST_CASES:
        print(f"Test Case {test_case['case_id']}:")
        print(f"  Symptoms: {test_case['symptoms'][:60]}...")
        print(f"  Expected: {test_case['expected_diagnosis']} ({test_case['expected_risk']})")
        
        try:
            # Run diagnosis
            diagnosis = await engine.diagnose(
                symptoms_text=test_case['symptoms'],
                patient_data={
                    'age': test_case['age'],
                    'gender': test_case['gender'],
                    'medical_history': test_case['medical_history']
                }
            )
            
            predicted = diagnosis['primary_diagnosis']
            predicted_risk = diagnosis['risk_level']
            confidence = diagnosis['confidence_score']
            
            print(f"  Predicted: {predicted} ({predicted_risk})")
            print(f"  Confidence: {confidence}%")
            
            # Check accuracy
            primary_correct = test_case['expected_diagnosis'].lower() in predicted.lower()
            risk_correct = test_case['expected_risk'] == predicted_risk
            
            if primary_correct:
                correct_primary += 1
                print("  ✓ Primary diagnosis CORRECT")
            else:
                print("  ✗ Primary diagnosis INCORRECT")
            
            if risk_correct:
                correct_risk += 1
                print("  ✓ Risk level CORRECT")
            else:
                print("  ✗ Risk level INCORRECT")
            
            results.append({
                'case_id': test_case['case_id'],
                'primary_correct': primary_correct,
                'risk_correct': risk_correct,
                'confidence': confidence
            })
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results.append({
                'case_id': test_case['case_id'],
                'primary_correct': False,
                'risk_correct': False,
                'confidence': 0
            })
        
        print()
    
    # Calculate metrics
    total_cases = len(TEST_CASES)
    primary_accuracy = (correct_primary / total_cases) * 100
    risk_accuracy = (correct_risk / total_cases) * 100
    avg_confidence = sum(r['confidence'] for r in results) / total_cases
    
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total test cases: {total_cases}")
    print(f"Primary diagnosis accuracy: {primary_accuracy:.1f}% ({correct_primary}/{total_cases})")
    print(f"Risk level accuracy: {risk_accuracy:.1f}% ({correct_risk}/{total_cases})")
    print(f"Average confidence: {avg_confidence:.1f}%")
    print("="*80)
    
    if primary_accuracy >= 90:
        print("\n✓ PASSED: >90% accuracy achieved!")
    else:
        print("\n✗ FAILED: Accuracy below 90% threshold")
    
    # Cleanup
    await llm.close()

if __name__ == '__main__':
    asyncio.run(validate_accuracy())

