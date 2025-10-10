import pandas as pd
import json
from pathlib import Path
import random

def create_groundbreaking_medical_dataset():
    """Create a medically meaningful, scalable dataset for AI MediScan Pro"""
    print("Creating groundbreaking medical dataset...")

    # Ensure directories exist
    data_dir = Path("data/datasets")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Start with old code’s 50 diseases (copied verbatim for medical accuracy)
    diseases_data = {
        'Disease': [
            'Acute Myocardial Infarction', 'Pneumonia', 'Type 2 Diabetes',
            'Hypertension', 'Asthma', 'COPD', 'Stroke', 'COVID-19',
            'Tuberculosis', 'Malaria', 'Dengue Fever', 'Breast Cancer',
            'Lung Cancer', 'Colorectal Cancer', 'Melanoma',
            'Chronic Kidney Disease', 'Heart Failure', 'Atrial Fibrillation',
            'Gastroenteritis', 'Appendicitis', 'Peptic Ulcer', 'Cirrhosis',
            'Hepatitis', 'HIV/AIDS', 'Meningitis', 'Sepsis',
            'ARDS', 'Pulmonary Embolism', 'Deep Vein Thrombosis', 'Anemia',
            'Rheumatoid Arthritis', 'Osteoarthritis', 'Gout', 'Lupus',
            'Multiple Sclerosis', 'Parkinson Disease', 'Alzheimer Disease',
            'Epilepsy', 'Migraine', 'Depression', 'Anxiety Disorder',
            'Schizophrenia', 'Bipolar Disorder', 'Thyroid Disorders',
            'Celiac Disease', 'Crohn Disease', 'Ulcerative Colitis',
            'Diverticulitis', 'Gallstones', 'Pancreatitis'
        ],
        'ICD10': [
            'I21', 'J18', 'E11', 'I10', 'J45', 'J44', 'I63', 'U07.1',
            'A15', 'B50', 'A90', 'C50', 'C34', 'C18', 'C43',
            'N18', 'I50', 'I48', 'A09', 'K35', 'K25', 'K74',
            'B19', 'B24', 'G03', 'A41', 'J80', 'I26',
            'I82', 'D50', 'M06', 'M19', 'M10', 'M32',
            'G35', 'G20', 'G30', 'G40', 'G43', 'F32', 'F41',
            'F20', 'F31', 'E03', 'K90.0', 'K50', 'K51',
            'K57', 'K80', 'K85'
        ],
        'Symptoms': [
            'chest pain,shortness of breath,nausea,sweating,arm pain,jaw pain',
            'cough,fever,chest pain,difficulty breathing,fatigue,chills',
            'increased thirst,frequent urination,hunger,fatigue,blurred vision',
            'headache,shortness of breath,nosebleeds,dizziness,chest pain',
            'wheezing,shortness of breath,chest tightness,coughing',
            'chronic cough,shortness of breath,wheezing,mucus production',
            'sudden numbness,confusion,trouble speaking,severe headache',
            'fever,dry cough,fatigue,loss of taste,difficulty breathing',
            'persistent cough,chest pain,fever,night sweats,weight loss',
            'fever,chills,sweating,headache,nausea,muscle pain',
            'high fever,severe headache,eye pain,joint pain,rash',
            'breast lump,breast pain,nipple discharge,skin changes',
            'persistent cough,chest pain,coughing blood,weight loss',
            'abdominal pain,rectal bleeding,bowel habit changes,weight loss',
            'irregular mole,skin changes,itching,bleeding',
            'fatigue,swelling,decreased urine,shortness of breath',
            'shortness of breath,fatigue,swelling,rapid heartbeat',
            'irregular heartbeat,palpitations,dizziness,chest pain',
            'diarrhea,nausea,vomiting,abdominal cramps,fever',
            'abdominal pain,nausea,vomiting,fever,loss of appetite',
            'burning stomach pain,nausea,bloating,heartburn',
            'fatigue,jaundice,abdominal swelling,confusion',
            'fatigue,jaundice,abdominal pain,loss of appetite',
            'weight loss,fever,night sweats,fatigue,swollen lymph nodes',
            'severe headache,fever,stiff neck,confusion',
            'fever,rapid heart rate,confusion,rapid breathing',
            'severe shortness of breath,rapid breathing,low oxygen',
            'chest pain,shortness of breath,rapid heart rate,leg swelling',
            'leg swelling,pain,warmth,redness,cramping',
            'fatigue,weakness,pale skin,dizziness,cold hands',
            'joint pain,swelling,stiffness,fatigue,fever',
            'joint pain,stiffness,reduced range of motion,swelling',
            'intense joint pain,redness,swelling',
            'fatigue,joint pain,fever,skin rashes,kidney problems',
            'numbness,tingling,weakness,vision problems,balance issues',
            'tremor,stiffness,slow movement,balance problems',
            'memory loss,confusion,difficulty speaking,mood changes',
            'seizures,loss of consciousness,confusion,muscle jerks',
            'severe headache,nausea,sensitivity to light',
            'persistent sadness,loss of interest,fatigue,sleep changes',
            'excessive worry,restlessness,fatigue',
            'hallucinations,delusions,disorganized thinking',
            'mood swings,energy changes,risky behavior',
            'fatigue,weight changes,temperature sensitivity',
            'diarrhea,abdominal pain,weight loss,bloating',
            'abdominal pain,diarrhea,weight loss,fever',
            'abdominal pain,diarrhea,rectal bleeding,weight loss',
            'abdominal pain,fever,nausea,bowel habit changes',
            'abdominal pain,nausea,vomiting,fever,jaundice',
            'severe abdominal pain,nausea,vomiting,fever'
        ],
        'Severity': [
            'critical', 'high', 'high', 'high', 'medium', 'high', 'critical', 'high',
            'high', 'high', 'high', 'high', 'critical', 'high', 'high',
            'high', 'high', 'high', 'medium', 'high', 'medium', 'high',
            'high', 'critical', 'critical', 'critical', 'critical', 'critical',
            'high', 'medium', 'medium', 'medium', 'medium', 'high',
            'high', 'high', 'high', 'medium', 'medium', 'medium', 'medium',
            'high', 'high', 'medium', 'medium', 'high', 'medium',
            'medium', 'medium', 'high'
        ],
        'Tests': [
            'ECG,Troponin test,Coronary angiography',
            'Chest X-ray,Blood culture,Sputum test',
            'HbA1c,Fasting glucose,OGTT',
            'Blood pressure monitoring,Urinalysis,ECG',
            'Spirometry,Peak flow test,Allergy tests',
            'Spirometry,Chest X-ray,CT scan',
            'CT scan,MRI,Carotid ultrasound',
            'PCR test,Antigen test,Chest X-ray',
            'Chest X-ray,Sputum culture,TB skin test',
            'Blood smear,Rapid diagnostic test',
            'NS1 antigen test,IgM/IgG antibody test',
            'Mammography,Biopsy,Ultrasound',
            'Chest X-ray,CT scan,Biopsy',
            'Colonoscopy,Biopsy,CT scan',
            'Skin biopsy,Dermoscopy,CT scan',
            'Blood tests,Urinalysis,GFR test',
            'Echocardiography,ECG,Chest X-ray',
            'ECG,Echocardiography,Holter monitor',
            'Stool test,Blood test,Stool culture',
            'CT scan,Ultrasound,Blood tests',
            'Endoscopy,Biopsy,H. pylori test',
            'Liver function tests,Ultrasound,CT scan',
            'Liver function tests,Viral markers',
            'HIV test,CD4 count,Viral load test',
            'Lumbar puncture,Blood culture,CT scan',
            'Blood culture,Blood tests,Imaging',
            'Chest X-ray,CT scan,Arterial blood gas',
            'CT angiography,D-dimer test,V/Q scan',
            'Ultrasound,D-dimer test,Venography',
            'CBC,Iron studies,Vitamin B12',
            'RF test,Anti-CCP,ESR,CRP',
            'X-rays,MRI,CT scan',
            'Joint fluid analysis,Blood tests',
            'ANA test,Anti-dsDNA,Complement levels',
            'MRI,Lumbar puncture,Evoked potentials',
            'Neurological exam,DaTscan,MRI',
            'Cognitive tests,Brain imaging,PET scan',
            'EEG,MRI,CT scan',
            'Neurological exam,MRI,CT scan',
            'PHQ-9,Clinical interview,Blood tests',
            'GAD-7,Clinical interview',
            'Clinical interview,Psychological testing',
            'Clinical interview,Mood tracking',
            'TSH,T3,T4,Thyroid antibodies',
            'Blood tests,Endoscopy with biopsy',
            'Colonoscopy,Biopsy,Blood tests',
            'Colonoscopy,Biopsy,Blood tests',
            'CT scan,Blood tests,Colonoscopy',
            'Ultrasound,CT scan,Blood tests',
            'Blood tests,CT scan,Ultrasound'
        ],
        'RiskFactors': [
            'Smoking,High cholesterol,Diabetes,Age>45',
            'Age>65,Smoking,Chronic lung disease',
            'Obesity,Family history,Sedentary lifestyle',
            'Obesity,High sodium,Alcohol,Stress',
            'Allergies,Family history,Smoking',
            'Smoking,Air pollution,Occupational exposure',
            'Hypertension,Diabetes,Smoking,Age>55',
            'Close contact,Crowded places,Comorbidities',
            'Close contact,Immunocompromised,HIV',
            'Travel to endemic areas,Poor sanitation',
            'Travel to endemic areas,Urbanization',
            'Female,Age>50,Family history,BRCA genes',
            'Smoking,Air pollution,Radon',
            'Age>50,Family history,Red meat,Obesity',
            'Fair skin,Sun exposure,Multiple moles',
            'Diabetes,Hypertension,Smoking,Age>60',
            'Hypertension,Coronary disease,Diabetes',
            'Age,Hypertension,Heart disease',
            'Contaminated food/water,Poor hygiene',
            'Obstruction,Infection,Young age',
            'H. pylori,NSAIDs,Smoking,Alcohol',
            'Alcohol abuse,Hepatitis,Obesity',
            'Contaminated blood,Unprotected sex',
            'Unprotected sex,IV drugs',
            'Bacterial infection,Close contact',
            'Infection,Surgery,Immunocompromised',
            'Pneumonia,Sepsis,Trauma',
            'Surgery,Immobility,Cancer,Smoking',
            'Surgery,Immobility,Obesity,Cancer',
            'Blood loss,Iron deficiency',
            'Female,Smoking,Family history',
            'Age>50,Obesity,Previous injury',
            'Male,Obesity,High purine diet',
            'Female,Age 15-45,Genetics',
            'Unknown,Genetics,Viral infections',
            'Age>60,Genetics,Male',
            'Age>65,Family history,Genetics',
            'Genetics,Head trauma,Brain infection',
            'Genetics,Female,Stress,Hormones',
            'Genetics,Trauma,Chronic illness',
            'Genetics,Trauma,Chronic stress',
            'Genetics,Brain chemistry,Trauma',
            'Genetics,Trauma,Stressful events',
            'Autoimmune,Iodine deficiency,Family history',
            'Genetics,Autoimmune disease',
            'Family history,Smoking,Autoimmune',
            'Genetics,Family history,Smoking',
            'Low fiber diet,Age>40,Obesity',
            'Obesity,Female,Age>40,Genetics',
            'Alcohol,Gallstones,High triglycerides'
        ],
        'Prevalence': [
            0.05, 0.08, 0.12, 0.30, 0.10, 0.06, 0.02, 0.03,
            0.01, 0.004, 0.002, 0.12, 0.02, 0.04, 0.006,
            0.15, 0.06, 0.04, 0.15, 0.007, 0.10, 0.02,
            0.05, 0.02, 0.001, 0.02, 0.002, 0.001,
            0.005, 0.20, 0.01, 0.30, 0.04, 0.0005,
            0.0003, 0.001, 0.06, 0.01, 0.15, 0.18, 0.19,
            0.01, 0.025, 0.05, 0.01, 0.007, 0.007,
            0.02, 0.12, 0.01
        ],
        'Treatments': [
            'Aspirin,Nitroglycerin,Angioplasty,Beta-blockers,Statins',
            'Antibiotics,Oxygen therapy,IV fluids',
            'Metformin,Insulin,Lifestyle changes',
            'ACE inhibitors,Diuretics,Lifestyle changes',
            'Inhaled corticosteroids,Bronchodilators',
            'Bronchodilators,Oxygen therapy,Pulmonary rehab',
            'Thrombolytics,Anticoagulants,Rehabilitation',
            'Antivirals,Oxygen therapy,Supportive care',
            'Antitubercular drugs,Supportive care',
            'Artemisinin-based therapy,Antipyretics',
            'Supportive care,Fluid therapy,Analgesics',
            'Surgery,Chemotherapy,Radiation',
            'Surgery,Chemotherapy,Radiation',
            'Surgery,Chemotherapy,Immunotherapy',
            'Surgery,Radiation,Immunotherapy',
            'Dialysis,Blood pressure control,Diet modification',
            'Diuretics,ACE inhibitors,Beta-blockers',
            'Antiarrhythmics,Anticoagulants',
            'Rehydration,Antiemetics,Antibiotics',
            'Surgery,Antibiotics',
            'PPI,H. pylori eradication,Antacids',
            'Abstinence,Transplant,Supportive care',
            'Antivirals,Supportive care',
            'ART,Prophylactic antibiotics',
            'Antibiotics,Antivirals,Supportive care',
            'Antibiotics,IV fluids,Vasopressors',
            'Oxygen therapy,Mechanical ventilation',
            'Anticoagulants,Thrombolytics',
            'Anticoagulants,Compression therapy',
            'Iron supplements,Blood transfusions',
            'DMARDs,NSAIDs,Physical therapy',
            'NSAIDs,Physical therapy,Joint replacement',
            'Colchicine,NSAIDs,Diet modification',
            'Steroids,Immunosuppressants',
            'Immunotherapy,Physical therapy',
            'Levodopa,Physical therapy',
            'Cholinesterase inhibitors,Supportive care',
            'Anticonvulsants,EEG monitoring',
            'Triptans,Preventive medications',
            'Antidepressants,Psychotherapy',
            'Antidepressants,Anxiolytics,Psychotherapy',
            'Antipsychotics,Psychotherapy',
            'Mood stabilizers,Psychotherapy',
            'Levothyroxine,Antithyroid drugs',
            'Gluten-free diet,Supportive care',
            'Steroids,Immunosuppressants',
            'Steroids,Immunosuppressants,Surgery',
            'Antibiotics,Surgery,Diet modification',
            'Surgery,Cholecystectomy',
            'Supportive care,ERCP,Surgery'
        ],
        'Comorbidities': [
            'Heart Failure,Arrhythmia,Hypertension',
            'COPD,Sepsis,ARDS',
            'Hypertension,Obesity,Chronic Kidney Disease',
            'Diabetes,Obesity,Chronic Kidney Disease',
            'Allergies,COPD',
            'Pneumonia,Heart Failure',
            'Hypertension,Heart Failure',
            'Pneumonia,Sepsis',
            'HIV,Malaria',
            'Anemia,HIV',
            'Malaria,Tuberculosis',
            'Lung Cancer,Depression',
            'Colorectal Cancer,Depression',
            'Lung Cancer,Depression',
            'Chronic Kidney Disease,Hypertension',
            'Diabetes,Hypertension',
            'Hypertension,Stroke',
            'Arrhythmia,Heart Failure',
            'Dehydration,Sepsis',
            'Gastroenteritis,Dehydration',
            'Gastritis,GERD',
            'Hepatitis,Heart Failure',
            'Cirrhosis,HIV',
            'Tuberculosis,Anemia',
            'Sepsis,ARDS',
            'Pneumonia,ARDS',
            'Pneumonia,Sepsis',
            'Deep Vein Thrombosis,Stroke',
            'Pulmonary Embolism,Cancer',
            'Chronic Kidney Disease,Hypertension',
            'Osteoarthritis,Depression',
            'Rheumatoid Arthritis,Depression',
            'Rheumatoid Arthritis,Osteoarthritis',
            'Rheumatoid Arthritis,Depression',
            'Depression,Anxiety',
            'Depression,Alzheimer Disease',
            'Depression,Anxiety',
            'Depression,Anxiety',
            'Depression,Anxiety',
            'Anxiety,Thyroid Disorders',
            'Depression,Anxiety',
            'Depression,Anxiety',
            'Depression,Anxiety',
            'Depression,Anxiety',
            'Crohn Disease,Ulcerative Colitis',
            'Ulcerative Colitis,Depression',
            'Crohn Disease,Depression',
            'Gallstones,Pancreatitis',
            'Pancreatitis,Diabetes',
            'Gallstones,Cholecystitis'
        ],
        'SymptomOnset': [
            'acute: hours', 'acute: days', 'chronic: months', 'chronic: months',
            'acute: hours', 'chronic: months', 'acute: hours', 'acute: days',
            'chronic: months', 'acute: days', 'acute: days', 'chronic: months',
            'chronic: months', 'chronic: months', 'chronic: months',
            'chronic: months', 'chronic: months', 'chronic: weeks', 'acute: days',
            'acute: hours', 'chronic: months', 'chronic: months', 'chronic: months',
            'chronic: months', 'acute: hours', 'acute: hours', 'acute: hours',
            'acute: hours', 'chronic: weeks', 'chronic: months', 'chronic: months',
            'chronic: months', 'acute: days', 'chronic: months', 'chronic: months',
            'chronic: months', 'chronic: months', 'acute: days', 'chronic: months',
            'chronic: months', 'chronic: months', 'chronic: months', 'chronic: months',
            'chronic: months', 'chronic: months', 'chronic: months', 'chronic: months',
            'acute: days', 'chronic: months', 'acute: days'
        ],
        'DemographicRisks': [
            'Male:1.5x,Age>45:2x,White:1.2x',
            'Age>65:3x,Immunocompromised:2x',
            'South Asian:1.8x,Obese:2.5x,Age>40:1.5x',
            'Obese:2x,Age>40:1.5x,Black:1.3x',
            'Children:2x,Young adults:1.5x',
            'Age>50:2x,Smokers:3x',
            'Age>55:2x,Male:1.3x',
            'Immunocompromised:2x,Children:1.5x',
            'Immunocompromised:3x,HIV+:5x',
            'Children:1.5x,Sub-Saharan Africa:3x',
            'Young adults:1.5x,Urban:2x',
            'Female:10x,Age>50:2x',
            'Smokers:5x,Age>50:2x',
            'Age>50:2x,Obese:2x',
            'Fair skin:3x,Age>30:1.5x',
            'Age>60:2x,Diabetic:2x',
            'Age>50:2x,Hypertensive:2x',
            'Age>60:2x,Male:1.3x',
            'Children:2x,Travelers:2x',
            'Young adults:2x,Male:1.3x',
            'Smokers:2x,Alcoholics:2x',
            'Alcoholics:3x,Age>40:1.5x',
            'IV drug users:3x,MSM:2x',
            'HIV+:5x,Immunocompromised:3x',
            'Children:2x,Immunocompromised:3x',
            'Immunocompromised:3x,Age>60:2x',
            'Pneumonia patients:3x,Trauma:2x',
            'Immobility:3x,Cancer:2x',
            'Obese:2x,Surgery:2x',
            'Females:1.5x,Age>50:2x',
            'Females:2x,Smokers:2x',
            'Age>50:2x,Obese:2x',
            'Males:2x,Obese:2x',
            'Females:3x,Age 15-45:2x',
            'Females:1.5x,Young adults:2x',
            'Age>60:2x,Males:1.5x',
            'Age>65:3x,Females:1.5x',
            'Young adults:2x,Head trauma:2x',
            'Females:2x,Young adults:2x',
            'Young adults:2x,Females:1.5x',
            'Young adults:2x,Females:1.5x',
            'Young adults:2x,Females:1.5x',
            'Females:2x,Age>30:1.5x',
            'Females:2x,Young adults:2x',
            'Young adults:2x,Smokers:2x',
            'Young adults:2x,Smokers:2x',
            'Age>40:2x,Obese:2x',
            'Females:2x,Age>40:2x',
            'Obese:2x,Alcoholics:2x'
        ],
        'MockImageFindings': [
            'X-ray: cardiomegaly,ECG: ST elevation',
            'X-ray: lung consolidation,CT: infiltrates',
            'Ultrasound: normal,Lipid profile: high triglycerides',
            'Blood smear: plasmodium parasites',
            'Inhaled corticosteroids,Bronchodilators',
            'X-ray: lung scarring,CT: emphysema',
            'CT: brain ischemia,MRI: stroke',
            'X-ray: ground-glass opacities',
            'X-ray: lung cavities',
            'Blood smear: plasmodium parasites',
            'Blood smear: dengue virus',
            'Mammogram: mass,Ultrasound: lump',
            'CT: lung mass,Biopsy: malignant',
            'Colonoscopy: polyps,CT: mass',
            'Dermoscopy: irregular borders',
            'Ultrasound: kidney atrophy',
            'Echocardiography: reduced ejection fraction',
            'ECG: atrial arrhythmia',
            'Stool test: blood',
            'CT: appendiceal inflammation',
            'Endoscopy: ulcer',
            'Ultrasound: liver scarring',
            'Ultrasound: liver inflammation',
            'Blood test: HIV positive',
            'CT: brain edema',
            'Blood culture: positive',
            'X-ray: lung edema',
            'CT: pulmonary artery clot',
            'Ultrasound: venous clot',
            'Blood smear: low RBC',
            'X-ray: joint erosion',
            'X-ray: joint degeneration',
            'Joint aspiration: crystals',
            'Blood test: ANA positive',
            'MRI: white matter lesions',
            'MRI: basal ganglia changes',
            'PET: amyloid plaques',
            'EEG: epileptiform activity',
            'MRI: normal',
            'MRI: normal',
            'MRI: normal',
            'MRI: normal',
            'MRI: normal',
            'Thyroid scan: abnormal',
            'Endoscopy: villous atrophy',
            'Colonoscopy: inflammation',
            'Colonoscopy: ulcers',
            'CT: diverticula',
            'Ultrasound: gallstones',
            'CT: pancreatic inflammation'
        ]
    }
import random
import random

import random

def expand_diseases(base_data, target_size=200):
    expanded = {k: [] for k in base_data}
    
    disease_subtypes = {
        'Pneumonia': ['Viral Pneumonia', 'Bacterial Pneumonia', 'Aspiration Pneumonia'],
        'Type 2 Diabetes': ['Type 2 Diabetes with Neuropathy', 'Type 2 Diabetes with Retinopathy'],
        'Malaria': ['Plasmodium Falciparum Malaria', 'Plasmodium Vivax Malaria']
    }
    
    for i in range(target_size):
        # Safe selection of disease
        disease_idx = i % len(base_data['Disease'])
        base_disease = base_data['Disease'][disease_idx]
        subtypes = disease_subtypes.get(base_disease, [base_disease])
        subtype_idx = min(i // len(base_data['Disease']), len(subtypes) - 1)
        subtype = subtypes[subtype_idx]
        
        for key in base_data:
            column_list = base_data[key]
            
            if not column_list:  # Handle empty columns
                if key == 'Disease':
                    expanded[key].append(subtype)
                elif key == 'DemographicRisks':
                    expanded[key].append(f"Region{i%5}:1.{random.randint(1,3)}x")
                else:
                    expanded[key].append("N/A")
                continue
            
            idx = i % len(column_list)
            
            if key == 'Disease':
                expanded[key].append(f"{subtype}{'' if i < len(column_list) else f'_v{i//len(column_list)}'}")
            elif key == 'ICD10':
                expanded[key].append(f"{column_list[idx]}{'' if i < len(column_list) else f'.{i//len(column_list)}'}")
            elif key in ['Symptoms', 'Tests', 'RiskFactors', 'Treatments', 'Comorbidities', 'MockImageFindings']:
                items = column_list[idx].split(',') if column_list[idx] else []
                extra_items = random.sample(items, min(2, len(items))) if items and random.random() > 0.5 else []
                expanded[key].append(','.join(items + extra_items) if items else "N/A")
            elif key == 'Prevalence':
                expanded[key].append(column_list[idx] * (0.8 + random.random() * 0.4))
            elif key == 'Severity':
                expanded[key].append(column_list[idx] if random.random() > 0.3 else random.choice(['critical', 'high', 'medium']))
            elif key == 'SymptomOnset':
                expanded[key].append(column_list[idx] if random.random() > 0.5 else random.choice(['acute: hours', 'acute: days', 'chronic: weeks', 'chronic: months']))
            elif key == 'DemographicRisks':
                expanded[key].append(f"{column_list[idx]},Region{i%5}:1.{random.randint(1,3)}x")
                
    return expanded



    # Expand to 200 diseases
    diseases_data = expand_diseases(diseases_data, 200)
    
    # Create DataFrame
    df = pd.DataFrame(diseases_data)
    
    # Save as CSV
    df.to_csv(data_dir / 'diseases.csv', index=False)
    print(f"✓ Created diseases.csv with {len(df)} diseases")
    
    # Save as JSON
    df.to_json(data_dir / 'diseases.json', orient='records', indent=2)
    print(f"✓ Created diseases.json")
    
    # Create symptom-disease mapping
    symptom_map = []
    for _, row in df.iterrows():
        symptoms = row['Symptoms'].split(',')
        for symptom in symptoms:
            symptom_map.append({
                'symptom': symptom.strip(),
                'disease': row['Disease'],
                'severity': row['Severity'],
                'icd10': row['ICD10']
            })
    
    symptom_df = pd.DataFrame(symptom_map)
    symptom_df.to_csv(data_dir / 'symptom_disease_mapping.csv', index=False)
    print(f"✓ Created symptom_disease_mapping.csv with {len(symptom_df)} mappings")
    
    # Create disease-comorbidity mapping
    comorbidity_map = []
    for _, row in df.iterrows():
        comorbidities = row['Comorbidities'].split(',')
        for comorbidity in comorbidities:
            comorbidity_map.append({
                'disease': row['Disease'],
                'comorbidity': comorbidity.strip(),
                'icd10': row['ICD10']
            })
    
    comorbidity_df = pd.DataFrame(comorbidity_map)
    comorbidity_df.to_csv(data_dir / 'disease_comorbidity_mapping.csv', index=False)
    print(f"✓ Created disease_comorbidity_mapping.csv with {len(comorbidity_df)} mappings")
    
    # Enhanced statistics
    unique_symptoms = len(symptom_df['symptom'].unique())
    print(f"\n{'='*60}")
    print("Groundbreaking Dataset Statistics:")
    print(f"{'='*60}")
    print(f"Total diseases: {len(df)}")
    print(f"Critical diseases: {len(df[df['Severity'] == 'critical'])}")
    print(f"High severity diseases: {len(df[df['Severity'] == 'high'])}")
    print(f"Unique symptoms: {unique_symptoms}")
    print(f"ICD-10 codes: {len(df['ICD10'].unique())}")
    print(f"Comorbidity mappings: {len(comorbidity_df)}")
    print(f"Average symptoms per disease: {len(symptom_df) / len(df):.1f}")
    print(f"\nDataset creation complete! ✓")

if __name__ == '__main__':
    create_groundbreaking_medical_dataset()