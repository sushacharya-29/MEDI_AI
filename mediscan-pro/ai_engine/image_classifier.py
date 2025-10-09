
# ============================================================================
# FILE: ai_engine/image_classifier.py
# Medical Image Analysis with Deep Learning
# ============================================================================

"""
This module handles medical image analysis using transfer learning
on pre-trained CNNs (ResNet, EfficientNet).

For hackathon: These are initialized with ImageNet weights.
For production: Would be fine-tuned on medical datasets (ChestX-ray14, etc.)
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
from typing import Dict, Any, Optional, Tuple
from loguru import logger

from core.config import settings
from core.exceptions import ModelError


class MedicalImageClassifier:
    """
    Deep learning-based medical image classifier.
    
    Supports multiple imaging modalities:
    - Chest X-rays (ResNet50)
    - Skin lesions (EfficientNet)
    - CT scans (ResNet50)
    
    Uses transfer learning for fast deployment.
    """
    
    def __init__(self):
        self.device = torch.device(settings.torch_device)
        self.models = {}
        self.transforms = {}
        self._initialize_models()
        
        logger.info(f"Image Classifier initialized on {self.device}")
    
    def _initialize_models(self):
        """Initialize pre-trained models for different image types"""
        
        # Standard ImageNet normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
        
        # Chest X-ray model (ResNet50)
        self.models['xray'] = self._create_xray_model()
        
        # Skin lesion model (EfficientNet)
        self.models['skin'] = self._create_skin_model()
        
        # CT scan model (ResNet50)
        self.models['ct'] = self._create_ct_model()
        
        logger.info(f"Loaded {len(self.models)} medical imaging models")
    
    def _create_xray_model(self) -> nn.Module:
        """
        Create chest X-ray classification model.
        
        In production: Load fine-tuned weights from ChestX-ray14 dataset.
        For hackathon: Use ImageNet pre-trained as base.
        """
        model = models.resnet50(pretrained=True)
        
        # Replace final layer for 14 chest conditions
        # (Common findings in chest X-rays)
        num_classes = 14
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # In production, load trained weights:
        # checkpoint = torch.load(settings.models_dir / 'xray_model.pth')
        # model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        return model
    
    def _create_skin_model(self) -> nn.Module:
        """Create skin lesion classification model"""
        model = models.efficientnet_b0(pretrained=True)
        
        # Replace classifier for 7 skin conditions
        num_classes = 7
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )
        
        model.to(self.device)
        model.eval()
        return model
    
    def _create_ct_model(self) -> nn.Module:
        """Create CT scan classification model"""
        # For hackathon, use same architecture as X-ray
        return self._create_xray_model()
    
    async def analyze_image(
        self,
        image_data: bytes,
        image_type: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Analyze medical image and return findings.
        
        Args:
            image_data: Raw image bytes
            image_type: 'xray', 'skin', 'ct', or 'auto' for auto-detection
        
        Returns:
            Dictionary with findings and confidence scores
        """
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Auto-detect image type if needed
            if image_type == 'auto':
                image_type = self._detect_image_type(image)
            
            # Get appropriate model
            model = self.models.get(image_type, self.models['xray'])
            
            # Preprocess
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Map prediction to clinical finding
            findings = self._map_to_clinical_findings(
                predicted.item(),
                confidence.item(),
                image_type,
                probabilities[0].cpu().numpy()
            )
            
            return findings
            
        except Exception as e:
            logger.error(f"Image analysis error: {str(e)}")
            raise ModelError(f"Image analysis failed: {str(e)}")
    
    def _detect_image_type(self, image: Image.Image) -> str:
        """
        Auto-detect medical image type from characteristics.
        
        Simple heuristics:
        - Grayscale → likely X-ray/CT
        - Color with skin tones → likely dermoscopy
        """
        img_array = np.array(image)
        
        # Check if grayscale
        if len(img_array.shape) == 2:
            return 'xray'
        
        # Check color variance
        if img_array.shape[2] == 3:
            # Calculate variance in each channel
            r_var = np.var(img_array[:,:,0])
            g_var = np.var(img_array[:,:,1])
            b_var = np.var(img_array[:,:,2])
            
            # High variance in all channels suggests color image (skin)
            if min(r_var, g_var, b_var) > 1000:
                return 'skin'
        
        # Default to X-ray
        return 'xray'
    
    def _map_to_clinical_findings(
        self,
        pred_idx: int,
        confidence: float,
        image_type: str,
        all_probs: np.ndarray
    ) -> Dict[str, Any]:
        """
        Map model predictions to clinical findings with interpretation.
        
        This is where we translate model outputs into medical language.
        """
        # Define condition labels for each imaging type
        labels_map = {
            'xray': [
                'Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis',
                'Lung Cancer', 'Pneumothorax', 'Pleural Effusion',
                'Cardiomegaly', 'Atelectasis', 'Consolidation',
                'Edema', 'Emphysema', 'Fibrosis', 'Nodule'
            ],
            'skin': [
                'Melanoma', 'Basal Cell Carcinoma', 'Benign Keratosis',
                'Dermatofibroma', 'Melanocytic Nevus', 'Vascular Lesion',
                'Actinic Keratosis'
            ],
            'ct': [
                'Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis',
                'Lung Cancer', 'Pneumothorax', 'Pleural Effusion',
                'Cardiomegaly', 'Atelectasis', 'Consolidation',
                'Edema', 'Emphysema', 'Fibrosis', 'Nodule'
            ]
        }
        
        labels = labels_map.get(image_type, labels_map['xray'])
        primary_finding = labels[pred_idx] if pred_idx < len(labels) else 'Unknown'
        
        # Get top 3 predictions
        top3_indices = np.argsort(all_probs)[-3:][::-1]
        differential_findings = [
            {'finding': labels[i] if i < len(labels) else 'Unknown', 
             'probability': float(all_probs[i])}
            for i in top3_indices
        ]
        
        # Generate clinical interpretation
        interpretation = self._generate_interpretation(
            primary_finding,
            confidence,
            image_type
        )
        
        # Determine urgency
        requires_urgent_attention = self._assess_urgency(
            primary_finding,
            confidence
        )
        
        return {
            'status': 'success',
            'image_type': image_type,
            'primary_finding': primary_finding,
            'confidence': float(confidence),
            'differential_findings': differential_findings,
            'requires_immediate_attention': requires_urgent_attention,
            'clinical_interpretation': interpretation,
            'model_used': f'{image_type}_classifier',
            'device': str(self.device)
        }
    
    def _generate_interpretation(
        self,
        finding: str,
        confidence: float,
        image_type: str
    ) -> str:
        """Generate human-readable clinical interpretation"""
        if confidence < 0.3:
            return ("Image quality may be insufficient for reliable automated analysis. "
                   "Manual review by radiologist recommended.")
        
        elif confidence < 0.6:
            return (f"Imaging findings suggest possible {finding}. "
                   f"Confidence level is moderate ({confidence*100:.1f}%). "
                   f"Clinical correlation and additional imaging may be needed.")
        
        elif confidence < 0.8:
            return (f"Imaging findings are consistent with {finding} "
                   f"(confidence: {confidence*100:.1f}%). "
                   f"Clinical correlation recommended.")
        
        else:
            return (f"Imaging findings strongly suggest {finding} "
                   f"(high confidence: {confidence*100:.1f}%). "
                   f"Recommend urgent clinical evaluation and appropriate management.")
    
    def _assess_urgency(self, finding: str, confidence: float) -> bool:
        """Determine if findings require immediate medical attention"""
        urgent_conditions = {
            'Pneumonia', 'COVID-19', 'Tuberculosis', 'Lung Cancer',
            'Pneumothorax', 'Melanoma', 'Basal Cell Carcinoma'
        }
        
        return finding in urgent_conditions and confidence > 0.7
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'available_models': list(self.models.keys()),
            'device': str(self.device),
            'model_architectures': {
                'xray': 'ResNet50',
                'skin': 'EfficientNet-B0',
                'ct': 'ResNet50'
            }
        }
