
# ============================================================================
# FILE: ai_engine/llm_interface.py
# Grok API Integration with RAG Context
# ============================================================================

"""
This is the LLM reasoning layer with anti-hallucination measures.
It ALWAYS receives RAG context from the knowledge graph.
"""

import httpx
import json
from typing import Dict, Any, Optional
from loguru import logger
import asyncio

from core.config import settings
from core.exceptions import APIError


class GrokLLMInterface:
    """
    Interface to Grok-2 LLM with medical reasoning capabilities.
    
    Key features:
    - RAG-augmented prompting (prevents hallucination)
    - Medical reasoning system prompt
    - Retry logic with exponential backoff
    - Response validation
    """
    
    def __init__(self):
        self.api_key = settings.grok_api_key
        self.api_url = settings.grok_api_url
        self.model = settings.grok_model
        self.temperature = settings.grok_temperature
        self.max_tokens = settings.grok_max_tokens
        self.timeout = settings.request_timeout
        
        self.client = httpx.AsyncClient(timeout=self.timeout)
        
        logger.info(f"Grok LLM Interface initialized (model: {self.model})")
    
    async def get_medical_diagnosis(
        self,
        rag_context: str,
        additional_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get medical diagnosis from Grok with RAG context.
        
        This is the core function that prevents hallucination by
        providing verified medical knowledge.
        
        Args:
            rag_context: RAG context from knowledge graph
            additional_instructions: Additional diagnostic instructions
        
        Returns:
            Structured diagnosis response
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(rag_context, additional_instructions)
        
        try:
            response = await self._call_grok_api(system_prompt, user_prompt)
            diagnosis = self._parse_response(response)
            return diagnosis
            
        except Exception as e:
            logger.error(f"LLM diagnosis error: {str(e)}")
            raise APIError(f"LLM reasoning failed: {str(e)}")
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt that defines LLM behavior.
        
        This is CRITICAL - it sets the rules for medical reasoning.
        """
        return """You are an expert medical AI diagnostic system with advanced clinical reasoning capabilities.

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE:

1. KNOWLEDGE BASE GROUNDING:
   - You will receive a "VERIFIED MEDICAL KNOWLEDGE BASE" with diseases from validated medical databases
   - Your primary diagnosis MUST come from the diseases listed in this knowledge base
   - You CANNOT suggest diseases not present in the knowledge base
   - Always reference ICD-10 codes from the knowledge base

2. DIAGNOSTIC REASONING:
   - Analyze the patient's symptoms against the knowledge base matches
   - Consider match scores, coverage percentages, and ICD-10 codes
   - If imaging results are provided, integrate them with symptom analysis
   - Use step-by-step clinical reasoning

3. SAFETY FIRST:
   - Flag any CRITICAL or urgent conditions immediately
   - If symptoms suggest multiple serious conditions, list all in differentials
   - Be conservative - when uncertain, recommend clinical evaluation
   - Never dismiss concerning symptoms

4. RESPONSE FORMAT:
   - Return ONLY valid JSON (no markdown, no explanations outside JSON)
   - Use the exact structure specified below
   - All probabilities must sum to reasonable values
   - Be specific and actionable in recommendations

5. TRANSPARENCY:
   - Explain how knowledge base matches support your diagnosis
   - If confidence is low, acknowledge limitations
   - Recommend appropriate next steps

REQUIRED JSON STRUCTURE:
{
  "primary_diagnosis": "Most likely condition from knowledge base",
  "confidence_score": 0-100 (be honest about uncertainty),
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "reasoning": "Step-by-step clinical reasoning referencing KB matches",
  "differential_diagnoses": [
    {
      "disease": "Disease name from knowledge base",
      "probability": 0-100,
      "rationale": "Why this is considered",
      "icd10": "ICD-10 code from knowledge base"
    }
  ],
  "red_flags": ["Any urgent warning signs found"],
  "recommended_tests": ["Specific diagnostic tests from KB"],
  "immediate_actions": ["What patient should do NOW"],
  "clinical_notes": "Additional important context",
  "knowledge_base_alignment": "How well KB matches support this diagnosis"
}

REMEMBER: This is AI-assisted medical screening, NOT a replacement for professional diagnosis. Always recommend clinical evaluation for definitive diagnosis."""
    
    def _build_user_prompt(
        self,
        rag_context: str,
        additional_instructions: Optional[str]
    ) -> str:
        """Build the user prompt with RAG context"""
        prompt_parts = [rag_context]
        
        if additional_instructions:
            prompt_parts.append(f"\n{additional_instructions}")
        
        prompt_parts.append(
            "\nProvide comprehensive medical diagnosis based on the information above. "
            "Return ONLY valid JSON following the exact structure specified in your instructions."
        )
        
        return "\n".join(prompt_parts)
    
    async def _call_grok_api(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3
    ) -> str:
        """
        Call Grok API with retry logic.
        
        Implements exponential backoff for reliability.
        """
        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    return content
                
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    raise APIError(f"Grok API returned {response.status_code}")
                
            except httpx.RequestError as e:
                if attempt == max_retries - 1:
                    raise APIError(f"API request failed after {max_retries} attempts")
                
                wait_time = 2 ** attempt
                logger.warning(f"Request failed, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        
        raise APIError("Max retries exceeded")
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse and validate LLM response.
        
        Extracts JSON from response and validates structure.
        """
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            diagnosis = json.loads(json_str)
            
            # Validate required fields
            required_fields = [
                'primary_diagnosis', 'confidence_score', 'risk_level',
                'reasoning', 'differential_diagnoses', 'recommended_tests',
                'immediate_actions', 'clinical_notes'
            ]
            
            for field in required_fields:
                if field not in diagnosis:
                    logger.warning(f"Missing field in response: {field}")
                    diagnosis[field] = self._get_default_value(field)
            
            # Validate confidence score
            if not 0 <= diagnosis['confidence_score'] <= 100:
                diagnosis['confidence_score'] = min(100, max(0, diagnosis['confidence_score']))
            
            return diagnosis
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {str(e)}")
            logger.error(f"Response was: {response[:500]}")
            raise APIError("Failed to parse LLM response as JSON")
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing fields"""
        defaults = {
            'primary_diagnosis': 'Requires clinical evaluation',
            'confidence_score': 50,
            'risk_level': 'MEDIUM',
            'reasoning': 'Insufficient information for detailed reasoning',
            'differential_diagnoses': [],
            'red_flags': [],
            'recommended_tests': ['Comprehensive medical evaluation'],
            'immediate_actions': ['Consult healthcare provider'],
            'clinical_notes': 'Unable to generate complete analysis',
            'knowledge_base_alignment': 'Unknown'
        }
        return defaults.get(field, '')
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
