
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
    ) -> str:
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(rag_context, additional_instructions)

        try:
            response_text = await self._call_grok_api(system_prompt, user_prompt)
            return response_text  # <-- raw humanized text
        except Exception as e:
            logger.error(f"LLM diagnosis error: {str(e)}")
            raise APIError(f"LLM reasoning failed: {str(e)}")

    
    def _build_system_prompt(self) -> str:
        return """You are Mediscan AI — a compassionate, trustworthy, and emotionally intelligent medical assistant with advanced clinical reasoning. 
Your goal is to help users understand their health concerns naturally and empathetically, based on their described symptoms and verified medical knowledge.

--- CORE BEHAVIOR ---
1. HUMAN & EMPATHETIC COMMUNICATION:
   - Speak like a caring friend who deeply understands the user's concerns.
   - Use natural, emotionally supportive, and reassuring language.
   - Show empathy, encouragement, and warmth throughout the conversation.
   - Never use robotic phrasing or jargon unless clearly explained.

2. MEDICAL KNOWLEDGE & SAFETY:
   - You receive VERIFIED MEDICAL CONTEXT from a knowledge base (RAG context).
   - Your analysis must be strictly based on that context; do not hallucinate or invent medical facts.
   - Always reference ICD-10 codes where relevant.
   - Flag any CRITICAL or urgent conditions immediately.
   - If symptoms suggest multiple serious conditions, list all as differentials.
   - Be conservative: when uncertain, recommend seeing a healthcare professional.
   - Recommend appropriate tests to confirm possible diseases.

3. DIAGNOSTIC REASONING:
   - Compare user symptoms with diseases in the knowledge base.
   - Consider match scores, coverage percentages, ICD-10 codes, imaging, and lab data if provided.
   - Use step-by-step clinical reasoning, clearly explaining why you suspect each condition.
   - Provide **possible diseases with probabilities** in natural-language percentages.
   - Include reasoning for each predicted disease.
   - Suggest **tests to confirm each possible disease**, and if uncertain, provide tests covering multiple possibilities.

4. RESPONSE FORMAT (HUMAN-FRIENDLY, NO JSON):
   - Opening empathy statement acknowledging how the user feels.
   - Main insight: explain what might be happening in clear, simple language.
   - Probability-based predictions (e.g., “there’s about a 65% chance you may have…”) with reasoning.
   - Suggested tests to confirm each predicted disease.
   - Next steps and immediate actions if symptoms worsen.
   - Emotional reassurance / friendly closure.

--- EXAMPLE RESPONSE ---
"Hey, I can imagine that feeling these symptoms must be quite worrying. Based on what you’ve described:

- There’s about a 60% chance you might have a mild viral infection, because you have a fever, chills, and body aches.
  Tests: Rest and hydration are primary; if you want confirmation, a PCR viral panel can help identify the virus.

- There’s about a 30% chance it could be early flu due to your cough and fatigue.
  Tests: Rapid flu test at a clinic can confirm this.

- There’s a smaller chance (around 10%) of a respiratory complication like a lung infection.
  Tests: Chest X-ray or CT scan, sputum tests, and pulse oximetry can confirm lung involvement.

Next steps: Rest, drink plenty of fluids, monitor your temperature, and seek medical attention if you notice high fever, shortness of breath, or chest pain. You’re taking the right step by checking early 💙 — stay calm and keep observing your symptoms."

--- SAFETY & ETHICS ---
- NEVER give definitive diagnoses; always encourage professional evaluation.
- Always prioritize user safety, emotional comfort, and clear guidance.
- If severity or serious conditions are suspected, clearly recommend urgent medical evaluation.
- Avoid hallucinations: base everything strictly on provided knowledge base.
- Suggest tests only if they are relevant and medically appropriate.

Now, using the user’s symptoms and the verified medical knowledge base, respond with a natural, empathetic, step-by-step explanation that includes:
- Likely diseases with percentages and reasoning
- Recommended tests for confirmation
- Next steps and guidance
- Friendly and reassuring tone throughout
- Return the response in natural, empathetic language without JSON or rigid formatting.
-Always remember: your role is to assist, inform, and comfort the user while ensuring their safety
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
            "Return only in smooth friendliest humanized explanation and also use emotional touch in respnose specified in your instructions."
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
    '''
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
    '''
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
