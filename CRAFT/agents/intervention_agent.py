
import copy
import json
import random
import re
import os
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI
from datetime import datetime
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class InterventionAgent:
    """Intervention agent - generates interventions to help resolve issues in the CRAFT game"""
    
    def __init__(self, use_api=True, api_key=None, model_name="gpt-4o-mini", 
                 local_model=None, local_tokenizer=None):
        self.use_api = use_api
        self.model_name = model_name
        self.local_model = local_model
        self.local_tokenizer = local_tokenizer
        
        if use_api:
            self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
    
    def create_intervention_prompt(self, conversation_history, game_state):
        """Create intervention prompt focused only on friction detection"""
        
        progress_summary = game_state.get_progress_summary()
        
        return f"""You are an Intervention Agent for a collaborative LEGO construction task.

CONVERSATION HISTORY:
{conversation_history}

CURRENT PROGRESS:
- Turn: {progress_summary.get('current_turn', 0)}
- Overall Progress: {progress_summary.get('overall_progress', 0):.2f}
- Completion: {progress_summary.get('completion_percentage', 0):.2f}
- Recent Trend: {progress_summary.get('recent_trend', 0):.3f}

IDENTIFY FRICTION POINTS:
Analyze the conversation for:
- Builder confusion or unclear instructions
- Directors giving conflicting information
- Lack of progress or repeated mistakes
- Communication breakdowns
- Directors not specifying block sizes or positions clearly

OUTPUT FORMAT:
<friction>
D1: [specific issue D1 should address, max 1 sentence or "No issues detected"]
D2: [specific issue D2 should address, max 1 sentence or "No issues detected"] 
D3: [specific issue D3 should address, max 1 sentence or "No issues detected"]
Builder: [specific issue Builder should address, max 1 sentence or "No issues detected"]
GROUP: [coordination strategy for the team, max 1 sentence or "No issues detected"]
</friction>

Focus only on actionable friction points that are hindering progress."""
    
    def analyze_friction(self, conversation_history, game_state) -> Dict:
        """Analyze conversation for friction points"""
        
        prompt = self.create_intervention_prompt(conversation_history, game_state)
        
        try:
            if self.use_api:
                response_text = self._generate_with_api(prompt)
            else:
                response_text = self._generate_with_local_model(prompt)
            
            return self._parse_friction_response(response_text)
            
        except Exception as e:
            return {
                'friction_analysis': f"Error in friction analysis: {str(e)}",
                'raw_response': ""
            }
    
    def _parse_friction_response(self, response_text) -> Dict:
        """Parse friction response"""
        friction_match = re.search(r'<friction>(.*?)</friction>', response_text, re.DOTALL | re.IGNORECASE)
        
        if friction_match:
            friction_text = friction_match.group(1).strip()
            return {
                'friction_analysis': friction_text,
                'raw_response': response_text
            }
        else:
            return {
                'friction_analysis': "No friction analysis found",
                'raw_response': response_text
            }
    
    def _generate_with_api(self, prompt) -> str:
        """Generate using API"""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an Intervention Agent that identifies friction points in collaborative tasks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        if completion and completion.choices:
            return completion.choices[0].message.content
        else:
            raise Exception("Empty API response")
    
    def _generate_with_local_model(self, prompt) -> str:
        """Generate using local model"""
        # Similar pattern as director agent
        if not self.local_model or not self.local_tokenizer:
            raise Exception("Local model and tokenizer required")
        
        inputs = self.local_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.local_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=600,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.local_tokenizer.pad_token_id,
            )
        
        prompt_length = inputs['input_ids'].shape[-1]
        new_tokens = outputs[0][prompt_length:]
        response = self.local_tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response
