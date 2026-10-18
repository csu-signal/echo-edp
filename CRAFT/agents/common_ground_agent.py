
import json
import re
from typing import Dict, List, Optional, Any
from openai import OpenAI

def create_common_ground_prompt(director_responses, current_board_state, conversation_history, last_move):
    """
    Create prompt for Common Ground agent to analyze true alignment vs surface agreement
    """
    
    # Extract internal thoughts and public messages
    internal_thoughts = {}
    public_messages = {}
    
    for director_id, response in director_responses.items():
        internal_thoughts[director_id] = response.get('internal_thinking', '')
        public_messages[director_id] = response.get('public_message', '')
    
    prompt = f"""You are a Common Ground Analysis agent for a collaborative LEGO construction task. 

Your job is to determine the TRUE alignment between three directors (D1, D2, D3) based on both their public messages and private internal thoughts.
The three Directors (D1, D2, and D3) have to instruct one Builder to build a single structure that is consistent with the private views of the structure they have. From a top-down view of the target structure, D1's private view is of the left wall of the structure, D2's view is of the top wall of the structure, and D3's view of the right wall of the structure.

CURRENT BOARD STATE:
{json.dumps(current_board_state, indent=2)}

CONVERSATION HISTORY:
{conversation_history}

DIRECTOR INTERNAL THOUGHTS (PRIVATE):
D1 Internal: {internal_thoughts.get('D1', 'No internal thoughts provided')}
D2 Internal: {internal_thoughts.get('D2', 'No internal thoughts provided')}  
D3 Internal: {internal_thoughts.get('D3', 'No internal thoughts provided')}

DIRECTOR PUBLIC MESSAGES:
D1 Message: {public_messages.get('D1', 'No message provided')}
D2 Message: {public_messages.get('D2', 'No message provided')}
D3 Message: {public_messages.get('D3', 'No message provided')}

LAST MOVE INFORMATION:
{last_move['move']['confirmation']}

ANALYSIS INSTRUCTIONS:
1. Look for discrepancies between what directors say publicly vs. think privately
2. Identify blocks/positions where directors have genuine alignment vs. surface agreement
3. Detect uncertainty, confusion, or hidden disagreements
4. Consider confidence levels expressed in internal thoughts

OUTPUT TWO SECTIONS:

<analysis>
[Provide a brief analysis (3-4 sentences) about the true alignment state. Highlight any:
- Surface agreements that mask underlying disagreement/uncertainty
- Positions where directors are genuinely aligned
- Areas of confusion or misunderstanding
- Confidence mismatches between directors]
</analysis>

<groupAgreement>
[Based only on what the directors say publicly, provide a single "Yes" or "No" to whether or not last move is agreeable as a small part of the overall structure, thus allowing the group to move on.
An agreeable move might only incorporate two directors' perspectives, since no single block is visible in the privately-held side views of all three directors, therefore agreement between two director may be all that is needed for effective group agreement.]
</groupAgreement>

<groupAgreementJustification>
[Justification for the group agreement value.]
</groupAgreementJustification>

<aligned_structure>
{{
    "D1": {{
        "row_0": [{{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}],
        "row_1": [{{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}],
        "row_2": [{{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}]
    }},
    "D2": {{
        "row_0": [{{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}],
        "row_1": [{{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}],
        "row_2": [{{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}]
    }},
    "D3": {{
        "row_0": [{{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}],
        "row_1": [{{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}],
        "row_2": [{{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}, {{"color":"[color]", "size":[size], "confidence":"high/medium/low"}}]
    }}
}}
</aligned_structure>

IMPORTANT NOTES:
- Use "unknown" for positions not mentioned or unclear
- Use "disputed" for positions where directors disagree
- Set confidence based on internal thoughts: "high" = certain, "medium" = somewhat sure, "low" = uncertain/guessing
- The aligned structure should reflect what each director TRULY believes (from internal thoughts), not just what they said publicly
- Colors: "red", "blue", "green", "yellow", "orange", "brown", "unknown", "disputed"
- Group agreement should NOT be based on ANY director internal thoughts, ONLY what is said publicly. This field MUST be a single word: "yes" or "no".
"""
    
    return prompt

def parse_common_ground_response(response_text):
    """
    Parse the Common Ground agent response to extract analysis and structure
    """
    
    # Extract analysis section
    analysis_match = re.search(r'<analysis>(.*?)</analysis>', response_text, re.DOTALL | re.IGNORECASE)
    analysis = analysis_match.group(1).strip() if analysis_match else ""

    groupAgreement = re.search(r'<groupAgreement>(.*?)</groupAgreement>', response_text, re.DOTALL | re.IGNORECASE)
    agreement = groupAgreement.group(1).strip() if groupAgreement else ""

    groupAgreementJust = re.search(r'<groupAgreementJustification>(.*?)</groupAgreementJustification>', response_text, re.DOTALL | re.IGNORECASE)
    agreementJust = groupAgreementJust.group(1).strip() if groupAgreementJust else ""
    
    # Extract aligned structure section
    structure_match = re.search(r'<aligned_structure>(.*?)</aligned_structure>', response_text, re.DOTALL | re.IGNORECASE)
    
    aligned_structure = None
    if structure_match:
        structure_text = structure_match.group(1).strip()
        try:
            # Try to parse as JSON
            aligned_structure = json.loads(structure_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse aligned structure JSON: {e}")
            print(f"Raw structure text: {structure_text}")
            aligned_structure = None
    
    return {
        'analysis': analysis,
        'aligned_structure': aligned_structure,
        'raw_response': response_text,
        'agreement': agreement,
        'agreement_justification': agreementJust
    }

class CommonGroundAgent:
    """
    Common Ground agent that can use either API or local models
    """
    
    def __init__(self, use_api=True, api_key=None, model_name="gpt-4o-mini", local_model=None, local_tokenizer=None):
        self.use_api = use_api
        self.model_name = model_name
        self.local_model = local_model
        self.local_tokenizer = local_tokenizer
        
        if use_api:
            self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
    
    def generate_common_ground(self, director_responses, current_board_state, conversation_history, last_move):
        """
        Generate common ground analysis using either API or local model
        """
        
        prompt = create_common_ground_prompt(director_responses, current_board_state, conversation_history, last_move)
        
        try:
            if self.use_api:
                response = self._generate_with_api(prompt)
            else:
                response = self._generate_with_local_model(prompt)
            
            return parse_common_ground_response(response)
            
        except Exception as e:
            print(f"Error in common ground generation: {str(e)}")
            return {
                'analysis': f"Error in analysis: {str(e)}",
                'aligned_structure': None,
                'raw_response': "",
                'agreement':" No",
                'agreement_justification': None
            }
    
    def _generate_with_api(self, prompt):
        """Generate response using OpenAI API"""
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a Common Ground Analysis agent that determines true alignment between directors in a collaborative task by analyzing both public statements and private thoughts."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=1024
        )
        
        if completion and completion.choices:
            return completion.choices[0].message.content
        else:
            raise Exception("Empty or invalid API response")
    
    def _generate_with_local_model(self, prompt):
        """Generate response using local model (placeholder for your existing model loading code)"""
        
        if not self.local_model or not self.local_tokenizer:
            raise Exception("Local model and tokenizer must be provided when use_api=False")
        
        # Use your existing generation code here
        # This would follow the pattern from your process_dialogues function
        inputs = self.local_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.local_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.local_tokenizer.pad_token_id,
            )
        
        prompt_length = inputs['input_ids'].shape[-1]
        new_tokens = outputs[0][prompt_length:]
        response = self.local_tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response

# Example usage and testing
def test_common_ground_agent():
    """Test the Common Ground agent"""
    
    # Sample director responses
    director_responses = {
        "D1": {
            "internal_thinking": "I can clearly see a green block at (0,0) and a blue block at (1,1). I'm confident about these positions. D2 said we need blocks on the right, which I agree with based on my target view. I trust the builder understood my coordinates correctly.",
            "public_message": "Builder, place a green block at (1,0) next to the existing green, then add a blue block at (2,0)."
        },
        "D2": {
            "internal_thinking": "From my top-down view, I can see the green and blue blocks that D1 mentioned. However, I'm not entirely sure about the exact positions since my perspective is different. I think we do need more blocks but I'm guessing about the specific positions.",
            "public_message": "Yes, I agree with D1. We definitely need more blocks on the right side."
        },
        "D3": {
            "internal_thinking": "I can see from my side view that there are blocks stacked, but I'm confused about which ones D1 and D2 are referring to. I don't want to disagree publicly but I'm not sure if I'm seeing the same thing they are.",
            "public_message": "That sounds right. Let's add those blocks."
        }
    }
    
    current_board_state = {
        "(0, 0)": ["gs"],
        "(1, 1)": ["bs"],
        # ... other positions empty
    }
    
    conversation_history = "D1: I see a green block at bottom left. Builder: Placed green at (0,0). D2: Looking good so far."
    
    # Test with API (you can set your API key)
    cg_agent = CommonGroundAgent(use_api=True, model_name="gpt-4o-mini")
    
    result = cg_agent.generate_common_ground(director_responses, current_board_state, conversation_history)
    
    print("=== COMMON GROUND ANALYSIS ===")
    print("Analysis:", result['analysis'])
    print("\nAligned Structure Sample:", json.dumps(result['aligned_structure'], indent=2) if result['aligned_structure'] else "Failed to parse")
    
    return result

if __name__ == "__main__":
    test_common_ground_agent()
