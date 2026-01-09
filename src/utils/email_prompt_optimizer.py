import os
import json
from openai import OpenAI
from utils.config import Config
from utils.email_rules_manager import EmailRulesManager
from utils.sql_manager import SQLManager


class EmailPromptOptimizer:
    """
    Optimizes email classification rules and agent instructions based on user feedback
    Similar to given code's multi-prompt optimizer
    """
    
    def __init__(self, config: Config, sql_manager: SQLManager, user_id: int):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cfg = config
        self.rules_manager = EmailRulesManager(sql_manager, user_id)
    
    def optimize_rules(self, feedback: dict) -> dict:
        """
        Optimize rules based on user feedback
        
        Args:
            feedback: {
                'conversation': list,  # Email processing conversation/messages
                'feedback_text': str,  # User feedback
                'rules_to_update': list[str]  # Which rules to update (optional, defaults to all)
            }
        
        Returns:
            dict: Updated rules {rule_name: updated_content}
        """
        # Determine which rules to update
        rules_to_update = feedback.get('rules_to_update', [
            'ignore_rules', 'notify_rules', 'respond_rules', 'agent_instructions'
        ])
        
        # Get current rules
        prompts_to_optimize = []
        for rule_type in rules_to_update:
            current_rule = self.rules_manager.get_rules(rule_type)
            
            prompts_to_optimize.append({
                "name": rule_type,
                "prompt": current_rule,
                "update_instructions": "Keep the rules concise, actionable, and specific",
                "when_to_update": f"Update when feedback relates to {rule_type}"
            })
        
        # Build optimization prompt
        optimization_prompt = f"""You are optimizing email classification rules and agent instructions based on user feedback.

## Current Rules:

{json.dumps({p['name']: p['prompt'] for p in prompts_to_optimize}, indent=2)}

## User Feedback:

{feedback.get('feedback_text', 'No specific feedback provided')}

## Conversation Context:

{json.dumps(feedback.get('conversation', []), indent=2) if feedback.get('conversation') else 'No conversation context provided'}

## Task:

Analyze the feedback and conversation context. For each rule that needs improvement based on the feedback, provide an optimized version.

Guidelines:
- Only update rules that clearly need improvement based on the feedback
- Keep updates minimal and focused
- Maintain clarity and specificity
- Preserve the core purpose of each rule

Return a JSON object with updated rules:
{{
    "rule_name": "updated_rule_content",
    ...
}}

Only include rules that need updating. If a rule doesn't need changes, don't include it in the response."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.cfg.chat_model,
                messages=[{"role": "user", "content": optimization_prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            optimized = json.loads(response.choices[0].message.content)
            
            # Update rules in database
            updated_rules = {}
            for rule_name, rule_content in optimized.items():
                if rule_name in ['ignore_rules', 'notify_rules', 'respond_rules', 'agent_instructions']:
                    version = self.rules_manager.update_rules(rule_name, rule_content)
                    updated_rules[rule_name] = {
                        'content': rule_content,
                        'version': version
                    }
                    print(f"✅ Updated {rule_name} to version {version}")
            
            return updated_rules
            
        except Exception as e:
            print(f"Error optimizing rules: {e}")
            return {}
    
    def optimize_single_rule(self, rule_type: str, feedback_text: str, conversation: list = None) -> dict:
        """
        Optimize a single rule type
        
        Args:
            rule_type: 'ignore_rules', 'notify_rules', 'respond_rules', or 'agent_instructions'
            feedback_text: User feedback
            conversation: Optional conversation context
        
        Returns:
            dict: Updated rule info
        """
        return self.optimize_rules({
            'conversation': conversation or [],
            'feedback_text': feedback_text,
            'rules_to_update': [rule_type]
        })
    
    def get_optimization_suggestions(self, feedback: dict) -> dict:
        """
        Get suggestions for rule improvements without updating
        
        Args:
            feedback: Same format as optimize_rules
        
        Returns:
            dict: Suggestions for each rule
        """
        rules_to_check = feedback.get('rules_to_update', [
            'ignore_rules', 'notify_rules', 'respond_rules', 'agent_instructions'
        ])
        
        current_rules = {}
        for rule_type in rules_to_check:
            current_rules[rule_type] = self.rules_manager.get_rules(rule_type)
        
        suggestion_prompt = f"""Analyze the following rules and user feedback to suggest improvements.

Current Rules:
{json.dumps(current_rules, indent=2)}

User Feedback:
{feedback.get('feedback_text', '')}

Conversation Context:
{json.dumps(feedback.get('conversation', []), indent=2) if feedback.get('conversation') else 'None'}

Provide suggestions for improvement in JSON format:
{{
    "rule_name": {{
        "suggestion": "what to improve",
        "reason": "why this improvement is needed",
        "improved_version": "suggested improved rule"
    }}
}}

Only suggest improvements where feedback indicates a need."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.cfg.chat_model,
                messages=[{"role": "user", "content": suggestion_prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            suggestions = json.loads(response.choices[0].message.content)
            return suggestions
            
        except Exception as e:
            print(f"Error getting suggestions: {e}")
            return {}

