from utils.sql_manager import SQLManager
from typing import Optional


class EmailRulesManager:
    """
    Manages email classification rules and agent instructions
    Stores and retrieves rules from SQL database with versioning
    """
    
    def __init__(self, sql_manager: SQLManager, user_id: int):
        self.sql_manager = sql_manager
        self.user_id = user_id
    
    def get_rules(self, rule_type: str) -> str:
        """
        Get latest rules for a specific type
        
        Args:
            rule_type: 'ignore_rules', 'notify_rules', 'respond_rules', or 'agent_instructions'
        
        Returns:
            str: Rule content or default if not found
        """
        query = """
            SELECT rule_content FROM email_rules
            WHERE rule_type = ? AND user_id = ?
            ORDER BY version DESC LIMIT 1
        """
        result = self.sql_manager.execute_query(
            query, (rule_type, self.user_id), fetch_one=True
        )
        
        if result:
            return result[0]
        else:
            # Return default rules if not found
            return self._get_default_rules(rule_type)
    
    def update_rules(self, rule_type: str, rule_content: str) -> int:
        """
        Update rules with new version
        
        Args:
            rule_type: 'ignore_rules', 'notify_rules', 'respond_rules', or 'agent_instructions'
            rule_content: New rule content
        
        Returns:
            int: New version number
        """
        # Get current version
        query = """
            SELECT MAX(version) FROM email_rules 
            WHERE rule_type = ? AND user_id = ?
        """
        result = self.sql_manager.execute_query(
            query, (rule_type, self.user_id), fetch_one=True
        )
        next_version = (result[0] or 0) + 1
        
        # Insert new version
        query = """
            INSERT INTO email_rules (rule_type, rule_content, version, user_id)
            VALUES (?, ?, ?, ?)
        """
        self.sql_manager.execute_query(
            query, (rule_type, rule_content, next_version, self.user_id)
        )
        print(f"✅ Updated {rule_type} to version {next_version}")
        return next_version
    
    def get_all_rules(self) -> dict:
        """
        Get all current rules
        
        Returns:
            dict: {rule_type: rule_content}
        """
        query = """
            SELECT rule_type, rule_content FROM email_rules
            WHERE user_id = ?
            AND (rule_type, version) IN (
                SELECT rule_type, MAX(version)
                FROM email_rules
                WHERE user_id = ?
                GROUP BY rule_type
            )
        """
        results = self.sql_manager.execute_query(
            query, (self.user_id, self.user_id), fetch_all=True
        )
        
        rules_dict = {}
        for rule_type, rule_content in results:
            rules_dict[rule_type] = rule_content
        
        # Fill in defaults for missing rules
        all_types = ['ignore_rules', 'notify_rules', 'respond_rules', 'agent_instructions']
        for rule_type in all_types:
            if rule_type not in rules_dict:
                rules_dict[rule_type] = self._get_default_rules(rule_type)
        
        return rules_dict
    
    def get_rule_version(self, rule_type: str) -> int:
        """
        Get current version number for a rule type
        
        Args:
            rule_type: Rule type to check
        
        Returns:
            int: Current version number
        """
        query = """
            SELECT MAX(version) FROM email_rules 
            WHERE rule_type = ? AND user_id = ?
        """
        result = self.sql_manager.execute_query(
            query, (rule_type, self.user_id), fetch_one=True
        )
        return result[0] if result and result[0] else 0
    
    def _get_default_rules(self, rule_type: str) -> str:
        """
        Get default rules if none exist in database
        
        Args:
            rule_type: Type of rule
        
        Returns:
            str: Default rule content
        """
        defaults = {
            'ignore_rules': 'Spam, promotional emails, mass announcements, no action needed',
            'notify_rules': 'Important information that user should know but doesn\'t need a response (e.g., system notifications, status updates)',
            'respond_rules': 'Emails that need a direct response (e.g., questions, meeting requests, action items)',
            'agent_instructions': 'Use these tools when appropriate to help manage tasks efficiently. Generate professional, courteous email responses.'
        }
        return defaults.get(rule_type, '')

