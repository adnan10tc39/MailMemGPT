from typing import Optional, List
from openai import OpenAI
from utils.sql_manager import SQLManager
from utils.utils import Utils
import json


class EmailHistoryManager:
    """
    Manages email history and summarization - same concept as ChatHistoryManager
    Uses Chatbot v3's long-term memory approach for emails
    """

    def __init__(self, sql_manager: SQLManager, user_id: str, session_id: str, client: OpenAI, summary_model: str, max_tokens: int) -> None:
        self.utils = Utils()
        self.client = client
        self.summary_model = summary_model
        self.max_tokens = max_tokens
        self.sql_manager = sql_manager
        self.user_id = user_id
        self.session_id = session_id
        self.email_history = []  # Same as chat_history but for emails
        self.pairs_since_last_summary = 0

    def add_to_history(self, email_data: dict, response_text: str, max_history_pairs: int) -> None:
        """
        Adds email and response to history - same as ChatHistoryManager.add_to_history
        """
        # Format: {"email": {...}, "response": "..."}
        self.email_history.append({"email": email_data})
        self.email_history.append({"response": response_text})

        if len(self.email_history) > max_history_pairs * 2:
            self.email_history = self.email_history[-max_history_pairs * 2:]

        self.save_to_db(email_data, response_text)
        self.pairs_since_last_summary += 1
        print("Email history saved to database.")
        
        # Token limit check - same as ChatHistoryManager
        email_history_token_count = self.utils.count_number_of_tokens(
            str(self.email_history))
        if email_history_token_count > self.max_tokens:
            print("Summarizing the email history ...")
            print("\nOld number of tokens:", email_history_token_count)

            self.summarize_email_history()

            email_history_token_count = self.utils.count_number_of_tokens(
                str(self.email_history))
            print("\nNew number of tokens:", email_history_token_count)

    def save_to_db(self, email_data: dict, response_text: str) -> None:
        """
        Saves email and response to database - same concept as ChatHistoryManager.save_to_db
        """
        if not self.user_id:
            print("Error: No user found in the database.")
            return
        query = """
            INSERT INTO email_history (user_id, email_id, sender, subject, email_body, response_text, session_id, classification, classification_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.sql_manager.execute_query(
            query, (
                self.user_id,
                email_data.get('email_id', ''),
                email_data.get('sender', ''),
                email_data.get('subject', ''),
                email_data.get('body', ''),
                response_text,
                self.session_id,
                email_data.get('classification'),
                email_data.get('classification_confidence')
            )
        )

    def get_latest_email_pairs(self, num_pairs: int) -> List[tuple]:
        """
        Fetches latest email pairs - same as ChatHistoryManager.get_latest_chat_pairs
        """
        query = """
            SELECT email_body, response_text FROM email_history
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?;
        """
        email_data = self.sql_manager.execute_query(
            query, (self.session_id, num_pairs * 2), fetch_all=True)
        return list(reversed(email_data))

    def get_latest_summary(self) -> Optional[str]:
        """
        Retrieves latest summary - same as ChatHistoryManager.get_latest_summary
        Uses same summary table
        """
        query = """
            SELECT summary_text FROM summary
            WHERE session_id = ? ORDER BY timestamp DESC LIMIT 1;
        """
        summary = self.sql_manager.execute_query(
            query, (self.session_id,), fetch_one=True)
        return summary[0] if summary else None

    def save_summary_to_db(self, summary_text: str) -> None:
        """
        Saves summary - same as ChatHistoryManager.save_summary_to_db
        """
        if not self.user_id or not summary_text:
            return
        query = """
            INSERT INTO summary (user_id, session_id, summary_text)
            VALUES (?, ?, ?);
        """
        self.sql_manager.execute_query(
            query, (self.user_id, self.session_id, summary_text))
        print("Summary saved to database.")

    def update_email_summary(self, max_history_pairs: int) -> None:
        """
        Updates email summary - same as ChatHistoryManager.update_chat_summary
        """
        print("pairs_since_last_summary:", self.pairs_since_last_summary)
        if self.pairs_since_last_summary < max_history_pairs:
            return None
        
        email_data = self.get_latest_email_pairs(max_history_pairs)
        previous_summary = self.get_latest_summary()

        if len(email_data) <= max_history_pairs:
            print("Insufficient email data. Skipping summary.")
            return

        summary_text = self.generate_the_new_summary(
            self.client, self.summary_model, email_data, previous_summary)

        if summary_text:
            self.save_summary_to_db(summary_text)
            self.pairs_since_last_summary = 0
            print("Email history summary generated and saved to database.")

    def generate_the_new_summary(
        self,
        client: OpenAI,
        summary_model: str,
        email_data: List[tuple],
        previous_summary: Optional[str]
    ) -> Optional[str]:
        """
        Generates summary - same as ChatHistoryManager.generate_the_new_summary
        """
        if not email_data:
            return None

        summary_prompt = "Summarize the following email conversations:\n\n"

        if previous_summary:
            summary_prompt += f"Previous summary:\n{previous_summary}\n\n"

        for email_body, response in email_data:
            summary_prompt += f"Email: {email_body}\nResponse: {response}\n\n"

        summary_prompt += "Provide a concise summary while keeping important details."

        try:
            response = client.chat.completions.create(
                model=summary_model,
                messages=[{"role": "system", "content": summary_prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return None

    def summarize_email_history(self):
        """
        Summarizes older email history - same as ChatHistoryManager.summarize_chat_history
        """
        pairs_to_keep = 1
        pairs_to_summarize = self.email_history[:-pairs_to_keep * 2]

        if len(pairs_to_summarize) == 0:
            return

        prompt = f"""
        Summarize the following email conversations while preserving key details:
        {pairs_to_summarize}

        Return the summarized conversations (in JSON format with 'email' and 'response' pairs):
        """
        try:
            response = self.client.chat.completions.create(
                model=self.summary_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            summarized_pairs = response.choices[0].message.content
            summarized_pairs = json.loads(summarized_pairs)

            if isinstance(summarized_pairs, dict):
                summarized_pairs = [summarized_pairs]

            if isinstance(summarized_pairs, list) and all(
                isinstance(pair, dict) and 'email' in pair and 'response' in pair
                for pair in summarized_pairs
            ):
                self.email_history = summarized_pairs + \
                    self.email_history[-pairs_to_keep * 2:]
                print("Email history summarized.")
            else:
                raise ValueError("Invalid format received from LLM.")

        except Exception as e:
            print(f"Failed to summarize email history: {e}")

