import os
import uuid
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
from utils.config import Config
from utils.email_rules_manager import EmailRulesManager
from utils.sql_manager import SQLManager


class EmailTriageManager:
    """
    Email Triage Manager - Classifies emails using VectorDB with few-shot examples
    Separate VectorDB collections for each category: ignore, notify, respond
    """
    
    def __init__(self, config: Config, sql_manager: SQLManager = None, user_id: int = None):
        self.cfg = config
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=self.cfg.embedding_model
        )
        
        # Initialize ChromaDB client
        self.db_client = chromadb.PersistentClient(
            path=str(self.cfg.vectordb_dir)
        )
        
        # Create separate collections for each category
        self.collections = {
            'ignore': self.db_client.get_or_create_collection(
                name="email_triage_ignore",
                embedding_function=self.embedding_function
            ),
            'notify': self.db_client.get_or_create_collection(
                name="email_triage_notify",
                embedding_function=self.embedding_function
            ),
            'respond': self.db_client.get_or_create_collection(
                name="email_triage_respond",
                embedding_function=self.embedding_function
            )
        }
        
        # Initialize rules manager if provided
        self.rules_manager = None
        if sql_manager and user_id:
            self.rules_manager = EmailRulesManager(sql_manager, user_id)
    
    def add_few_shot_example(self, category: str, email_data: dict, label: str):
        """
        Add few-shot example to the specified category's VectorDB
        
        Args:
            category: 'ignore', 'notify', or 'respond'
            email_data: Email data dict with sender, subject, body
            label: The classification label
        """
        if category not in self.collections:
            raise ValueError(f"Invalid category: {category}. Must be 'ignore', 'notify', or 'respond'")
        
        # Format email for storage
        email_text = f"From: {email_data.get('sender', '')}\nSubject: {email_data.get('subject', '')}\nBody: {email_data.get('body', '')}"
        
        # Store with label
        document = f"{email_text}\n\nLabel: {label}"
        
        self.collections[category].add(
            ids=str(uuid.uuid4()),
            documents=[document],
            metadatas=[{"label": label, "category": category}]
        )
        print(f"✅ Added few-shot example to '{category}' category")
    
    def classify_email(self, email_data: dict) -> tuple[str, float]:
        """
        Classify email by searching VectorDB collections and finding most similar category
        
        Args:
            email_data: Email data dict with sender, subject, body
            
        Returns:
            tuple: (classification, confidence_score)
            classification: 'ignore', 'notify', or 'respond'
            confidence_score: Similarity score (0-1)
        """
        # Format email for search
        email_text = f"From: {email_data.get('sender', '')}\nSubject: {email_data.get('subject', '')}\nBody: {email_data.get('body', '')}"
        
        category_scores = {}
        
        # Search each category collection
        for category, collection in self.collections.items():
            try:
                # Get count of examples in this category
                count = collection.count()
                
                if count == 0:
                    # No examples yet, skip this category
                    category_scores[category] = 0.0
                    continue
                
                # Search for similar emails in this category
                results = collection.query(
                    query_texts=[email_text],
                    n_results=min(3, count)  # Get top 3 similar examples
                )
                
                # Calculate average similarity score
                if results['distances'] and len(results['distances'][0]) > 0:
                    # ChromaDB returns distances (lower is better), convert to similarity
                    distances = results['distances'][0]
                    similarities = [1 - d for d in distances]  # Convert distance to similarity
                    avg_similarity = sum(similarities) / len(similarities)
                    category_scores[category] = avg_similarity
                else:
                    category_scores[category] = 0.0
                    
            except Exception as e:
                print(f"Error searching {category} category: {e}")
                category_scores[category] = 0.0
        
        # If no examples exist in any category, use LLM for classification
        if all(score == 0.0 for score in category_scores.values()):
            print("⚠️ No few-shot examples found. Using LLM for classification...")
            return self._classify_with_llm(email_data), 0.5
        
        # Find category with highest similarity
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]
        
        # If confidence is too low, use LLM as fallback (use configurable threshold)
        confidence_threshold = getattr(self.cfg, 'classification_confidence_threshold', 0.37)
        if best_score < confidence_threshold:
            print(f"⚠️ Low confidence ({best_score:.2f} < {confidence_threshold}). Using LLM for classification...")
            llm_classification = self._classify_with_llm(email_data)
            # Validate LLM classification result
            if llm_classification not in ['ignore', 'notify', 'respond']:
                print(f"⚠️ Invalid LLM classification '{llm_classification}', using best category '{best_category}'")
                return best_category, best_score
            return llm_classification, best_score
        
        print(f"📧 Classified as '{best_category}' with confidence {best_score:.2f}")
        return best_category, best_score
    
    def _classify_with_llm(self, email_data: dict) -> str:
        """
        Fallback: Use LLM for classification when no examples or low confidence
        Uses dynamic rules from database if available
        """
        # Load rules from database if rules_manager is available
        if self.rules_manager:
            ignore_rules = self.rules_manager.get_rules('ignore_rules')
            notify_rules = self.rules_manager.get_rules('notify_rules')
            respond_rules = self.rules_manager.get_rules('respond_rules')
        else:
            # Default rules if rules_manager not available
            ignore_rules = 'Spam, promotional emails, mass announcements, no action needed'
            notify_rules = 'Important information that user should know but doesn\'t need a response (e.g., system notifications, status updates)'
            respond_rules = 'Emails that need a direct response (e.g., questions, meeting requests, action items)'
        
        classification_prompt = f"""Classify this email into one category based on the following rules:

IGNORE rules: {ignore_rules}

NOTIFY rules: {notify_rules}

RESPOND rules: {respond_rules}

Email:
From: {email_data.get('sender', '')}
Subject: {email_data.get('subject', '')}
Body: {email_data.get('body', '')[:500]}

Respond with ONLY one word: ignore, notify, or respond"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.cfg.chat_model,
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=0.0
            )
            
            classification = response.choices[0].message.content.strip().lower()
            
            # Validate classification
            if classification in ['ignore', 'notify', 'respond']:
                return classification
            else:
                print(f"⚠️ Invalid classification '{classification}', defaulting to 'respond'")
                return 'respond'
        except Exception as e:
            print(f"Error in LLM classification: {e}")
            return 'respond'  # Default to respond
    
    def get_few_shot_examples(self, category: str, limit: int = 5) -> list:
        """
        Get few-shot examples for a category
        
        Args:
            category: 'ignore', 'notify', or 'respond'
            limit: Number of examples to retrieve
            
        Returns:
            list: List of example documents
        """
        if category not in self.collections:
            return []
        
        try:
            # Get all examples (we'll get random ones)
            count = self.collections[category].count()
            if count == 0:
                return []
            
            # Get examples
            results = self.collections[category].get(limit=min(limit, count))
            return results['documents'] if results else []
        except Exception as e:
            print(f"Error getting examples for {category}: {e}")
            return []

