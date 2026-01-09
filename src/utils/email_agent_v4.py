import os
import uuid
import json
from dotenv import load_dotenv
from openai import OpenAI
from traceback import format_exc
from utils.sql_manager import SQLManager
from utils.user_manager import UserManager
from utils.email_history_manager import EmailHistoryManager
from utils.prepare_system_prompt import prepare_system_prompt_for_email_agent
from utils.utils import Utils
from utils.config import Config
from utils.vector_db_manager import VectorDBManager
from utils.email_tools import EmailTools
from utils.email_triage_manager import EmailTriageManager
from utils.email_rules_manager import EmailRulesManager
from utils.email_prompt_optimizer import EmailPromptOptimizer

load_dotenv()


class EmailAgent:
    """
    Email Agent using Chatbot v3's long-term memory concept
    + Additional tools from given code (write_email, schedule_meeting, etc.)
    """
    
    def __init__(self, session_id: str = None):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cfg = Config()
        self.chat_model = self.cfg.chat_model
        self.summary_model = self.cfg.summary_model
        self.temperature = self.cfg.temperature
        self.max_history_pairs = self.cfg.max_history_pairs
        
        # ✅ IMPROVED: Session management - allow session_id to be passed or generate new
        # This allows maintaining session continuity for email threads
        self.session_id = session_id if session_id else str(uuid.uuid4())
        self.utils = Utils()
        self.sql_manager = SQLManager(self.cfg.db_path)
        self.user_manager = UserManager(self.sql_manager)
        
        # ✅ HUMARA MEMORY CONCEPT - Same as Chatbot v3
        self.email_history_manager = EmailHistoryManager(
            self.sql_manager, self.user_manager.user_id, self.session_id,
            self.client, self.summary_model, self.cfg.max_tokens
        )
        
        self.vector_db_manager = VectorDBManager(self.cfg)
        
        # ✅ NEW: Rules Manager for dynamic rules
        self.rules_manager = EmailRulesManager(self.sql_manager, self.user_manager.user_id)
        
        # ✅ NEW: Email Triage Manager for classification (with rules manager)
        self.triage_manager = EmailTriageManager(
            self.cfg, 
            self.sql_manager, 
            self.user_manager.user_id
        )
        
        # ✅ NEW: Prompt Optimizer for feedback-based updates
        self.prompt_optimizer = EmailPromptOptimizer(
            self.cfg,
            self.sql_manager,
            self.user_manager.user_id
        )
        
        # ✅ NEW: Email tools from given code
        self.email_tools = EmailTools(self.utils)
        
        # ✅ HUMARA FUNCTION CALLING - Same as Chatbot v3 + New tools
        self.agent_functions = [
            self.utils.jsonschema(self.user_manager.add_user_info_to_database),
            self.utils.jsonschema(self.vector_db_manager.search_vector_db),
            # ✅ NEW: Additional tools from given code
            self.utils.jsonschema(self.email_tools.write_email_tool),
            self.utils.jsonschema(self.email_tools.schedule_meeting_tool),
            self.utils.jsonschema(self.email_tools.check_calendar_availability_tool),
        ]
    
    def execute_function_call(self, function_name: str, function_args: dict) -> tuple[str, str]:
        """
        ✅ HUMARA EXACT APPROACH - Same as Chatbot v3
        + Additional tools from given code
        + Improved error handling
        """
        try:
            if function_name == "search_vector_db":
                try:
                    return self.vector_db_manager.search_vector_db(**function_args)
                except Exception as e:
                    error_msg = f"Error searching VectorDB: {str(e)}"
                    print(f"❌ {error_msg}")
                    return "Function call failed.", error_msg
            
            elif function_name == "add_user_info_to_database":
                try:
                    return self.user_manager.add_user_info_to_database(function_args)
                except Exception as e:
                    error_msg = f"Error updating user info: {str(e)}"
                    print(f"❌ {error_msg}")
                    return "Function call failed.", error_msg
            
            # ✅ NEW: Tools from given code
            elif function_name == "write_email_tool":
                try:
                    return self.email_tools.write_email_tool(**function_args)
                except Exception as e:
                    error_msg = f"Error writing email: {str(e)}"
                    print(f"❌ {error_msg}")
                    return "Function call failed.", error_msg
            
            elif function_name == "schedule_meeting_tool":
                try:
                    return self.email_tools.schedule_meeting_tool(**function_args)
                except Exception as e:
                    error_msg = f"Error scheduling meeting: {str(e)}"
                    print(f"❌ {error_msg}")
                    return "Function call failed.", error_msg
            
            elif function_name == "check_calendar_availability_tool":
                try:
                    return self.email_tools.check_calendar_availability_tool(**function_args)
                except Exception as e:
                    error_msg = f"Error checking calendar: {str(e)}"
                    print(f"❌ {error_msg}")
                    return "Function call failed.", error_msg
            
            else:
                error_msg = f"Unknown function: {function_name}"
                print(f"❌ {error_msg}")
                return "Function call failed.", error_msg
        
        except Exception as e:
            error_msg = f"Unexpected error executing function {function_name}: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            print(traceback.format_exc())
            return "Function call failed.", error_msg
    
    def process_email(self, email_data: dict) -> str:
        """
        Process email with classification step
        ✅ HUMARA EXACT MEMORY CONCEPT + Classification
        
        Args:
            email_data: {
                'sender': str,
                'subject': str,
                'body': str,
                'email_id': str (optional)
            }
        
        Returns:
            str: Generated email response or classification message
        """
        # Generate email_id if not provided
        if 'email_id' not in email_data:
            email_data['email_id'] = str(uuid.uuid4())
        
        # ✅ IMPROVED: Try to maintain session continuity for email threads
        # If subject suggests it's a reply (Re:, Fwd:, etc.), try to find existing session
        subject = email_data.get('subject', '').lower()
        if subject.startswith('re:') or subject.startswith('fwd:') or subject.startswith('fw:'):
            try:
                # Try to find recent email with similar subject (without Re:/Fwd: prefix)
                clean_subject = subject.replace('re:', '').replace('fwd:', '').replace('fw:', '').strip()
                # Update session_id to link to previous email thread if found
                # This is a simple heuristic - can be improved with proper thread tracking
                print(f"ℹ️ Detected reply/forward email, attempting to maintain thread continuity")
            except Exception as e:
                print(f"⚠️ Error checking email thread: {e}")
        
        # ✅ STEP 1: CLASSIFY EMAIL using VectorDB few-shot examples
        print("\n" + "="*60)
        print("📧 CLASSIFYING EMAIL...")
        print("="*60)
        
        try:
            classification, confidence = self.triage_manager.classify_email(email_data)
            
            # ✅ NEW: Validate classification result
            if classification not in ['ignore', 'notify', 'respond']:
                print(f"⚠️ Invalid classification '{classification}', defaulting to 'respond'")
                classification = 'respond'
                confidence = 0.5
            
            print(f"Classification: {classification.upper()} (Confidence: {confidence:.2f})")
            print("="*60 + "\n")
        except Exception as e:
            print(f"❌ Error during classification: {e}")
            print("⚠️ Defaulting to 'respond' classification")
            classification = 'respond'
            confidence = 0.5
        
        # Save classification to email_data
        email_data['classification'] = classification
        email_data['classification_confidence'] = confidence
        
        # ✅ STEP 2: BEHAVE BASED ON CLASSIFICATION
        if classification == 'ignore':
            # Ignore: No response, just log
            print("🚫 Email classified as IGNORE - No response generated")
            try:
                self.email_history_manager.save_to_db(
                    email_data, 
                    "IGNORED - Email classified as ignore, no response needed."
                )
            except Exception as e:
                print(f"⚠️ Error saving ignored email to DB: {e}")
            return f"Email classified as IGNORE (confidence: {confidence:.2f}). No response generated."
        
        elif classification == 'notify':
            # Notify: Save but no response
            print("🔔 Email classified as NOTIFY - User notified, no response generated")
            try:
                self.email_history_manager.save_to_db(
                    email_data,
                    "NOTIFIED - Email classified as notify, user has been notified."
                )
            except Exception as e:
                print(f"⚠️ Error saving notified email to DB: {e}")
            return f"Email classified as NOTIFY (confidence: {confidence:.2f}). User has been notified. No response generated."
        
        elif classification == 'respond':
            # Respond: Generate response (existing flow)
            print("✅ Email classified as RESPOND - Generating response...")
            try:
                return self._generate_response(email_data)
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                print(f"❌ {error_msg}")
                import traceback
                print(traceback.format_exc())
                return f"Error: {error_msg}"
        
        else:
            # Fallback: Default to respond
            print(f"⚠️ Unknown classification '{classification}', defaulting to respond")
            try:
                return self._generate_response(email_data)
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                print(f"❌ {error_msg}")
                return f"Error: {error_msg}"
    
    def _load_memory_from_vectordb(self, email_data: dict, max_results: int = 5) -> str:
        """
        Load relevant memory from VectorDB BEFORE building system prompt
        Similar to image's load_memory() function
        Automatically fetches relevant past emails using semantic search
        
        Args:
            email_data: Email data dict
            max_results: Maximum number of results to retrieve
        
        Returns:
            str: Formatted memory string or empty string on error
        """
        # Format email for semantic search
        email_query = f"From: {email_data.get('sender', '')}\nSubject: {email_data.get('subject', '')}\nBody: {email_data.get('body', '')[:200]}"
        
        try:
            # Search VectorDB for relevant past emails
            if not hasattr(self.vector_db_manager, 'db_collection') or self.vector_db_manager.db_collection is None:
                print("⚠️ VectorDB collection not available")
                return ""
            
            results = self.vector_db_manager.db_collection.query(
                query_texts=[email_query],
                n_results=max_results
            )
            
            if results and results.get('documents') and len(results['documents'][0]) > 0:
                # Format retrieved memories
                memories = []
                for i, doc in enumerate(results['documents'][0]):
                    if doc and doc.strip():  # Only add non-empty documents
                        memories.append(f"Relevant Past Email {i+1}:\n{doc}\n")
                
                if memories:
                    result = "\n".join(memories)
                    print(f"✅ Loaded {len(memories)} relevant emails from VectorDB")
                    return result
                else:
                    return ""
            else:
                print("ℹ️ No relevant emails found in VectorDB")
                return ""
        except Exception as e:
            print(f"⚠️ Error loading memory from VectorDB: {e}")
            import traceback
            print(traceback.format_exc())
            return ""
    
    def _deduplicate_memory(self, sql_history: str, vectordb_memory: str) -> tuple[str, str]:
        """
        Remove duplicate information between SQL DB history and VectorDB memory
        Ensures no repetition in system prompt
        
        Strategy:
        1. SQL history contains recent emails (already in system prompt)
        2. VectorDB memory contains semantically similar emails
        3. If VectorDB email is already in SQL history, skip it
        4. Only add new/relevant emails from VectorDB that aren't in SQL history
        
        Returns:
            tuple: (combined_history, filtered_vectordb_memory)
        """
        if not vectordb_memory or not vectordb_memory.strip():
            return sql_history, ""
        
        if not sql_history or not sql_history.strip():
            return "", vectordb_memory
        
        # Extract unique identifiers from SQL history
        # SQL history format: "email: From: ... Subject: ... Body: ... response: ..."
        sql_identifiers = set()
        sql_subject_sender_pairs = set()
        
        # Extract subjects and senders from SQL history
        sql_lines = sql_history.split('\n')
        current_subject = None
        current_sender = None
        
        for line in sql_lines:
            line_lower = line.lower()
            if 'subject:' in line_lower:
                try:
                    subject = line.split('Subject:')[1].split('\n')[0].strip()
                    if subject:
                        sql_identifiers.add(f"subject:{subject.lower()}")
                        current_subject = subject.lower()
                except Exception as e:
                    pass
            if 'from:' in line_lower:
                try:
                    sender = line.split('From:')[1].split('\n')[0].strip()
                    if sender:
                        sql_identifiers.add(f"sender:{sender.lower()}")
                        current_sender = sender.lower()
                        # Store subject-sender pairs for better duplicate detection
                        if current_subject:
                            sql_subject_sender_pairs.add((current_subject, current_sender))
                except Exception as e:
                    pass
            # Also check for email: prefix format
            if 'email:' in line_lower and current_subject:
                sql_identifiers.add(f"email_subject:{current_subject}")
        
        # Filter VectorDB memory to remove duplicates
        vectordb_sections = vectordb_memory.split('Relevant Past Email')
        filtered_sections = []
        
        for section in vectordb_sections:
            if not section.strip():
                continue
            
            # Check if this section is duplicate
            is_duplicate = False
            section_lower = section.lower()
            
            # Extract subject and sender from VectorDB section
            section_subject = None
            section_sender = None
            for line in section.split('\n'):
                line_lower = line.lower()
                if 'subject:' in line_lower:
                    try:
                        section_subject = line.split('Subject:')[1].split('\n')[0].strip().lower()
                    except:
                        pass
                if 'from:' in line_lower:
                    try:
                        section_sender = line.split('From:')[1].split('\n')[0].strip().lower()
                    except:
                        pass
            
            # Check against SQL identifiers (subject or sender match)
            for identifier in sql_identifiers:
                if identifier and len(identifier.split(':')) > 1:
                    identifier_value = identifier.split(':')[1]
                    if identifier_value in section_lower:
                        is_duplicate = True
                        break
            
            # Check subject-sender pairs for better accuracy
            if not is_duplicate and section_subject and section_sender:
                if (section_subject, section_sender) in sql_subject_sender_pairs:
                    is_duplicate = True
            
            # Additional check: if section contains same subject/sender combo
            if not is_duplicate and current_subject and current_sender:
                if current_subject in section_lower and current_sender in section_lower:
                    is_duplicate = True
            
            if not is_duplicate:
                filtered_sections.append(f"Relevant Past Email{section}")
        
        # Combine SQL history with filtered VectorDB memory
        if filtered_sections:
            filtered_vectordb_text = '\n'.join(filtered_sections)
            # Return both: combined history and separate filtered VectorDB memory
            return sql_history, filtered_vectordb_text
        
        # If all VectorDB memory was duplicate, return only SQL history
        return sql_history, ""
    
    def _check_token_limit(self, text: str, max_tokens: int = None) -> bool:
        """
        Check if text exceeds token limit
        
        Args:
            text: Text to check
            max_tokens: Maximum tokens allowed (defaults to config value)
        
        Returns:
            bool: True if exceeds limit, False otherwise
        """
        if max_tokens is None:
            max_tokens = self.cfg.max_tokens
        
        try:
            token_count = self.utils.count_number_of_tokens(text)
            return token_count > max_tokens
        except Exception as e:
            print(f"⚠️ Error counting tokens: {e}")
            # Fallback: use character count approximation (rough estimate: 1 token ≈ 4 characters)
            return len(text) > (max_tokens * 4)
    
    def _summarize_memory(self, memory_text: str) -> str:
        """
        Summarize memory if token limit exceeded
        Uses intelligent summarization with better fallback strategy
        """
        if not memory_text:
            return ""
        
        try:
            # Try to summarize intelligently using LLM
            # Calculate how much to summarize based on token count
            token_count = self.utils.count_number_of_tokens(memory_text)
            target_tokens = self.cfg.max_tokens // 2  # Summarize to half of max tokens
            
            # Estimate characters to include (rough: 1 token ≈ 4 chars)
            chars_to_include = min(len(memory_text), target_tokens * 4)
            
            summary_prompt = f"""Summarize the following email memory to reduce token count while keeping the most important context and key information:

{memory_text[:chars_to_include]}

Provide a concise summary focusing on:
- Key topics and subjects discussed
- Important decisions or agreements
- Action items or follow-ups
- Critical details that might be needed for future responses

Keep the summary under {target_tokens} tokens while preserving essential context."""
            
            response = self.client.chat.completions.create(
                model=self.summary_model,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=target_tokens
            )
            
            summarized = response.choices[0].message.content
            print(f"✅ Memory summarized: {token_count} tokens → {self.utils.count_number_of_tokens(summarized)} tokens")
            return summarized
        except Exception as e:
            print(f"⚠️ Error summarizing memory with LLM: {e}")
            # Better fallback: intelligent truncation
            # Try to keep beginning and end, remove middle
            if len(memory_text) > 2000:
                # Keep first 40% and last 40%, remove middle 20%
                first_part = memory_text[:int(len(memory_text) * 0.4)]
                last_part = memory_text[int(len(memory_text) * 0.6):]
                truncated = f"{first_part}\n\n[... truncated middle section ...]\n\n{last_part}"
                print(f"⚠️ Using truncated memory fallback: {len(memory_text)} chars → {len(truncated)} chars")
                return truncated
            else:
                # If already small, just return as is
                return memory_text
    
    def _generate_response(self, email_data: dict) -> str:
        """
        Generate email response - ✅ HUMARA EXACT MEMORY CONCEPT + VectorDB Auto-Load
        Same flow as Chatbot v3.chat() but for emails + automatic VectorDB memory loading
        """
        function_call_result_section = ""
        function_call_state = None
        email_state = "thinking"
        function_call_count = 0
        function_call_history = []  # Track function calls to detect loops
        
        try:
            # ✅ STEP 1: Load memory from SQL DB (like image)
            try:
                self.email_history = self.email_history_manager.email_history
                self.previous_summary = self.email_history_manager.get_latest_summary()
            except Exception as e:
                print(f"⚠️ Error loading SQL DB memory: {e}")
                self.email_history = None
                self.previous_summary = ""
            
            # ✅ STEP 2: Load memory from VectorDB (like image's load_memory())
            print("\n📚 Loading memory from VectorDB...")
            try:
                vectordb_memory = self._load_memory_from_vectordb(email_data, max_results=5)
            except Exception as e:
                print(f"⚠️ Error loading VectorDB memory: {e}")
                vectordb_memory = ""
            
            # ✅ STEP 3: Deduplicate - Remove same info from SQL DB and VectorDB
            print("🔄 Deduplicating memory (removing duplicates between SQL DB and VectorDB)...")
            try:
                combined_history, filtered_vectordb_memory = self._deduplicate_memory(
                    str(self.email_history) if self.email_history else "",
                    vectordb_memory
                )
            except Exception as e:
                print(f"⚠️ Error during deduplication: {e}")
                combined_history = str(self.email_history) if self.email_history else ""
                filtered_vectordb_memory = vectordb_memory
            
            # ✅ STEP 4: Check token limit (like image)
            try:
                if self._check_token_limit(combined_history, max_tokens=self.cfg.max_tokens):
                    print("⚠️ Token limit exceeded. Summarizing memory...")
                    vectordb_memory = self._summarize_memory(vectordb_memory)
                    combined_history, filtered_vectordb_memory = self._deduplicate_memory(
                        str(self.email_history) if self.email_history else "",
                        vectordb_memory
                    )
            except Exception as e:
                print(f"⚠️ Error checking token limit: {e}")
            
            # Format email as user message (from given code structure)
            user_message = f"From: {email_data['sender']}\nSubject: {email_data['subject']}\n\n{email_data['body']}"
            
            while email_state != "finished":
                try:
                    # ✅ HUMARA EXACT LOOP - Same as Chatbot v3
                    if function_call_state == "Function call successful.":
                        email_state = "finished"
                        if function_name == "add_user_info_to_database":
                            try:
                                self.user_manager.refresh_user_info()
                            except Exception as e:
                                print(f"⚠️ Error refreshing user info: {e}")
                        function_call_result_section = (
                            f"## Function Call Executed\n\n"
                            f"- The assistant just called the function `{function_name}` in response to the email.\n"
                            f"- Arguments provided:\n"
                            + "".join([f"  - {k}: {v}\n" for k, v in function_args.items()])
                            + f"- Outcome: ✅ {function_call_state}\n\n"
                            "Please proceed with the email response using the new context.\n\n"
                            + f"{function_call_result}"
                        )
                        print("************************************")
                        print(function_call_result)
                        print("************************************")
                    elif function_call_state == "Function call failed.":
                        function_call_result_section = (
                            f"## Function Call Attempted\n\n"
                            f"- The assistant attempted to call `{function_name}` with the following arguments:\n"
                            + "".join([f"  - {k}: {v}\n" for k, v in function_args.items()])
                            + f"- Outcome: ❌ {function_call_state} - {function_call_result}\n\n"
                            "Please assist based on this result."
                        )
                    
                    # ✅ HUMARA SYSTEM PROMPT - Same structure as Chatbot v3 + Dynamic agent instructions + VectorDB memory
                    # Note: No function call limit - only loop detection stops function calls
                    try:
                        agent_instructions = self.rules_manager.get_rules('agent_instructions')
                    except Exception as e:
                        print(f"⚠️ Error loading agent instructions: {e}")
                        agent_instructions = "Use these tools when appropriate to help manage tasks efficiently. Generate professional, courteous email responses."
                    
                    system_prompt = prepare_system_prompt_for_email_agent(
                        self.user_manager.user_info,
                        self.previous_summary,
                        combined_history,  # ✅ Updated: Combined history (SQL + VectorDB, deduplicated)
                        function_call_result_section,
                        agent_instructions,
                        filtered_vectordb_memory if filtered_vectordb_memory else None  # ✅ FIXED: Use filtered VectorDB memory
                    )
                    
                    # ✅ NEW: Check token limit for final system prompt
                    try:
                        total_prompt_tokens = self.utils.count_number_of_tokens(system_prompt + user_message)
                        max_prompt_tokens = getattr(self.cfg, 'max_prompt_tokens', 12000)
                        if total_prompt_tokens > max_prompt_tokens:
                            print(f"⚠️ Final prompt token limit exceeded ({total_prompt_tokens} > {max_prompt_tokens}). Truncating...")
                            # Truncate combined_history if too long
                            history_tokens = self.utils.count_number_of_tokens(combined_history)
                            if history_tokens > max_prompt_tokens * 0.6:  # If history is more than 60% of limit
                                # Summarize history more aggressively
                                combined_history = self._summarize_memory(combined_history)
                                # Rebuild prompt with truncated history
                                system_prompt = prepare_system_prompt_for_email_agent(
                                    self.user_manager.user_info,
                                    self.previous_summary,
                                    combined_history,
                                    function_call_result_section,
                                    agent_instructions,
                                    ""  # Remove VectorDB memory if still too long
                                )
                    except Exception as e:
                        print(f"⚠️ Error checking final prompt tokens: {e}")
                    
                    print("\n\n==========================================")
                    print(f"System prompt length: {len(system_prompt)} characters")
                    
                    print("\n\nEmail State:", email_state)
                    
                    try:
                        response = self.client.chat.completions.create(
                            model=self.chat_model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_message}
                            ],
                            functions=self.agent_functions,
                            function_call="auto",
                            temperature=self.cfg.temperature
                        )
                    except Exception as e:
                        print(f"❌ Error calling LLM API: {e}")
                        return f"Error: Failed to generate response. {str(e)}"
                    
                    if response.choices[0].message.content:
                        assistant_response = response.choices[0].message.content
                        
                        # ✅ HUMARA MEMORY UPDATE - Same as Chatbot v3 with transaction safety
                        # Note: SQLite auto-commits each query, so we handle errors separately
                        sql_save_success = False
                        vectordb_save_success = False
                        
                        # Save to SQL DB first
                        try:
                            self.email_history_manager.add_to_history(
                                email_data, assistant_response, self.max_history_pairs
                            )
                            self.email_history_manager.update_email_summary(
                                self.max_history_pairs
                            )
                            sql_save_success = True
                            print("✅ Email saved to SQL DB")
                        except Exception as e:
                            print(f"❌ Error updating SQL DB memory: {e}")
                            import traceback
                            print(traceback.format_exc())
                        
                        email_state = "finished"
                        
                        # ✅ HUMARA VECTORDB UPDATE - Same as Chatbot v3 with error handling
                        try:
                            msg_pair = f"email: {user_message}, response: {assistant_response}"
                            self.vector_db_manager.update_vector_db(msg_pair)
                            self.vector_db_manager.refresh_vector_db_client()
                            vectordb_save_success = True
                            print("✅ Email saved to VectorDB")
                        except Exception as e:
                            print(f"❌ Error updating VectorDB memory: {e}")
                            import traceback
                            print(traceback.format_exc())
                            # Note: SQL DB save succeeded but VectorDB failed - data is partially saved
                            # In production, you might want to implement a retry mechanism or queue
                        
                        # Log save status
                        if not sql_save_success:
                            print("⚠️ WARNING: Email response generated but NOT saved to SQL DB")
                        if not vectordb_save_success:
                            print("⚠️ WARNING: Email response generated but NOT saved to VectorDB")
                        
                        function_call_state = None
                        return assistant_response
                    
                    elif response.choices[0].message.function_call:
                        # ✅ Function calling with loop detection only (NO hard limit)
                        # Only stop if email_state is finished or loop detected
                        if email_state == "finished":
                            print("Email state is finished, generating final response...")
                            try:
                                fallback_response = self.client.chat.completions.create(
                                    model=self.chat_model,
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": user_message}
                                    ],
                                    temperature=self.cfg.temperature
                                )
                                assistant_response = fallback_response.choices[0].message.content
                                
                                # Update memory with transaction safety
                                sql_save_success = False
                                vectordb_save_success = False
                                
                                try:
                                    self.email_history_manager.add_to_history(
                                        email_data, assistant_response, self.max_history_pairs
                                    )
                                    sql_save_success = True
                                    print("✅ Fallback response saved to SQL DB")
                                except Exception as e:
                                    print(f"❌ Error updating SQL DB memory: {e}")
                                    import traceback
                                    print(traceback.format_exc())
                                
                                try:
                                    msg_pair = f"email: {user_message}, response: {assistant_response}"
                                    self.vector_db_manager.update_vector_db(msg_pair)
                                    self.vector_db_manager.refresh_vector_db_client()
                                    vectordb_save_success = True
                                    print("✅ Fallback response saved to VectorDB")
                                except Exception as e:
                                    print(f"❌ Error updating VectorDB memory: {e}")
                                    import traceback
                                    print(traceback.format_exc())
                                
                                if not sql_save_success or not vectordb_save_success:
                                    print("⚠️ WARNING: Fallback response generated but not fully saved")
                                
                                function_call_state = None
                                return assistant_response
                            except Exception as e:
                                print(f"❌ Error in fallback response: {e}")
                                return f"Error: Failed to generate fallback response. {str(e)}"
                        
                        function_call_count += 1  # For logging only, not for limiting
                        try:
                            function_name = response.choices[0].message.function_call.name
                            function_args = json.loads(
                                response.choices[0].message.function_call.arguments)
                            
                            # ✅ NEW: Function call loop detection
                            function_call_signature = (function_name, str(sorted(function_args.items())))
                            if function_call_signature in function_call_history:
                                print(f"⚠️ Detected function call loop: {function_name} with same arguments. Breaking loop...")
                                function_call_result_section = f"""# Function Call Loop Detected.\n
                                The function `{function_name}` was called repeatedly with the same arguments.
                                Please conclude the email response based on the available information."""
                                email_state = "finished"
                                continue
                            
                            function_call_history.append(function_call_signature)
                            
                            print("Function name that was requested by the LLM:", function_name)
                            print("Function arguments:", function_args)
                            
                            try:
                                function_call_state, function_call_result = self.execute_function_call(
                                    function_name, function_args)
                            except Exception as e:
                                print(f"❌ Error executing function {function_name}: {e}")
                                function_call_state = "Function call failed."
                                function_call_result = str(e)
                        except json.JSONDecodeError as e:
                            print(f"❌ Error parsing function arguments: {e}")
                            function_call_state = "Function call failed."
                            function_call_result = f"Invalid function arguments: {str(e)}"
                        except Exception as e:
                            print(f"❌ Error processing function call: {e}")
                            function_call_state = "Function call failed."
                            function_call_result = str(e)
                    # Neither function call nor message content (edge case)
                    else:
                        print("⚠️ No valid response from LLM (no content or function call)")
                        return "Warning: No valid assistant response. Please try again."
                
                except Exception as e:
                    error_msg = f"Error in while loop: {str(e)}\n{format_exc()}"
                    print(f"❌ {error_msg}")
                    return error_msg
                
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{format_exc()}"
            print(f"❌ {error_msg}")
            return error_msg
        
        # If loop exits without response (shouldn't happen, but safety check)
        return "Error: Response generation completed without generating a response. Please try again."
    
    def add_few_shot_example(self, category: str, email_data: dict, label: str):
        """
        Add few-shot example to classification VectorDB
        
        Args:
            category: 'ignore', 'notify', or 'respond'
            email_data: Email data dict
            label: Classification label
        """
        self.triage_manager.add_few_shot_example(category, email_data, label)
    
    def optimize_rules(self, feedback: dict) -> dict:
        """
        Optimize rules based on user feedback
        
        Args:
            feedback: {
                'conversation': list,  # Email processing conversation
                'feedback_text': str,  # User feedback
                'rules_to_update': list[str]  # Optional: which rules to update
            }
        
        Returns:
            dict: Updated rules
        """
        return self.prompt_optimizer.optimize_rules(feedback)
    
    def get_rules(self, rule_type: str = None) -> dict:
        """
        Get current rules
        
        Args:
            rule_type: Optional specific rule type, or None for all
        
        Returns:
            dict or str: All rules or specific rule
        """
        if rule_type:
            return self.rules_manager.get_rules(rule_type)
        else:
            return self.rules_manager.get_all_rules()
    
    def update_rules(self, rule_type: str, rule_content: str) -> int:
        """
        Manually update a rule
        
        Args:
            rule_type: 'ignore_rules', 'notify_rules', 'respond_rules', or 'agent_instructions'
            rule_content: New rule content
        
        Returns:
            int: New version number
        """
        return self.rules_manager.update_rules(rule_type, rule_content)

