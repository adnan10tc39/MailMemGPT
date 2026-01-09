import os
import sqlite3
from pyprojroot import here


def create_user_info():
    """
    Creates a SQLite database and initializes tables for user information, chat history, and summaries.

    This function:
    - Creates a `data` directory if it doesn't exist.
    - Establishes a SQLite connection to `chatbot.db`.
    - Creates the following tables if they don't already exist:
        - `user_info`: Stores user details (e.g., name, occupation, location, etc.).
        - `chat_history`: Records chat interactions with timestamps and session IDs.
        - `summary`: Stores summarized chat sessions.
    - Inserts a sample user (`Farzad Roozitalab`) if no user record exists.

    Tables:
        user_info:
            - id (INTEGER, PRIMARY KEY)
            - name (TEXT, NOT NULL)
            - last_name (TEXT, NOT NULL)
            - occupation (TEXT, NOT NULL)
            - location (TEXT, NOT NULL)
            - age (INTEGER, NULLABLE)
            - gender (TEXT, NULLABLE)
            - interests (TEXT, NULLABLE)

        chat_history:
            - id (INTEGER, PRIMARY KEY)
            - user_id (INTEGER, FOREIGN KEY -> user_info.id)
            - timestamp (DATETIME, DEFAULT CURRENT_TIMESTAMP)
            - question (TEXT, NOT NULL)
            - answer (TEXT, NOT NULL)
            - session_id (TEXT, NOT NULL)

        summary:
            - id (INTEGER, PRIMARY KEY)
            - user_id (INTEGER, FOREIGN KEY -> user_info.id)
            - session_id (TEXT, NOT NULL)
            - summary_text (TEXT, NOT NULL)
            - timestamp (DATETIME, DEFAULT CURRENT_TIMESTAMP)
    """
    # Connect to SQLite database (or create it if it doesn't exist)
    if not os.path.exists(here("data")):
        # If it doesn't exist, create the directory and create the embeddings
        os.makedirs(here("data"))
        print(f"Directory `{here('data')}` was created.")
    conn = sqlite3.connect(here("data/chatbot.db"))
    cursor = conn.cursor()

    # Create Tables
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS user_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        occupation TEXT NOT NULL,
        location TEXT NOT NULL,
        age INTEGER,  -- Allow NULL values (so chatbot can fill later)
        gender TEXT,  -- Allow NULL values
        interests TEXT  -- Allow NULL values
    );

    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        session_id TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES user_info(id)
    );

    CREATE TABLE IF NOT EXISTS summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        session_id TEXT NOT NULL,
        summary_text TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES user_info(id)
    );

    CREATE TABLE IF NOT EXISTS email_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        email_id TEXT UNIQUE NOT NULL,
        sender TEXT NOT NULL,
        subject TEXT NOT NULL,
        email_body TEXT NOT NULL,
        response_text TEXT NOT NULL,
        session_id TEXT NOT NULL,
        classification TEXT,
        classification_confidence REAL,
        FOREIGN KEY (user_id) REFERENCES user_info(id)
    );

    CREATE TABLE IF NOT EXISTS email_rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rule_type TEXT NOT NULL,
        rule_content TEXT NOT NULL,
        version INTEGER DEFAULT 1,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        user_id INTEGER,
        FOREIGN KEY (user_id) REFERENCES user_info(id)
    );
    """)

    # Insert Sample User if Not Exists (leaving age, gender, interests empty)
    cursor.execute("""
    INSERT INTO user_info (name, last_name, occupation, location, age, gender, interests)
    SELECT 'Muhammad', 'Adnan', 'ML Engineer', 'Islamabad', 32, NULL, NULL
    WHERE NOT EXISTS (SELECT 1 FROM user_info);
    """)
    
    # Insert Default Rules if Not Exists
    user_id_result = cursor.execute("SELECT id FROM user_info LIMIT 1").fetchone()
    if user_id_result:
        user_id = user_id_result[0]
        default_rules = [
            ('ignore_rules', 'Spam, promotional emails, mass announcements, no action needed'),
            ('notify_rules', 'Important information that user should know but doesn\'t need a response (e.g., system notifications, status updates)'),
            ('respond_rules', 'Emails that need a direct response (e.g., questions, meeting requests, action items)'),
            ('agent_instructions', 'Use these tools when appropriate to help manage tasks efficiently. Generate professional, courteous email responses.')
        ]
        
        for rule_type, rule_content in default_rules:
            cursor.execute("""
                INSERT INTO email_rules (rule_type, rule_content, version, user_id)
                SELECT ?, ?, 1, ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM email_rules 
                    WHERE rule_type = ? AND user_id = ?
                );
            """, (rule_type, rule_content, user_id, rule_type, user_id))

    # Commit changes and close the connection
    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_user_info()
