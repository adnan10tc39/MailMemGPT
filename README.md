# MailMemGPT: LLM-Based Email Agent with Long-Term Memory

A production-ready email agent system that implements a multi-tier memory architecture for LLM-based email management. The system automatically classifies emails, maintains long-term context through hybrid storage (SQL + Vector Database), and generates intelligent responses using GPT models.

## 🎯 Features

- **Intelligent Email Classification**: Automatically classifies emails into Ignore, Notify, or Respond categories using few-shot learning
- **Multi-Tier Memory Architecture**: 
  - **Hot Memory**: Fast SQL database for recent email history
  - **Warm Memory**: Vector database for semantic similarity search
  - **Cold Memory**: Compressed archive for long-term retention
- **Automatic Memory Loading**: Five-stage pipeline that automatically loads relevant context before response generation
- **Cross-Tier Deduplication**: Intelligent deduplication prevents redundant information in prompts
- **Token Management**: Multi-stage token limit checking with intelligent summarization
- **Function Calling**: Supports email writing, meeting scheduling, calendar checks, and user info updates
- **Dynamic Rules**: Configurable classification rules and agent instructions stored in database

## 📋 Prerequisites

- Python 3.11 (Python 3.12 may cause compatibility issues)
- OpenAI API key
- Git

## 🚀 Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/adnan10tc39/MailMemGPT.git
cd MailMemGPT
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 5: Build Data Folder Structure

The system requires a `data` folder with two subdirectories for storage:

```bash
# Create data directory structure
mkdir -p data/vectordb
```

The structure will be:
```
data/
├── chatbot.db          # SQLite database (created automatically)
└── vectordb/           # Vector database storage (created automatically)
```

### Step 6: Initialize SQL Database

Run the SQL database setup script to create tables and initialize default data:

```bash
cd src
python prepare_sqldb.py
```

This script will:
- Create the `data` directory if it doesn't exist
- Create SQLite database at `data/chatbot.db`
- Initialize tables: `user_info`, `email_history`, `summary`, `email_rules`
- Insert default user information
- Insert default classification rules

**Expected Output:**
```
Directory `data` was created.
```

### Step 7: Initialize Vector Database

Run the vector database setup script:

```bash
python prepare_vectordb.py
```

This script will:
- Create the `data/vectordb` directory if it doesn't exist
- Initialize ChromaDB with OpenAI embeddings
- Create the `chat_history` collection for semantic search

**Expected Output:**
```
Directory 'data/vectordb' was created.
DB collection created: <Collection>
DB collection count: 0
```

### Step 8: Add Few-Shot Examples for Email Classification

The system uses few-shot examples stored in the vector database for email classification. Initialize these examples:

```bash
python initialize_few_shot_examples.py
```

This script adds example emails to three categories:
- **IGNORE**: Spam, newsletters, promotional emails (5 examples)
- **NOTIFY**: System notifications, announcements (5 examples)
- **RESPOND**: Questions, meeting requests, action items (7 examples)

**Expected Output:**
```
============================================================
INITIALIZING FEW-SHOT EXAMPLES FOR EMAIL CLASSIFICATION
============================================================

📧 Adding IGNORE category examples...
✅ Added few-shot example to 'ignore' category
...

📧 Adding NOTIFY category examples...
✅ Added few-shot example to 'notify' category
...

📧 Adding RESPOND category examples...
✅ Added few-shot example to 'respond' category
...

============================================================
✅ FEW-SHOT EXAMPLES INITIALIZED SUCCESSFULLY!
============================================================

Summary:
  - IGNORE examples: 5
  - NOTIFY examples: 5
  - RESPOND examples: 7

You can now test email classification!
```

### Step 9: Test the Email Agent

Run the test script to verify everything is working:

```bash
python test_email_agent.py
```

This will test email classification and response generation with sample emails.

## 📖 Usage

### Basic Usage

```python
from utils.email_agent_v4 import EmailAgent

# Initialize the agent
agent = EmailAgent()

# Process an email
email_data = {
    "sender": "Alice Smith <alice.smith@company.com>",
    "subject": "Quick question about API documentation",
    "body": """Hi John,

I was reviewing the API documentation and noticed some missing endpoints.
Could you help clarify?

Thanks!
Alice"""
}

# Process email (classification + response generation)
response = agent.process_email(email_data)
print(response)
```

### Adding Custom Few-Shot Examples

You can add custom few-shot examples to improve classification accuracy:

```python
from utils.email_agent_v4 import EmailAgent

agent = EmailAgent()

# Add IGNORE example
email = {
    "sender": "Spam <spam@example.com>",
    "subject": "Buy Now!",
    "body": "Limited time offer..."
}
agent.add_few_shot_example('ignore', email, 'ignore')

# Add NOTIFY example
email = {
    "sender": "HR <hr@company.com>",
    "subject": "Holiday Calendar",
    "body": "Please find attached holiday calendar."
}
agent.add_few_shot_example('notify', email, 'notify')

# Add RESPOND example
email = {
    "sender": "Team Member <member@company.com>",
    "subject": "Need Help",
    "body": "Can you help me with this issue?"
}
agent.add_few_shot_example('respond', email, 'respond')
```

### Checking Database Contents

**Check SQL Database:**
```bash
python check_sqldb.py
```

**Check Vector Database:**
```bash
python check_vectordb.py
```

## 🏗️ Architecture

### Memory Tiers

1. **Hot Memory (SQL Database)**
   - Stores recent email-response pairs (configurable limit, default: 2)
   - Stores conversation summaries
   - Stores user information and classification rules
   - Access time: ~milliseconds

2. **Warm Memory (Vector Database)**
   - Stores semantic embeddings of all email-response pairs
   - Stores few-shot examples for classification
   - Enables similarity-based retrieval
   - Access time: ~100ms

3. **Cold Memory (Archive Storage)**
   - Compressed long-term storage (JSONL+gzip)
   - Historical email retention
   - Access time: ~seconds

### Automatic Memory Loading Pipeline

The system automatically executes a 5-stage pipeline for each email:

1. **Load from SQL Database**: Recent history, summaries, user info, rules
2. **Semantic Retrieval from Vector DB**: Top-k similar past emails
3. **Cross-Tier Deduplication**: Remove duplicates between SQL and Vector sources
4. **Token Management**: Check limits, summarize/truncate if needed
5. **Hierarchical Prompt Assembly**: Build structured system prompt

### Email Classification Flow

1. Email arrives → Triage Manager processes it
2. Searches Vector DB for similar few-shot examples
3. Calculates similarity scores for each category (ignore, notify, respond)
4. If confidence > threshold → Uses semantic match
5. If confidence < threshold → Falls back to LLM classification with rules
6. Routes email based on classification:
   - **IGNORE**: Saved to DB, no response
   - **NOTIFY**: Saved to DB, user notified, no response
   - **RESPOND**: Proceeds to memory loading and response generation

## ⚙️ Configuration

Edit `config/config.yml` to customize system behavior:

```yaml
directories:
  db_path: "data/chatbot.db"
  vectordb_dir: "data/vectordb"

llm_config:
  chat_model: "gpt-4o"              # Main LLM model
  summary_model: "gpt-3.5-turbo"     # For summarization
  temperature: 0.0

chat_history_config:
  max_history_pairs: 2               # Recent email pairs in SQL DB
  max_tokens: 2000                    # Token limit for memory

agent_config:
  classification_confidence_threshold: 0.37  # Classification threshold
  max_prompt_tokens: 12000           # Total prompt token limit

vectordb_config:
  collection_name: "chat_history"
  embedding_model: "text-embedding-3-small"
  k: 3                                # Top-k similar emails to retrieve
```

## 📁 Project Structure

```
MailMemGPT/
├── config/
│   └── config.yml                    # Configuration file
├── data/                             # Data storage (created automatically)
│   ├── chatbot.db                    # SQLite database
│   └── vectordb/                     # Vector database storage
├── src/
│   ├── utils/                        # Core utilities
│   │   ├── email_agent_v4.py         # Main EmailAgent class
│   │   ├── email_triage_manager.py   # Email classification
│   │   ├── email_history_manager.py  # SQL database management
│   │   ├── vector_db_manager.py      # Vector database management
│   │   ├── prepare_system_prompt.py  # Prompt construction
│   │   ├── email_tools.py            # Function calling tools
│   │   ├── config.py                 # Configuration loader
│   │   └── ...
│   ├── prepare_sqldb.py              # Initialize SQL database
│   ├── prepare_vectordb.py           # Initialize vector database
│   ├── initialize_few_shot_examples.py  # Add classification examples
│   ├── test_email_agent.py           # Test script
│   ├── check_sqldb.py                # Inspect SQL database
│   └── check_vectordb.py            # Inspect vector database
├── requirements.txt                  # Python dependencies
├── .env                              # Environment variables (create this)
└── README.md                         # This file
```

## 🔧 Troubleshooting

### Issue: "Directory not found" errors

**Solution**: Make sure you've run `prepare_sqldb.py` and `prepare_vectordb.py` first to create the data directories.

### Issue: "OpenAI API key not found"

**Solution**: Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_key_here
```

### Issue: Classification always returns same category

**Solution**: Add more diverse few-shot examples using `initialize_few_shot_examples.py` or manually add examples using the `add_few_shot_example()` method.

### Issue: Token limit exceeded errors

**Solution**: Adjust `max_tokens` and `max_prompt_tokens` in `config/config.yml` or reduce `max_history_pairs`.

## 📝 Database Schema

### SQL Database Tables

**user_info**: User profile information
- id, name, last_name, occupation, location, age, gender, interests

**email_history**: Email-response pairs
- id, user_id, timestamp, email_id, sender, subject, email_body, response_text, session_id, classification, classification_confidence

**summary**: Conversation summaries
- id, user_id, session_id, summary_text, timestamp

**email_rules**: Classification rules and agent instructions
- id, rule_type, rule_content, version, timestamp, user_id

### Vector Database Collections

**chat_history**: All email-response pairs for semantic search

**email_triage_ignore**: Few-shot examples for IGNORE category

**email_triage_notify**: Few-shot examples for NOTIFY category

**email_triage_respond**: Few-shot examples for RESPOND category

## 🧪 Testing

Run the test suite:

```bash
cd src
python test_email_agent.py
```

This will test:
- Email classification (IGNORE, NOTIFY, RESPOND)
- Response generation
- Memory loading and deduplication

## 📚 Key Concepts

### Few-Shot Learning for Classification

The system uses semantic similarity to match incoming emails against few-shot examples stored in the vector database. This enables accurate classification without requiring model fine-tuning.

### Automatic Memory Loading

Unlike traditional RAG systems that require explicit retrieval function calls, this system automatically loads relevant context before prompt construction, making it transparent to the LLM.

### Cross-Tier Deduplication

The system intelligently removes duplicate information between SQL and Vector database sources, maximizing token efficiency while maintaining comprehensive context.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Inspired by MemGPT paper for virtual context management
- LangChain and LangGraph for memory architecture concepts
- OpenAI for GPT models and embeddings

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This system is designed for research and development purposes. For production use, consider additional security measures, error handling, and scalability optimizations.
