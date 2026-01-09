"""
Test script for Email Agent with Classification
This script demonstrates email classification and response generation
"""
from utils.email_agent_v4 import EmailAgent

def test_email_classification():
    """Test email classification with different email types"""
    
    # Initialize email agent
    print("Initializing Email Agent...")
    agent = EmailAgent()
    
    # Test emails for each category
    test_emails = [
        {
            "name": "RESPOND - Question Email",
            "data": {
                "sender": "Alice Smith <alice.smith@company.com>",
                "subject": "Quick question about API documentation",
                "body": """Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice"""
            }
        },
        {
            "name": "IGNORE - Newsletter",
            "data": {
                "sender": "Newsletter <newsletter@company.com>",
                "subject": "Weekly Newsletter - Product Updates",
                "body": """Hi,

This is our weekly newsletter with product updates and promotions.

Check out our latest features and special offers!

Best regards,
Marketing Team"""
            }
        },
        {
            "name": "NOTIFY - System Notification",
            "data": {
                "sender": "System Admin <admin@company.com>",
                "subject": "Server Maintenance Scheduled",
                "body": """Hi Team,

We have scheduled server maintenance for this weekend (Saturday 2 AM - 4 AM).

Services will be temporarily unavailable during this time.

No action needed from your side.

Thanks!"""
            }
        }
    ]
    
    print("\n" + "="*60)
    print("TESTING EMAIL CLASSIFICATION")
    print("="*60)
    
    for i, test in enumerate(test_emails, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test['name']}")
        print(f"{'='*60}")
        print(f"From: {test['data']['sender']}")
        print(f"Subject: {test['data']['subject']}")
        print(f"\nBody:\n{test['data']['body'][:100]}...")
        
        # Process email
        response = agent.process_email(test['data'])
        
        print(f"\n{'='*60}")
        print("RESULT:")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_email_classification()

