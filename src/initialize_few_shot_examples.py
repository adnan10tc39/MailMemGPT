"""
Initialize few-shot examples for email classification
This script adds example emails to each category (ignore, notify, respond) in VectorDB
"""
from utils.email_agent_v4 import EmailAgent

def initialize_few_shot_examples():
    """Initialize few-shot examples for email classification"""
    
    agent = EmailAgent()
    
    print("="*60)
    print("INITIALIZING FEW-SHOT EXAMPLES FOR EMAIL CLASSIFICATION")
    print("="*60)
    
    # IGNORE Category Examples
    print("\n📧 Adding IGNORE category examples...")
    ignore_examples = [
        {
            "sender": "Newsletter <newsletter@company.com>",
            "subject": "Weekly Newsletter - Product Updates",
            "body": """Hi,

This is our weekly newsletter with product updates and promotions.

Check out our latest features and special offers!

Best regards,
Marketing Team"""
        },
        {
            "sender": "noreply@system.com",
            "subject": "System Notification - Maintenance Complete",
            "body": """System maintenance has been completed successfully.

No action required."""
        },
        {
            "sender": "Promotions <promo@company.com>",
            "subject": "Special Offer - 50% Off",
            "body": """Limited time offer! Get 50% off on all products.

Use code: SAVE50

Shop now!"""
        },
        {
            "sender": "Spam <spam@example.com>",
            "subject": "Buy Now! Limited Time",
            "body": """Click here to buy now! Limited time offer!"""
        },
        {
            "sender": "Auto-Reply <noreply@company.com>",
            "subject": "Out of Office - Automatic Reply",
            "body": """I am currently out of office. Will respond when I return."""
        }
    ]
    
    for email in ignore_examples:
        agent.add_few_shot_example('ignore', email, 'ignore')
    
    # NOTIFY Category Examples
    print("\n📧 Adding NOTIFY category examples...")
    notify_examples = [
        {
            "sender": "System Admin <admin@company.com>",
            "subject": "Server Maintenance Scheduled",
            "body": """Hi Team,

We have scheduled server maintenance for this weekend (Saturday 2 AM - 4 AM).

Services will be temporarily unavailable during this time.

No action needed from your side.

Thanks!"""
        },
        {
            "sender": "HR Team <hr@company.com>",
            "subject": "Holiday Calendar 2024",
            "body": """Dear Team,

Please find attached the holiday calendar for 2024.

Mark your calendars!

Best regards,
HR Team"""
        },
        {
            "sender": "Build System <build@company.com>",
            "subject": "Build Successful - Project v2.1",
            "body": """Build completed successfully.

Project: v2.1
Status: Passed
Duration: 15 minutes

No action required."""
        },
        {
            "sender": "IT Support <it@company.com>",
            "subject": "System Update Completed",
            "body": """Hello,

The system update has been completed successfully. All services are now running normally.

No action needed from you.

IT Team"""
        },
        {
            "sender": "HR <hr@company.com>",
            "subject": "New Policy Announcement",
            "body": """Dear All,

We have updated our company policy. Please review the attached document.

Thank you.

HR Department"""
        }
    ]
    
    for email in notify_examples:
        agent.add_few_shot_example('notify', email, 'notify')
    
    # RESPOND Category Examples
    print("\n📧 Adding RESPOND category examples...")
    respond_examples = [
        {
            "sender": "Alice Smith <alice.smith@company.com>",
            "subject": "Quick question about API documentation",
            "body": """Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice"""
        },
        {
            "sender": "Bob Johnson <bob.johnson@company.com>",
            "subject": "Meeting Request - Project Discussion",
            "body": """Hi John,

I'd like to schedule a meeting to discuss the upcoming project timeline. Would you be available for a 30-minute call this week?

Let me know your availability.

Best regards,
Bob"""
        },
        {
            "sender": "Sarah Chen <sarah.chen@company.com>",
            "subject": "Follow-up on UI mockups",
            "body": """Hi John,

Any update on the UI mockups for the dashboard that we discussed last week?

Thanks!
Sarah"""
        },
        {
            "sender": "Mike Davis <mike.davis@company.com>",
            "subject": "Urgent: Need help with deployment",
            "body": """Hi John,

We're having issues with the production deployment. Can you help us troubleshoot?

The error is: Connection timeout

Thanks!
Mike"""
        },
        {
            "sender": "Tom Wilson <tom.wilson@company.com>",
            "subject": "Can you help me?",
            "body": """Hi,

Can you help me with this problem? I need your assistance.

Thanks!"""
        },
        {
            "sender": "Lisa Brown <lisa.brown@company.com>",
            "subject": "Question about project",
            "body": """Hello,

I have a question about the project. When will it be completed?

Please let me know.

Lisa"""
        },
        {
            "sender": "David Lee <david.lee@company.com>",
            "subject": "Need your approval",
            "body": """Hi,

I need your approval for the budget proposal. Can you review it?

Thanks,
David"""
        }
    ]
    
    for email in respond_examples:
        agent.add_few_shot_example('respond', email, 'respond')
    
    print("\n" + "="*60)
    print("✅ FEW-SHOT EXAMPLES INITIALIZED SUCCESSFULLY!")
    print("="*60)
    print("\nSummary:")
    print(f"  - IGNORE examples: {len(ignore_examples)}")
    print(f"  - NOTIFY examples: {len(notify_examples)}")
    print(f"  - RESPOND examples: {len(respond_examples)}")
    print("\nYou can now test email classification!")

if __name__ == "__main__":
    initialize_few_shot_examples()

