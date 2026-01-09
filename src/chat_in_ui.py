import time
import gradio as gr
from utils.email_agent_v4 import EmailAgent

# Initialize email agent
email_agent = EmailAgent()


def process_email(sender, subject, body):
    """Process email and generate response with classification"""
    if not sender.strip() or not subject.strip() or not body.strip():
        return "Please fill in all fields: Sender, Subject, and Body."
    
    email_data = {
        "sender": sender.strip(),
        "subject": subject.strip(),
        "body": body.strip()
    }
    
    start_time = time.time()
    try:
        response = email_agent.process_email(email_data)
        end_time = time.time()
        
        # Get classification info
        classification = email_data.get('classification', 'unknown').upper()
        confidence = email_data.get('classification_confidence', 0.0)
        
        # Format the response nicely with classification
        formatted_response = f"""**📧 Email Classification: {classification}** (Confidence: {confidence:.2f})

---

**Response:**

{response}

---
*Processed in {round(end_time - start_time, 2)}s*"""
        return formatted_response
    except Exception as e:
        return f"Error processing email: {str(e)}"


with gr.Blocks(title="Email Agent v4") as demo:
    gr.Markdown("# 📧 Email Agent v4 - Long-Term Memory")
    gr.Markdown("Enter email details below. The agent will generate a professional response using long-term memory with SQL DB and VectorDB.")
    
    with gr.Row():
        with gr.Column(scale=1):
            sender_input = gr.Textbox(
                label="From (Sender)",
                placeholder="Alice Smith <alice.smith@company.com>",
                lines=1
            )
            subject_input = gr.Textbox(
                label="Subject",
                placeholder="Quick question about API documentation",
                lines=1
            )
            body_input = gr.Textbox(
                label="Email Body",
                placeholder="Hi John,\n\nI was reviewing the API documentation...",
                lines=10
            )
            email_submit_btn = gr.Button(value="Generate Response", variant="primary")
        
        with gr.Column(scale=1):
            email_output = gr.Markdown(
                label="Generated Response",
                value="Response will appear here..."
            )
    
    with gr.Row():
        clear_email_btn = gr.ClearButton([sender_input, subject_input, body_input, email_output])
    
    # Example emails - Organized by category
    gr.Markdown("### 📝 Example Emails (Click to load)")
    
    # RESPOND Category Examples
    gr.Markdown("#### ✅ RESPOND Category (Needs Response)")
    with gr.Row():
        example_respond1_btn = gr.Button(value="RESPOND 1: API Question", size="sm", variant="primary")
        example_respond2_btn = gr.Button(value="RESPOND 2: Meeting Request", size="sm", variant="primary")
        example_respond3_btn = gr.Button(value="RESPOND 3: Follow-up", size="sm", variant="primary")
        example_respond4_btn = gr.Button(value="RESPOND 4: Urgent Help", size="sm", variant="primary")
    
    # NOTIFY Category Examples
    gr.Markdown("#### 🔔 NOTIFY Category (Info Only)")
    with gr.Row():
        example_notify1_btn = gr.Button(value="NOTIFY 1: Server Maintenance", size="sm", variant="secondary")
        example_notify2_btn = gr.Button(value="NOTIFY 2: Holiday Calendar", size="sm", variant="secondary")
        example_notify3_btn = gr.Button(value="NOTIFY 3: Build Success", size="sm", variant="secondary")
    
    # IGNORE Category Examples
    gr.Markdown("#### 🚫 IGNORE Category (Spam/Promotions)")
    with gr.Row():
        example_ignore1_btn = gr.Button(value="IGNORE 1: Newsletter", size="sm")
        example_ignore2_btn = gr.Button(value="IGNORE 2: Promotion", size="sm")
        example_ignore3_btn = gr.Button(value="IGNORE 3: Spam", size="sm")
    
    # RESPOND Examples Functions
    def load_respond1():
        return (
            "Alice Smith <alice.smith@company.com>",
            "Quick question about API documentation",
            """Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice"""
        )
    
    def load_respond2():
        return (
            "Bob Johnson <bob.johnson@company.com>",
            "Meeting Request - Project Discussion",
            """Hi John,

I'd like to schedule a meeting to discuss the upcoming project timeline. Would you be available for a 30-minute call this week?

Let me know your availability.

Best regards,
Bob"""
        )
    
    def load_respond3():
        return (
            "Sarah Chen <sarah.chen@company.com>",
            "Follow-up on UI mockups",
            """Hi John,

Any update on the UI mockups for the dashboard that we discussed last week?

Thanks!
Sarah"""
        )
    
    def load_respond4():
        return (
            "Mike Davis <mike.davis@company.com>",
            "Urgent: Need help with deployment",
            """Hi John,

We're having issues with the production deployment. Can you help us troubleshoot?

The error is: Connection timeout

Thanks!
Mike"""
        )
    
    # NOTIFY Examples Functions
    def load_notify1():
        return (
            "System Admin <admin@company.com>",
            "Server Maintenance Scheduled",
            """Hi Team,

We have scheduled server maintenance for this weekend (Saturday 2 AM - 4 AM).

Services will be temporarily unavailable during this time.

No action needed from your side.

Thanks!"""
        )
    
    def load_notify2():
        return (
            "HR Team <hr@company.com>",
            "Holiday Calendar 2024",
            """Dear Team,

Please find attached the holiday calendar for 2024.

Mark your calendars!

Best regards,
HR Team"""
        )
    
    def load_notify3():
        return (
            "Build System <build@company.com>",
            "Build Successful - Project v2.1",
            """Build completed successfully.

Project: v2.1
Status: Passed
Duration: 15 minutes

No action required."""
        )
    
    # IGNORE Examples Functions
    def load_ignore1():
        return (
            "Newsletter <newsletter@company.com>",
            "Weekly Newsletter - Product Updates",
            """Hi,

This is our weekly newsletter with product updates and promotions.

Check out our latest features and special offers!

Best regards,
Marketing Team"""
        )
    
    def load_ignore2():
        return (
            "Promotions <promo@company.com>",
            "Special Offer - 50% Off",
            """Limited time offer! Get 50% off on all products.

Use code: SAVE50

Shop now!"""
        )
    
    def load_ignore3():
        return (
            "Spam <spam@example.com>",
            "Buy Now! Limited Time",
            """Click here to buy now! Limited time offer!"""
        )
    
    # Connect RESPOND buttons
    example_respond1_btn.click(
        fn=load_respond1,
        outputs=[sender_input, subject_input, body_input]
    )
    example_respond2_btn.click(
        fn=load_respond2,
        outputs=[sender_input, subject_input, body_input]
    )
    example_respond3_btn.click(
        fn=load_respond3,
        outputs=[sender_input, subject_input, body_input]
    )
    example_respond4_btn.click(
        fn=load_respond4,
        outputs=[sender_input, subject_input, body_input]
    )
    
    # Connect NOTIFY buttons
    example_notify1_btn.click(
        fn=load_notify1,
        outputs=[sender_input, subject_input, body_input]
    )
    example_notify2_btn.click(
        fn=load_notify2,
        outputs=[sender_input, subject_input, body_input]
    )
    example_notify3_btn.click(
        fn=load_notify3,
        outputs=[sender_input, subject_input, body_input]
    )
    
    # Connect IGNORE buttons
    example_ignore1_btn.click(
        fn=load_ignore1,
        outputs=[sender_input, subject_input, body_input]
    )
    example_ignore2_btn.click(
        fn=load_ignore2,
        outputs=[sender_input, subject_input, body_input]
    )
    example_ignore3_btn.click(
        fn=load_ignore3,
        outputs=[sender_input, subject_input, body_input]
    )
    
    # Handle email submission
    email_submit_btn.click(
        fn=process_email,
        inputs=[sender_input, subject_input, body_input],
        outputs=[email_output]
    )

if __name__ == "__main__":
    demo.launch()
