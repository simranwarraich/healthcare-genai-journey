import re
import string
from mask_insurance_transcript import mask_insurance_transcript  # your masking function from masking.py
from transformers import pipeline
def preprocess_text(text):
    """
    Preprocess text for NLP: lowercase, remove extra spaces and punctuation.
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def normalize_placeholders(text):
    """
    Replace masked placeholders with readable neutral values
    to help summarization.
    """
    text = text.replace("[PERSON]", "the customer")
    text = text.replace("[ORG]", "the hospital")
    text = text.replace("[PHONE]", "the phone number")
    text = text.replace("[EMAIL]", "the email address")
    text = text.replace("datetime", "the date")
    text = text.replace("[POLICY_ID]", "the policy")
    return text

def summarize_text(text):
    """
    Summarize text using a local Hugging Face model (Pegasus).
    """
    summarizer = pipeline("summarization", model="pszemraj/long-t5-tglobal-base-16384-book-summary")
    summary = summarizer(text, min_length=60, max_length=300)
    return summary[0]['summary_text']

def main():
    if __name__ == "__main__":
        sample_text = """
        Agent: Hi, thank you for calling HealthSure Insurance, this is Daniel, how may I help you today?

    Customer: Hi Daniel, my name is Mrs. Anita Verma, and I’m calling about my claim — I submitted it for my knee surgery last month at Lotus Care Hospital in Mumbai. I just got a message saying it was denied, and I don’t understand why.

    Agent: I’m sorry to hear that, Mrs. Verma. May I have your policy number to check the details?

    Customer: Yes, it’s HSI-458921-2025.

    Agent: Thank you. Please confirm your date of birth and phone number for verification.

    Customer: Sure, it’s February 19, 1981, and my phone number is +91 98765 43210.

    Agent: Great, thank you. I’m checking your claim for the surgery performed on June 24, 2025. It looks like the denial reason was that the contact number on file for the hospital was invalid, so our verification team couldn’t reach them for the pre-authorization check.

    Customer: That’s ridiculous! I gave the correct number — it’s 022-3492-8831 for the billing office. I even told the hospital staff to expect the call!

    Agent: I understand your frustration. The number on file was listed as 022-3948-8813, which seems to be a typo. Because the verification team couldn’t confirm, the claim was auto-denied under the “contact unreachable” rule.

    Customer: So what should I do now? I can’t afford to pay the whole ₹1,25,000 out of pocket.

    Agent: No worries — you can file a claim reconsideration request. I’ll email you the form at anita.verma84@gmail.com
    . Just attach a hospital contact confirmation letter, and we can reopen the case.

    Customer: Okay, please do that. Thank you, Daniel.

    Agent: You’re welcome, Mrs. Verma. I’ve also updated your contact details and marked this as urgent. You’ll get an update within 5 working days.

    Customer: Thanks. Goodbye.
        """
        
        # Step 1: Mask identifiers
        anonymized_text = mask_insurance_transcript(sample_text)
        
        # Step 2: Preprocess
        clean_text = preprocess_text(anonymized_text)
        clean_text = normalize_placeholders(clean_text)
        
        # Step 3: Summarize
        summary = summarize_text(clean_text)
        
        print("\n--- Anonymized Text ---\n", anonymized_text)
        print("\n--- Clean Text ---\n", clean_text)
        print("\n--- Summary ---\n", summary)

if __name__ == "__main__":
    main()
