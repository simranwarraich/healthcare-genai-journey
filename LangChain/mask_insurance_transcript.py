import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Initialize Presidio engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def mask_insurance_transcript(text: str) -> str:
    """Mask PHI/PII data in insurance transcripts (names, contacts, IDs, etc.)."""

    # --- Step 1: Run Presidio ---
    results = analyzer.analyze(
        text=text,
        entities=[
            "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "DATE_TIME",
            "LOCATION", "MEDICAL_LICENSE", "CREDIT_CARD", "IBAN_CODE",
            "US_SSN", "NRP", "ORGANIZATION"
        ],
        language="en"
    )

    anonymized_text = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    ).text

    # --- Step 2: Regex cleanup for insurance-specific patterns ---
    patterns = {
        # Policy, Claim, Account numbers
        r"\b(POL|CLM|CLAIM|POLICY|ACCT)[-_]?\d{4,10}\b": "[POLICY_ID]",
        # Phone numbers (local and international)
        r"(\+?\d{1,3}[-\s]?)?\d{10}\b": "[PHONE]",
        # Emails
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b": "[EMAIL]",
        # Doctor names (with title)
        r"\bDr\.?\s?[A-Z][a-z]{0,20}\b": "[DOCTOR_NAME]",
        # Dates (mask day and month, keep year)
        r"\b(\d{1,2}[-/]\d{1,2}[-/])?(\d{4})\b": r"[DATE-\2]",
        # ZIP codes (mask except first 3 digits)
        r"\b(\d{3})\d{2,4}\b": r"\1[ZIP]",
        # IP addresses
        r"\b\d{1,3}(?:\.\d{1,3}){3}\b": "[IP]",
        # Credit/Debit card numbers
        r"\b(?:\d[ -]*?){13,16}\b": "[CARD]",
        # SSN or national ID
        r"\b\d{3}-\d{2}-\d{4}\b": "[SSN]",
        # Vehicle numbers
        r"\b[A-Z]{1,3}-\d{1,4}-[A-Z]{1,2}\b": "[VEHICLE]",
        # Medical record numbers
        r"\bMRN\d{3,10}\b": "[MRN]",
        # Passport numbers
        r"\b[A-Z]\d{7}\b": "[PASSPORT]",
        # Driver’s license numbers
        r"\b[A-Z]{1,2}\d{6,8}\b": "[DL]",
        # Bank account numbers
        r"\b\d{8,20}\b": "[BANK_ACCOUNT]",
        # URLs
        r"\bhttps?://[^\s]+": "[URL]",
        # MAC addresses
        r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b": "[MAC]",
        # Fallback numeric identifiers
        r"\b\d{5,}\b": "[NUM]"
    }

    for pattern, replacement in patterns.items():
        anonymized_text = re.sub(pattern, replacement, anonymized_text, flags=re.IGNORECASE)
    return anonymized_text

