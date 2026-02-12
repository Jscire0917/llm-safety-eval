import re
# Extract claims from a text by splitting it into sentences and filtering out short ones.
def extract_claims(text):
    return [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 15]
