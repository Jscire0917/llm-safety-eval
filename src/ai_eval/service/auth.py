import os
from fastapi import Header, HTTPException

# Set env var: export API_KEYS=sk-mysecret123,sk-localtest
VALID_KEYS = set(
    k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()
)
if not VALID_KEYS:
    print("[WARNING] No API_KEYS set in environment – auth disabled")


# Dependency to authenticate API key from headers
def authenticate(api_key: str = Header(...)):
    if api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key
