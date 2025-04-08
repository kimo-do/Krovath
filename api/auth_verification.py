import os
from dotenv import load_dotenv
from fastapi import HTTPException, Header, status

if not os.getenv("RENDER"):
    # Always load the .env file from the backend root
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dotenv_path = os.path.join(BASE_DIR, '.env')
    load_dotenv(dotenv_path)

BOT_API_COMMUNICATION_TOKEN = os.getenv("KROVATH_COMMUNICATION_TOKEN")

def verify_bot_token(authorization: str = Header(None)):
    """
    A helper function to validate the Bearer token in the Authorization header.
    Raises HTTP 401 if the token is missing or invalid.
    """
    # 1. Check that the header is present and well-formed
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header"
        )

    # 2. Extract the token from the header
    token = authorization.split(" ")[1]

    # 3. Compare with the environment variable
    if token != BOT_API_COMMUNICATION_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token"
        )