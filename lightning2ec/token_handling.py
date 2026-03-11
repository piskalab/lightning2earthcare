from pathlib import Path
import requests
from eumdac import AccessToken, DataStore

# Path to credentials.txt; assumes it is one directory above the lightning2ec package
CREDENTIALS_FILE = Path(__file__).resolve().parent.parent / "credentials.txt"


def _load_credentials(file_path=CREDENTIALS_FILE):
    """Read key-value pairs from a credentials file into a dictionary."""
    creds = {}
    if not file_path.exists():
        raise FileNotFoundError(f"Credentials file not found: {file_path}")
    with file_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            creds[key.strip()] = value.strip()
    return creds


# --- EUMETSAT ---
def get_eumetsat_token():
    creds = _load_credentials()
    key = creds.get("EUMETSAT_KEY")
    secret = creds.get("EUMETSAT_SECRET")
    if not key or not secret:
        raise ValueError("Missing EUMETSAT_KEY or EUMETSAT_SECRET in credentials file")
    token = AccessToken((key, secret))
    datastore = DataStore(token)
    return token, datastore


# --- ESA MAAP API ---
def get_earthcare_token():
    """Use OFFLINE_TOKEN to fetch a short-lived access token."""
    creds = _load_credentials()

    offline_token = creds.get("OFFLINE_TOKEN")
    client_id = creds.get("CLIENT_ID")
    client_secret = creds.get("CLIENT_SECRET")

    if not all([offline_token, client_id, client_secret]):
        raise ValueError("Missing OFFLINE_TOKEN, CLIENT_ID, or CLIENT_SECRET in credentials file")

    url = "https://iam.maap.eo.esa.int/realms/esa-maap/protocol/openid-connect/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": offline_token,
        "scope": "offline_access openid"
    }

    response = requests.post(url, data=data)
    response.raise_for_status()

    response_json = response.json()
    access_token = response_json.get('access_token')

    if not access_token:
        raise RuntimeError("Failed to retrieve access token from IAM response")

    return access_token
