"""Vapi REST wrapper — phone number provisioning, assistant config."""

import httpx

from config import settings

BASE_URL = "https://api.vapi.ai"


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {settings.vapi_api_key}"},
        timeout=30.0,
    )


def provision_phone_number(agent_id: str) -> str:
    """Buy/assign a phone number for an agent. Returns Vapi phone number id."""
    raise NotImplementedError
