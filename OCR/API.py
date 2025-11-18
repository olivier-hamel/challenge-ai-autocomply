from enum import Enum
import random
import time

import requests


class MODELS(Enum):
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GPT_4O = "gpt-4o"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4.5"

    def find_by_name(name: str) -> "MODELS | None":
        for model in MODELS:
            if model.name == name:
                return model
        return None


class APIClient:
    def __init__(
        self,
        api_url: str = "https://ai-models.autocomply.ca",
        api_key: str = "sk-ac-7f8e9d2c4b1a6e5f3d8c7b9a2e4f6d1c",
    ):
        """
        Initialize the PDF processor with API configuration.

        Args:
            api_url: Base URL of the AutoComply API
            api_key: API key for authentication
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.calls_made = 0

    def call_count(self) -> int:
        """
        Get the number of API calls made.

        Returns:
            The number of API calls made
        """
        return self.calls_made

    def _back_off_delay(
        self, attempt: int, base_delay: float = 1.0, max_delay: float = 60.0
    ):
        """
        Sleep for an exponential back-off delay.

        Args:
            attempt: The current retry attempt number
            base_delay: The base delay in seconds
            max_delay: The maximum delay in seconds
        """
        exp_delay = base_delay * (2**attempt)
        delay_max = min(exp_delay, max_delay)
        delay = random.uniform(0, delay_max)
        print(f"Retrying after {delay:.2f} seconds...")
        time.sleep(delay)

    def _call_post(
        self, endpoint: str, payload: dict, max_retries: int = 3
    ) -> dict | None:
        """
        Internal method to make a POST API call with retries.

        Args:
            endpoint: API endpoint to call
            payload: JSON payload for the request
            max_retries: Maximum number of retries for failed requests
        Returns:
            The JSON response from the API or None if failed
        """
        url = f"{self.api_url}/{endpoint}"
        for attempt in range(max_retries):
            try:
                self.calls_made += 1
                response = requests.post(url, json=payload, headers=self.headers)
                response.raise_for_status()

                if response.status_code != 200:
                    print(
                        f"API returned status code {response.status_code}: {response.text}"
                    )
                    self._back_off_delay(attempt)
                    continue

                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"API request error (attempt {attempt + 1}/{max_retries}): {e}")
                self._back_off_delay(attempt)
        print("Max retries reached. API call failed.")
        return None

    def ask(
        self,
        prompt: str,
        model: MODELS = MODELS.CLAUDE_SONNET_4_5,
        max_retries: int = 3,
    ) -> str:
        """
        Send a prompt to the AutoComply API and get a response.

        Args:
            prompt: The input prompt string
            model: The model to use for processing
            max_retries: Maximum number of retries for failed requests
        Returns:
            The response from the API as a string
        """
        payload = {
            "query": prompt,
            "model": model.value,
        }

        try:
            response = self._call_post("ask", payload, max_retries=max_retries)
            if response is None:
                return ""
            return response.get("result", "")
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            return ""
