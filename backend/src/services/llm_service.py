"""
LLM Service using LM Studio (local) with deepseek-r1-distill-llama-8b
OpenAI-compatible API (no SDK required)
"""

import json
import urllib.request
import urllib.error
from typing import List, Dict


class LLMService:
    def __init__(
        self,
        api_base: str = "http://127.0.0.1:1234/v1",
        model: str = "deepseek-r1-distill-llama-8b",
        enabled: bool = True,
    ):
        self.api_base = api_base.rstrip("/")
        self.chat_url = f"{self.api_base}/chat/completions"
        self.models_url = f"{self.api_base}/models"
        self.model = model
        self.enabled = enabled

    # --------------------------------------------------
    # HEALTH CHECK
    # --------------------------------------------------
    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(self.models_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as res:
                return res.status == 200
        except Exception:
            return False

    # --------------------------------------------------
    # MAIN CALL
    # --------------------------------------------------
    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
        if not self.enabled:
            raise RuntimeError("LLM service disabled")

        payload = {
            "model": self.model,   # ignored internally by LM Studio but REQUIRED
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.chat_url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=90) as res:
                body = json.loads(res.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"LLM HTTP error: {e.read().decode()}")
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")


# --------------------------------------------------
# EXAMPLE: MATCH EXPLANATION
# --------------------------------------------------
def generate_match_explanation(
    llm: LLMService,
    job_description: str,
    resume_text: str,
    score: float,
    matched_skills: List[str],
    missing_skills: List[str],
) -> str:
    system_prompt = (
        "You are a senior technical recruiter and career advisor. "
        "Be honest, structured, and actionable."
    )

    user_prompt = f"""
JOB DESCRIPTION:
{job_description[:800]}

CANDIDATE RESUME:
{resume_text[:800]}

MATCH SCORE: {score:.0%}
MATCHED SKILLS: {", ".join(matched_skills) or "None"}
MISSING SKILLS: {", ".join(missing_skills) or "None"}

Provide:
1. Match verdict
2. Key strengths
3. Skill gaps
4. Actionable improvement steps
5. Final hiring recommendation
"""

    return llm.chat(system_prompt, user_prompt, max_tokens=700)


# --------------------------------------------------
# MANUAL TEST
# --------------------------------------------------
if __name__ == "__main__":
    llm = LLMService()

    if not llm.is_available():
        raise SystemExit("❌ LM Studio server not reachable")

    print("✅ LM Studio detected\n")

    response = llm.chat(
        system_prompt="You are a helpful assistant.",
        user_prompt="Explain TF-IDF in simple terms.",
        max_tokens=300,
    )

    print("LLM RESPONSE:\n")
    print(response)
