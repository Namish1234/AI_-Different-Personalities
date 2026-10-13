"""
Misinformation propagation ("telephone game") simulation using the Gemini API.

A short article is rewritten in turn by a chain of personas. After the chain
finishes, ONE auditing call grades every rewritten version against the original
facts and measures tone drift. Results are printed and saved to results.json
and results.csv.

Usage:
    python simulation.py                    # asks you to paste your key when it starts
    (optional) set the GEMINI_API_KEY environment variable to skip the prompt
    python simulation.py --personas 5 --delay 8
    python simulation.py --model gemini-3.1-flash-lite
"""

import argparse
import csv
import getpass
import json
import os
import sys
import time
from typing import List

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

# Free-tier daily request caps differ per model. Flash-Lite models have a much
# higher daily cap than full Flash models, so it is the safer default.
# Check your own limits at https://aistudio.google.com/rate-limit
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

ORIGINAL_ARTICLE = """
1. A new tech device called 'Neuro-Sync' was released today.
2. It reduces screen-induced eye strain by 75%.
3. The initial purchase price is $300.
4. It requires a constant Bluetooth connection to a smartphone to function.
5. In rare cases, the device can cause minor headaches during the first use.
""".strip()

PERSONALITIES = [
    "Tech Entrepreneur",
    "Doctor",
    "Ground Reporter",
    "Sensationalist News Agency",
    "Conspiracy Theorist",
    "Finance Analyst",
    "Lifestyle Influencer",
    "Political Commentator",
    "Comedian",
    "Conservative Parent",
    "BJP Supporter",
    "Left Wing Activist",
]


# --------------------------------------------------------------------------
# Auditor schema
# --------------------------------------------------------------------------

class NodeScore(BaseModel):
    node: int = Field(description="1-based index of the text being graded.")
    fact_1_present: int = Field(description="1 if the text clearly mentions the 'Neuro-Sync' device released today, else 0.")
    fact_2_present: int = Field(description="1 if the text states the 75% eye strain reduction accurately, else 0.")
    fact_3_present: int = Field(description="1 if the text states the $300 price accurately, else 0.")
    fact_4_present: int = Field(description="1 if the text states the constant Bluetooth/smartphone requirement, else 0.")
    fact_5_present: int = Field(description="1 if the text mentions minor headaches in rare cases on first use, else 0.")
    framing_drift_percentage: int = Field(description="0-100: how far tone, style and presentation drifted from neutral factual reporting.")
    notes: str = Field(description="One short sentence: what was lost, exaggerated, or invented in this text.")


class AuditReport(BaseModel):
    nodes: List[NodeScore]


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------

class DailyQuotaExhausted(Exception):
    pass


def make_client() -> genai.Client:
    # 1) environment variable if you set one, 2) otherwise ask when the script starts.
    # The key is only held in memory, so it never ends up saved inside this file.
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        key = getpass.getpass("Paste your Gemini API key (it will not be shown as you type), then press Enter: ").strip()
    if not key:
        sys.exit("No API key provided.")
    return genai.Client(api_key=key)


def call_with_retry(fn, max_retries: int = 5):
    """Retry on 429/503 with exponential backoff. Stop immediately if the
    DAILY quota is gone, since waiting a minute will not fix that."""
    wait = 20
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except errors.APIError as e:
            code = getattr(e, "code", None)
            msg = str(e)
            if code == 429:
                if "PerDay" in msg or "per day" in msg.lower():
                    raise DailyQuotaExhausted(msg)
                print(f"   rate limited (429). Waiting {wait}s (attempt {attempt}/{max_retries})...")
            elif code in (500, 503):
                print(f"   server busy ({code}). Waiting {wait}s (attempt {attempt}/{max_retries})...")
            else:
                raise
            time.sleep(wait)
            wait = min(wait * 2, 120)
    raise RuntimeError("Gave up after repeated retries.")


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def run_chain(client, model, personas, delay):
    """Pass the text through each persona in sequence. Returns list of dicts."""
    chain = []
    current = ORIGINAL_ARTICLE
    for i, persona in enumerate(personas, 1):
        print(f"[{i}/{len(personas)}] {persona} rewriting...")
        prompt = (
            f"You are a {persona}. Rewrite the following information strictly from your "
            f"unique perspective and tone. Do not break character. Keep it under 120 words.\n\n"
            f"{current}"
        )
        resp = call_with_retry(lambda: client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7),
        ))
        text = (resp.text or "").strip()
        if not text:
            # Usually a safety block. Keep the previous text so the chain continues.
            print("   empty response (possibly safety-blocked); reusing previous text.")
            text = current
        chain.append({"node": i, "persona": persona, "text": text})
        current = text
        time.sleep(delay)
    return chain


def audit_chain(client, model, chain):
    """Grade ALL rewritten texts in a single API call."""
    blocks = "\n\n".join(
        f"--- TEXT {n['node']} ({n['persona']}) ---\n{n['text']}" for n in chain
    )
    prompt = f"""You are an objective, strict auditor.

Original facts (ground truth):
{ORIGINAL_ARTICLE}

Below are {len(chain)} rewritten texts, each produced from the previous one. For EACH text,
independently decide which of the 5 original facts are still present AND accurate
(a changed number, a dropped condition, or a distorted claim counts as 0). Also rate the
framing drift away from neutral reporting, and note in one sentence what was lost,
exaggerated or invented. Return one entry per text, with 'node' matching the TEXT number.

{blocks}
"""
    resp = call_with_retry(lambda: client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AuditReport,
            temperature=0.1,
        ),
    ))
    report = resp.parsed or AuditReport.model_validate_json(resp.text)
    return {s.node: s for s in report.nodes}


def save_results(rows):
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    if rows and "facts_retained" in rows[0]:
        with open("results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["node", "persona", "facts_retained", "framing_drift_percentage", "notes", "text"],
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--personas", type=int, default=len(PERSONALITIES), help="how many personas to run")
    ap.add_argument("--delay", type=float, default=7.0, help="seconds between calls (keeps you under the per-minute limit)")
    args = ap.parse_args()

    personas = PERSONALITIES[: args.personas]
    client = make_client()

    print("=" * 60)
    print(f" MISINFORMATION PROPAGATION SIMULATION  (model: {args.model})")
    print(f" API calls needed: {len(personas) + 1}")
    print("=" * 60)
    print(f"ORIGINAL ARTICLE:\n{ORIGINAL_ARTICLE}\n")

    chain = []
    try:
        chain = run_chain(client, args.model, personas, args.delay)
        scores = audit_chain(client, args.model, chain)
    except DailyQuotaExhausted as e:
        print("\nDaily free-tier quota is used up for this model. Try again tomorrow, "
              "switch with --model gemini-3.1-flash-lite, or enable billing.")
        print(str(e)[:300])
        save_results(chain)  # keep whatever was generated
        sys.exit(1)

    rows = []
    for n in chain:
        s = scores.get(n["node"])
        if s is None:
            continue
        retained = s.fact_1_present + s.fact_2_present + s.fact_3_present + s.fact_4_present + s.fact_5_present
        row = {**n, "facts_retained": retained,
               "framing_drift_percentage": s.framing_drift_percentage, "notes": s.notes}
        rows.append(row)
        print(f"NODE {n['node']}: {n['persona'].upper()}")
        print("-" * 60)
        print(n["text"])
        print("-" * 60)
        print(f"FACTS RETAINED : {retained} / 5")
        print(f"TONE DRIFT     : {s.framing_drift_percentage}%")
        print(f"NOTES          : {s.notes}")
        print("=" * 60 + "\n")

    save_results(rows)
    print("Saved results.json and results.csv")


if __name__ == "__main__":
    main()