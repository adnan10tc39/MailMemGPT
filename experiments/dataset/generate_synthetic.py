"""Two-pass synthetic email corpus generator (Agent B).

Pass 1 (pure python, ``random.Random(seed)``) builds a deterministic scenario
skeleton: 500 emails over days 1..84 for one assistant owner (John Doe,
software engineering manager at NovaCore) — 200 IGNORE / 150 NOTIFY /
150 RESPOND, exactly 40 multi-email threads, >=60 context-dependent RESPOND
follow-ups (``requires_context=true`` with ``context_note`` naming the parent
email_id), and the recurring storylines required by the design spec (weekly
status meetings, Q1 platform migration, partnership negotiation,
invoice/payment thread, recruiting thread).

Pass 2 writes realistic subject+body text with gpt-4o (JSON mode,
temperature 0.7), 5 scenarios per call; follow-ups receive the parent
email's generated text so replies can quote their parents. Thread subjects
are normalized to ``Re: <parent subject>`` after generation.

CLI:
    python3 -m experiments.dataset.generate_synthetic
        [--pilot] [--out PATH] [--seed N] [--selftest]

Outputs (full mode): experiments/data/synthetic_500.jsonl,
experiments/data/synthetic_pilot_20.jsonl, experiments/data/stats.json.
Pilot mode generates ONLY the deterministic 20-email subset (4 LLM calls).
The selftest injects a fake LLM and makes no OpenAI call.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Protocol

ENV_PATH = "/media/adnan/DATA/Agentic-LongTerm-Memory/.env"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BASE_DATE = _dt.date(2026, 1, 3)  # day 1 -> 2026-01-03 (matches spec example)

OWNER = {"name": "John Doe", "role": "software engineering manager at NovaCore",
         "email": "john.doe@novacore.io"}

#: The 10 fictional correspondent personas (names/roles/addresses/domains).
PERSONAS: dict[str, dict[str, str]] = {
    "sarah": {"name": "Sarah Chen", "role": "Engineering Director (John's manager)",
              "email": "sarah.chen@novacore.io", "domain": "novacore.io"},
    "marcus": {"name": "Marcus Webb", "role": "Senior Backend Engineer",
               "email": "marcus.webb@novacore.io", "domain": "novacore.io"},
    "priya": {"name": "Priya Raghavan", "role": "Product Manager",
              "email": "priya.raghavan@novacore.io", "domain": "novacore.io"},
    "tom": {"name": "Tom Delaney", "role": "DevOps Lead",
            "email": "tom.delaney@novacore.io", "domain": "novacore.io"},
    "elena": {"name": "Elena Vasquez", "role": "Technical Recruiter",
              "email": "elena.vasquez@novacore.io", "domain": "novacore.io"},
    "jenna": {"name": "Jenna Morris", "role": "Office & People Ops Manager",
              "email": "jenna.morris@novacore.io", "domain": "novacore.io"},
    "alex": {"name": "Alex Turner", "role": "QA Engineer",
             "email": "alex.turner@novacore.io", "domain": "novacore.io"},
    "david": {"name": "David Okafor", "role": "Director of Partnerships, Meridian Labs",
              "email": "david.okafor@meridianlabs.com", "domain": "meridianlabs.com"},
    "grace": {"name": "Grace Liu", "role": "Accounts Receivable, Brightpath Consulting",
              "email": "grace.liu@brightpath-consulting.com",
              "domain": "brightpath-consulting.com"},
    "rachel": {"name": "Rachel Kim", "role": "Client Success Manager, Harborview Systems",
               "email": "rachel.kim@harborview.co", "domain": "harborview.co"},
}

SYSTEM_SENDERS: dict[str, tuple[str, str]] = {
    "ci": ("NovaCore CI", "ci-bot@novacore.io"),
    "monitoring": ("NovaCore Monitoring", "monitoring@novacore.io"),
    "calendar": ("NovaCore Calendar", "calendar-noreply@novacore.io"),
    "it": ("NovaCore IT", "it-support@novacore.io"),
    "reports": ("NovaCore Analytics", "reports@novacore.io"),
    "security": ("NovaCore Security", "security-scan@novacore.io"),
}

SPAM_SENDERS: list[tuple[str, str]] = [
    ("CloudDeals Weekly", "deals@clouddealsweekly.com"),
    ("TechGear Outlet", "promo@techgearoutlet.net"),
    ("Prime Office Supplies", "offers@primeofficesupplies.com"),
    ("Summit Webinars", "events@summitwebinars.io"),
    ("TravelPerks", "newsletter@travelperks.co"),
    ("SaaS Growth Digest", "digest@saasgrowthdigest.com"),
    ("DevWeekly", "news@devweekly.io"),
    ("Frontend Focus", "hello@frontendfocus.dev"),
    ("DataEng Roundup", "editor@dataengroundup.com"),
    ("LeadGen Pro", "sales@leadgenpro.biz"),
    ("Apex CRM", "outreach@apexcrm.com"),
    ("SecureStack", "sdr@securestack.ai"),
]

AUTOREPLY_SENDERS: list[tuple[str, str]] = [
    ("Colin Mercer", "colin.mercer@vantagepoint-cap.com"),
    ("Dana Whitfield", "dana.whitfield@osprey-legal.com"),
    ("Sanjay Patel", "s.patel@quantics.io"),
    ("Laura Beck", "laura.beck@crestline-media.com"),
]

WEEKLY_AGENDA = ["sprint progress and blockers", "Q1 platform migration status",
                 "hiring pipeline update", "incident review and reliability",
                 "quarterly OKR check-in", "release planning"]

MIGRATION_TOPICS = [
    "database cutover plan for the Q1 platform migration: proposed downtime window and rollback strategy",
    "API gateway migration under the Q1 platform migration: deprecating v1 endpoints and client compatibility",
    "staging environment parity issues blocking the Q1 platform migration test plan",
    "final go/no-go checklist for the Q1 platform migration cutover weekend",
]

PARTNERSHIP_TOPICS = [
    "opening terms for the Meridian Labs integration partnership: revenue share and support commitments",
    "redlines on the draft Meridian Labs partnership agreement: liability cap and data-processing terms",
    "final pricing schedule and launch timeline for the Meridian Labs partnership",
]

INVOICE_TOPICS = [
    "invoice #BP-2214 for January consulting hours: net-30 terms, asks John to confirm the PO number",
    "billing discrepancy on invoice #BP-2251: 12 hours were billed against the wrong project code",
    "outstanding balance reminder with updated remittance details; asks John for a payment date",
]

RECRUITING_TOPICS = [
    "senior backend engineer candidate Maya Torres: resume review and interview loop scheduling",
    "offer package approval for the platform team candidate: level and equity band need John's sign-off",
    "debrief scheduling for the SRE candidate onsite: conflicting feedback needs John's tie-break",
    "summer intern headcount: asks John to rank the three finalists",
]

MISC_THREAD_TOPICS: list[tuple[str, str]] = [
    ("rachel", "Harborview client escalation: the nightly export job has been failing since the weekend"),
    ("alex", "flaky checkout test suite blocking the release branch"),
    ("priya", "scope cut proposal for the March release"),
    ("tom", "on-call rotation swap and a paging-policy change"),
    ("marcus", "memory leak in the billing worker after the last deploy"),
    ("rachel", "Harborview contract renewal: usage report and SLA questions"),
    ("jenna", "team offsite logistics: budget approval and date options"),
    ("sarah", "performance review calibration prep for John's reports"),
    ("alex", "load test results for the new search service"),
    ("priya", "customer feedback themes that need an engineering response"),
    ("tom", "cloud cost spike investigation: unexpected egress charges"),
    ("marcus", "proposal to adopt a new message queue for async jobs"),
    ("rachel", "pilot feature rollout plan for Harborview's ops team"),
    ("jenna", "new starter onboarding checklist needs John's sign-off"),
    ("sarah", "board demo prep: which milestones to showcase"),
    ("alex", "regression triage ownership for the mobile API"),
    ("tom", "TLS certificate rotation runbook review"),
    ("marcus", "schema change review for the events table"),
]

SINGLE_RESPOND_TOPICS = [
    "asks John for a quick decision on a dependency version upgrade",
    "requests a 30-minute meeting to walk through an architecture proposal",
    "asks John to approve conference travel for a team member",
    "urgent: production error rate spiked after this morning's deploy, needs John's call",
    "asks whether John can present at the next engineering all-hands",
    "requests John's feedback on a draft job description",
    "asks John to review and approve a customer-facing incident summary",
    "asks for John's availability for a vendor demo next week",
    "wants John's opinion on splitting the payments service",
    "asks John to nominate someone for the internal mentorship program",
    "requests sign-off on the revised data-retention settings",
    "asks John how to handle a customer request for a custom SLA",
    "asks John to confirm headcount numbers for next quarter's plan",
    "needs John's answer on whether to extend the beta program",
]

CTX_FOLLOWUP_BRIEFS = [
    "asks John to confirm the specific decision proposed in the earlier email and flags one new constraint",
    "reports that circumstances changed since the earlier email and asks how John wants to proceed",
    "asks for the exact figures and dates discussed in the earlier email before moving ahead",
    "escalates urgency: the item from the earlier email is now blocking and needs John's answer today",
    "summarizes what was agreed in the earlier email and asks John to approve the final version",
]

NOTIFY_CI_TOPICS = ["nightly build for the platform-migration branch", "main branch CI pipeline",
                    "release-candidate build 2.14", "integration test suite on staging",
                    "deploy pipeline for the api-gateway service"]
NOTIFY_CI_STATUS = ["passed", "completed with warnings", "recovered after an automatic retry"]
NOTIFY_MON_SERVICES = ["api-gateway", "billing worker", "search service", "auth service", "events pipeline"]
NOTIFY_CAL_EVENTS = ["Weekly Eng Sync", "Q1 Migration Checkpoint", "1:1 with Sarah Chen",
                     "Sprint Review", "Vendor Demo"]
NOTIFY_CAL_KINDS = ["an attendee accepted the invitation", "reminder: event is tomorrow",
                    "the organizer updated the event time"]
NOTIFY_FYI_TOPICS = ["new expense-policy thresholds effective next month",
                     "org announcement: a new data-platform team is forming",
                     "the Q1 product roadmap has been published on the wiki",
                     "security awareness training window announced",
                     "benefits enrollment window dates for this year"]
NOTIFY_HOLIDAY_TOPICS = ["office closed for Presidents' Day", "reminder: floating holiday requests due",
                         "building maintenance closure this Friday", "company holiday calendar published",
                         "early close before the long weekend", "parking garage closed Saturday"]
NOTIFY_IT_TOPICS = ["scheduled VPN maintenance window", "SSO provider upgrade this weekend",
                    "email server migration window", "laptop security patch rollout",
                    "wiki upgrade downtime announcement"]
NOTIFY_MISC_TOPICS = ["the weekly analytics report is ready to view", "your password expires in 14 days",
                      "shared drive storage quota is at 80%", "security scan digest: no critical findings",
                      "monthly license usage report generated"]

IGNORE_PROMO_TOPICS = ["limited-time discount on cloud GPU instances", "clearance sale on office chairs",
                       "exclusive offer: 40% off a monitoring suite", "last chance: conference early-bird pricing",
                       "free trial extension on a project-management tool"]
IGNORE_NEWSLETTER_TOPICS = ["weekly engineering newsletter digest", "industry news roundup",
                            "product-management newsletter issue", "devops trends digest",
                            "startup funding weekly summary"]
IGNORE_COLD_TOPICS = ["sales pitch from an outsourcing agency", "cold pitch: AI code-review tool demo request",
                      "recruiting agency offering candidate pipelines", "managed-database vendor pitch",
                      "analytics platform cold outreach asking for 15 minutes"]

GEN_SYSTEM = (
    "You write realistic workplace emails for a synthetic research corpus. "
    "Every email is addressed to John Doe, a software engineering manager at NovaCore. "
    "Follow the per-scenario metadata exactly and return strict JSON."
)

GEN_INSTRUCTIONS = (
    'For each scenario in the JSON below, write the described email. Return STRICT JSON '
    '{"emails": [{"email_id": "...", "subject": "...", "body": "..."}]} with exactly one '
    "entry per scenario, using the given email_ids. Rules: plain-text body of 50-180 words; "
    "concrete realistic details (names, numbers, dates) consistent with the scenario and its "
    "date; write in the sender's voice; automated notices read like real system emails with "
    "no personal signature; human emails end with a natural sign-off and the sender's first "
    "name; when 'in_reply_to' is present the email is a reply in that thread - reference or "
    "briefly quote the earlier message naturally and stay consistent with it; when "
    "'must_include_tokens' is present, the body MUST contain every listed token verbatim, "
    "character-for-character (e.g. write PO-8389 exactly, never PO 8389); when "
    "'write_neutral' is present, follow it strictly; NEVER use placeholder text such as "
    "[Name], [Date] or lorem ipsum."
)

_PLACEHOLDER_SUBSTRINGS = ("[insert", "[your", "[name", "[date", "[company", "lorem ipsum", "placeholder")

# ----------------------------------------------------------------------------- #
# Class-critical context dependence (triage class decidable only via a parent
# email 15-60 positions earlier). Three paired designs keep class totals exact.
# ----------------------------------------------------------------------------- #

CC_CODENAMES = ["Project Kestrel", "Project Bluefin", "Project Ironwood", "Project Sable",
                "the Atlas rollout", "the Orion cutover", "the Vega pilot", "Project Redwood",
                "the Halcyon initiative", "Project Cobalt", "the Juniper upgrade", "Project Marlin"]
CC_NAMES = ["Maya Torres", "Daniel Reyes", "Farah Iqbal", "Peter Lindqvist",
            "Anita Rao", "Victor Osei", "Lena Hartmann", "Sam Whitaker"]
CC_VENDORS = [("Northbeam Consulting", "accounts@northbeam-consulting.com"),
              ("Clearline Analytics", "partners@clearlineanalytics.com"),
              ("Vantor Security", "engage@vantorsecurity.com"),
              ("Kite Infrastructure", "hello@kiteinfra.io"),
              ("Argent Data Services", "team@argentdata.co"),
              ("Solstice Cloud Advisors", "contact@solsticecloud.io")]
CC_SERVICES = [("Statusly Alerts", "alerts@statusly.io"),
               ("BuildRadar", "notify@buildradar.dev"),
               ("MetricsHub Digest", "digest@metricshub.io"),
               ("PagerPoint", "noreply@pagerpoint.com")]
CC_COMMON_FORBIDDEN = ["as agreed", "as decided", "as you requested", "per your decision",
                       "you asked", "per your request"]

def _cc_child_brief_a(code: str) -> str:
    return (f"a brief, deliberately neutral touching-base note about {code}: the sender mentions "
            f"that things are progressing and asks if there is anything to sync on. The message "
            f"MUST NOT say who owns any action, MUST NOT request an approval or decision, and "
            f"MUST NOT say that no action is needed - it is intentionally ambiguous on its own. "
            f"2-4 short sentences. {code} MUST appear in the subject line and in the first "
            f"sentence of the body.")

def _cc_child_brief_b(vendor: str, code: str, docid: str) -> str:
    return (f"an external follow-up from {vendor}: further to the earlier conversation about "
            f"{code}, they write that they are sharing their document {docid} and remain "
            f"available. The message MUST NOT reveal whether NovaCore accepted or declined "
            f"anything and MUST NOT ask for an explicit decision - intentionally ambiguous. "
            f"2-4 short sentences. {code} MUST appear in the subject line and in the first "
            f"sentence of the body.")

def _cc_child_brief_c(service: str, code: str) -> str:
    return (f"an automated notice from {service} concerning {code}: routine status/digest "
            f"content with plausible dates and numbers. It MUST NOT mention subscription "
            f"status, cancellation, registration, or whether anyone relies on the service - "
            f"intentionally ambiguous. {code} MUST appear in the subject line and in the "
            f"first sentence of the body, and it must read like a real system email.")

#: side -> (design, child_label, child_category_pool, count)
CC_SIDES = [
    ("A", "RESPOND", "direct_request", 10),
    ("A", "NOTIFY", "fyi_announcement", 10),
    ("B", "RESPOND", "direct_request", 8),
    ("B", "IGNORE", "cold_outreach", 8),
    ("C", "NOTIFY", "system_misc", 3),
    ("C", "IGNORE", "newsletter", 5),
]


_SCHEMA_KEYS = ("email_id", "day", "ts", "thread_id", "sender", "subject", "body",
                "label", "requires_context", "context_note",
                "class_critical", "class_critical_parent")


class JSONChat(Protocol):
    """Minimal LLM interface used by pass 2 (injectable for tests)."""

    def complete_json(self, system: str, user: str) -> dict:
        """Return the parsed JSON object for one chat completion."""
        ...


class OpenAIJSON:
    """JSON-mode chat wrapper with a provider switch.

    ``provider="groq"`` routes through the shared Groq LLM wrapper (inherits
    throttling, retries, cost accounting); ``provider="openai"`` calls the
    OpenAI API directly with native JSON mode (used for dataset operations,
    which are independent of the system-under-test model).
    """

    def __init__(self, model: str, temperature: float,
                 provider: str = "groq") -> None:
        self.model = model
        self.temperature = temperature
        self.provider = provider
        if provider == "openai":
            import os
            from dotenv import load_dotenv
            from openai import OpenAI
            load_dotenv(ENV_PATH, override=False)
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(f"OPENAI_API_KEY not found in {ENV_PATH}")
            self._client = OpenAI(api_key=key)
        else:
            from experiments.common.exp_config import ExpConfig
            from experiments.common.llm import LLM
            cfg = ExpConfig(phase="p3", run_id="datasetgen", dataset_path="")
            self._llm = LLM(cfg)

    def _complete_text(self, sys_msg: str, user: str) -> str:
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model, temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": user}])
            return resp.choices[0].message.content or ""
        result = self._llm.chat(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": user}],
            model=self.model, temperature=self.temperature,
            max_tokens=4000)
        return result.content or ""

    def complete_json(self, system: str, user: str) -> dict:
        """One JSON completion; parse retries on top of provider retries."""
        sys_msg = system + ("\nRespond with a single valid JSON object only - "
                            "no prose, no code fences.")
        delay = 2.0
        for attempt in range(5):
            try:
                text = self._complete_text(sys_msg, user).strip()
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(delay)
                delay *= 2
                continue
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:]
            try:
                return json.loads(text)
            except Exception:
                start, end = text.find("{"), text.rfind("}")
                if start >= 0 and end > start:
                    try:
                        return json.loads(text[start:end + 1])
                    except Exception:
                        pass
                if attempt == 4:
                    raise
                time.sleep(delay)
                delay *= 2


# --------------------------------------------------------------------------- #
# Pass 1: scenario skeleton
# --------------------------------------------------------------------------- #

def _persona_str(key: str) -> str:
    p = PERSONAS[key]
    return f"{p['name']} <{p['email']}>"


def _pair_str(pair: tuple[str, str]) -> str:
    return f"{pair[0]} <{pair[1]}>"


def build_scenarios(seed: int) -> list[dict]:
    """Pass 1: build the finalized, chronologically sorted 500-scenario list.

    Each scenario dict carries the public dataset fields (except subject/body,
    filled in pass 2) plus private keys: ``_category``, ``_storyline``,
    ``_brief``, ``_sender_role`` and ``_parent`` (parent scenario email_id or
    None). Deterministic for a given seed.
    """
    rng = random.Random(seed)
    scen: list[dict] = []
    tcount = [0]

    def add(day: int, sender: str, sender_role: str, label: str, category: str,
            brief: str, storyline: str | None = None, thread_key: str | None = None,
            parent: dict | None = None, ctx: bool = False) -> dict:
        s = {"day": day,
             "time": f"{rng.randint(7, 18):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}",
             "sender": sender, "label": label, "requires_context": ctx,
             "_sender_role": sender_role, "_category": category, "_storyline": storyline,
             "_brief": brief, "_tkey": thread_key, "_parent_obj": parent}
        scen.append(s)
        return s

    def new_tkey() -> str:
        tcount[0] += 1
        return f"th-{tcount[0]}"

    def chain(day_lo: int, day_hi: int, sender_key: str, category: str, storyline: str,
              parent_brief: str, n_children: int) -> None:
        tk = new_tkey()
        role = PERSONAS[sender_key]["role"]
        p = add(rng.randint(day_lo, day_hi), _persona_str(sender_key), role, "RESPOND",
                category, parent_brief, storyline, tk)
        prev, day = p, p["day"]
        for _ in range(n_children):
            day += rng.randint(1, 4)
            prev = add(day, _persona_str(sender_key), role, "RESPOND", category,
                       rng.choice(CTX_FOLLOWUP_BRIEFS), storyline, tk, parent=prev, ctx=True)

    # --- RESPOND: weekly status meetings (12 threads, 8 with a follow-up) ----
    followup_weeks = sorted(rng.sample(range(12), 8))
    for w in range(12):
        tk = new_tkey()
        agenda = rng.choice(WEEKLY_AGENDA)
        p = add(7 * w + rng.randint(1, 2), _persona_str("sarah"), PERSONAS["sarah"]["role"],
                "RESPOND", "weekly_status_meeting",
                f"requests John's attendance and inputs for the week-{w + 1} status meeting; agenda: {agenda}",
                "weekly status meetings", tk)
        if w in followup_weeks:
            add(p["day"] + rng.randint(1, 2), _persona_str("sarah"), PERSONAS["sarah"]["role"],
                "RESPOND", "weekly_status_meeting",
                "follows up on the meeting request: proposes moving the time and asks John to "
                "reconfirm, referencing the agenda from the earlier email",
                "weekly status meetings", tk, parent=p, ctx=True)

    # --- RESPOND: Q1 platform migration (4 threads x 1+3) --------------------
    windows = [(4, 12), (22, 30), (40, 48), (58, 66)]
    senders = ["marcus", "tom", "priya", "marcus"]
    for (lo, hi), sk, topic in zip(windows, senders, MIGRATION_TOPICS):
        chain(lo, hi, sk, "migration_project", "Q1 platform migration",
              f"raises for John's decision: {topic}", 3)

    # --- RESPOND: partnership negotiation (3 threads x 1+3) ------------------
    for (lo, hi), topic in zip([(6, 14), (30, 38), (54, 62)], PARTNERSHIP_TOPICS):
        chain(lo, hi, "david", "partnership_negotiation", "Meridian Labs partnership negotiation",
              f"negotiation step needing John's input: {topic}", 3)

    # --- RESPOND: invoice/payment (3 threads x 1+2) --------------------------
    for (lo, hi), topic in zip([(10, 18), (36, 44), (60, 68)], INVOICE_TOPICS):
        chain(lo, hi, "grace", "invoice_payment", "Brightpath invoice and payment thread",
              f"billing matter needing John's action: {topic}", 2)

    # --- RESPOND: recruiting (4 threads x 1+2) -------------------------------
    for (lo, hi), topic in zip([(3, 11), (24, 32), (45, 53), (63, 71)], RECRUITING_TOPICS):
        chain(lo, hi, "elena", "recruiting", "recruiting thread",
              f"recruiting step needing John's input: {topic}", 2)

    # --- RESPOND: misc threads (18 threads x 1+2) ----------------------------
    for sk, topic in MISC_THREAD_TOPICS:
        chain(2, 74, sk, "misc_thread", None,
              f"raises with John, expecting a reply: {topic}", 2)

    # --- RESPOND: 27 standalone --------------------------------------------
    persona_keys = list(PERSONAS)
    for _ in range(27):
        sk = rng.choice(persona_keys)
        add(rng.randint(1, 84), _persona_str(sk), PERSONAS[sk]["role"], "RESPOND",
            "direct_request", rng.choice(SINGLE_RESPOND_TOPICS))

    # --- NOTIFY (150) --------------------------------------------------------
    def sysadd(count: int, skey: str, category: str, brief_fn) -> None:
        for _ in range(count):
            add(rng.randint(1, 84), _pair_str(SYSTEM_SENDERS[skey]), "automated system",
                "NOTIFY", category, brief_fn())

    sysadd(34, "ci", "ci_build", lambda: (
        f"automated CI notification: {rng.choice(NOTIFY_CI_TOPICS)} {rng.choice(NOTIFY_CI_STATUS)}; "
        "stage summary and durations; no action required"))
    sysadd(16, "monitoring", "monitoring", lambda: (
        f"daily monitoring digest for {rng.choice(NOTIFY_MON_SERVICES)}: latency and error-rate "
        "summary, all within thresholds; informational only"))
    sysadd(20, "calendar", "calendar_notice", lambda: (
        f"calendar notification for '{rng.choice(NOTIFY_CAL_EVENTS)}': {rng.choice(NOTIFY_CAL_KINDS)}"))
    for _ in range(28):
        sk = rng.choice(["jenna", "sarah", "priya"])
        add(rng.randint(1, 84), _persona_str(sk), PERSONAS[sk]["role"], "NOTIFY",
            "fyi_announcement", f"FYI announcement, no reply expected: {rng.choice(NOTIFY_FYI_TOPICS)}")
    for _ in range(6):
        add(rng.randint(1, 84), _persona_str("jenna"), PERSONAS["jenna"]["role"], "NOTIFY",
            "holiday_notice", f"holiday/office notice: {rng.choice(NOTIFY_HOLIDAY_TOPICS)}")
    sysadd(14, "it", "it_maintenance", lambda: (
        f"IT notice: {rng.choice(NOTIFY_IT_TOPICS)}; what to expect and when; no reply needed"))
    for _ in range(32):
        skey = rng.choice(["reports", "security", "it"])
        add(rng.randint(1, 84), _pair_str(SYSTEM_SENDERS[skey]), "automated system", "NOTIFY",
            "system_misc", f"automated notice: {rng.choice(NOTIFY_MISC_TOPICS)}")

    # --- IGNORE (200) --------------------------------------------------------
    for _ in range(70):
        add(rng.randint(1, 84), _pair_str(rng.choice(SPAM_SENDERS)), "mass mailer", "IGNORE",
            "promo", f"promotional blast, irrelevant to John's work: {rng.choice(IGNORE_PROMO_TOPICS)}")
    for _ in range(60):
        add(rng.randint(1, 84), _pair_str(rng.choice(SPAM_SENDERS)), "mass mailer", "IGNORE",
            "newsletter", f"subscription newsletter: {rng.choice(IGNORE_NEWSLETTER_TOPICS)}")
    for _ in range(30):
        add(rng.randint(1, 84), _pair_str(rng.choice(AUTOREPLY_SENDERS)), "external contact",
            "IGNORE", "auto_reply",
            "automatic out-of-office reply to a message John sent; gives return date and an alternate contact")
    for _ in range(40):
        add(rng.randint(1, 84), _pair_str(rng.choice(SPAM_SENDERS)), "cold outreach", "IGNORE",
            "cold_outreach", f"unsolicited cold email: {rng.choice(IGNORE_COLD_TOPICS)}")

    return _inject_class_critical(_finalize(scen), seed)


def _finalize(scen: list[dict]) -> list[dict]:
    """Sort chronologically, assign email/thread ids, resolve context notes."""
    scen.sort(key=lambda s: (s["day"], s["time"]))
    for i, s in enumerate(scen):
        s["email_id"] = f"syn-{i + 1:04d}"
        s["ts"] = f"{(BASE_DATE + _dt.timedelta(days=s['day'] - 1)).isoformat()}T{s['time']}"
    tmap: dict[Any, str] = {}
    for s in scen:
        key = s["_tkey"] if s["_tkey"] is not None else s["email_id"]
        s["thread_id"] = tmap.setdefault(key, f"t-{len(tmap) + 1:03d}")
    for s in scen:
        parent = s.pop("_parent_obj")
        s["_parent"] = parent["email_id"] if parent else None
        if s["requires_context"] and parent is not None:
            s["context_note"] = (
                f"Follow-up: correct handling requires the content of earlier email "
                f"{parent['email_id']} in thread {s['thread_id']} ({s['_category']}).")
        else:
            s["context_note"] = ""
        del s["_tkey"], s["time"]
    return scen


def _inject_class_critical(scen: list[dict], seed: int) -> list[dict]:
    """Convert ~70 scenarios into class-critical children with distant parents.

    Labels are never changed, so class totals stay exact. Children get a
    deliberately ambiguous brief whose correct class follows only from a
    RESPOND parent 12-120 positions earlier whose brief is extended with an
    explicit decisive signal plus two anchor tokens absent from the child.
    Design A children join the parent's thread (natural check-in reply);
    designs B and C stay unthreaded (external/service mail) and link to the
    parent only via the shared codename and the context note.
    """
    rng = random.Random(seed ^ 0xCC)
    idx = {s["email_id"]: i for i, s in enumerate(scen)}
    used_child: set[int] = set()
    used_parent: set[int] = set()
    used_parent_threads: set[str] = set()

    def eligible_parent(i_child: int) -> int | None:
        for j in range(max(0, i_child - 120), i_child - 11):
            s = scen[j]
            if (s["label"] == "RESPOND" and j not in used_parent
                    and j not in used_child and not s.get("class_critical")
                    and s["_category"] != "direct_request"
                    and s["thread_id"] not in used_parent_threads):
                return j
        return None

    for design, child_label, pool_cat, count in CC_SIDES:
        candidates = [i for i, s in enumerate(scen)
                      if s["label"] == child_label and s["_category"] == pool_cat
                      and s["_parent"] is None and i >= 16 and i not in used_child]
        done = 0
        for i in candidates:
            if done == count:
                break
            j = eligible_parent(i)
            if j is None:
                continue
            child, parent = scen[i], scen[j]
            code = CC_CODENAMES[(len(used_child)) % len(CC_CODENAMES)]
            name = rng.choice(CC_NAMES)
            ida = f"PO-{rng.randint(4100, 9899)}" if design != "C" else f"SUB-{rng.randint(3100, 9899)}"
            if design == "A":
                if child_label == "RESPOND":
                    decisive = (f"it explicitly records that John owns the next action on {code}: "
                                f"review {ida} with {name} and reply with his decision")
                    forbidden = ["you own", "assigned to", "please review", "action required",
                                 "awaiting your", "need your", "your decision"]
                    reason = "parent assigns the open action to John, so a reply is required"
                else:
                    decisive = (f"it explicitly records that after this exchange the sender "
                                f"is taking {code} forward end-to-end: the sender will process "
                                f"{ida} with {name}, John's part is complete once he sends this "
                                f"reply, and no further action will be needed from John on {code}")
                    forbidden = ["no action", "i'll handle", "i will handle", "handled",
                                 "taken care", "your part is done", "no further action", "fyi"]
                    reason = "parent states the sender handles everything, so this is informational"
                child["sender"], child["_sender_role"] = parent["sender"], parent["_sender_role"]
                child["_brief"] = _cc_child_brief_a(code)
                child["_parent"] = parent["email_id"]
                child["thread_id"] = parent["thread_id"]
            elif design == "B":
                vendor = rng.choice(CC_VENDORS)
                docid = f"DOC-{rng.randint(2100, 9899)}"
                if child_label == "RESPOND":
                    decisive = (f"it records the decision that NovaCore is engaging "
                                f"{vendor[0]} for {code}: John asked {name} to have the vendor "
                                f"send their statement of work {ida} over for John's review")
                    forbidden = ["agreed to engage", "approved", "as promised"]
                    reason = "parent shows John requested this document, so a reply is required"
                else:
                    decisive = (f"it records the decision that NovaCore declined {vendor[0]} "
                                f"for {code} under procurement reference {ida}: John asked "
                                f"{name} to ensure the vendor is not engaged and said any "
                                f"further mail from them can be disregarded")
                    forbidden = ["declined", "not interested", "do not contact", "disregard",
                                 "unsubscribe"]
                    reason = "parent shows John declined this vendor, so this mail can be ignored"
                child["sender"] = f"{vendor[0]} <{vendor[1]}>"
                child["_sender_role"] = "external vendor contact"
                child["_brief"] = _cc_child_brief_b(vendor[0], code, docid)
                child["_parent"] = None
            else:
                service = rng.choice(CC_SERVICES)
                if child_label == "NOTIFY":
                    decisive = (f"it records that the team now depends on {service[0]} for "
                                f"{code} monitoring: John confirmed subscription {ida} and "
                                f"asked {name} to route its notices to his inbox")
                    forbidden = ["depends on", "registered", "confirmed subscription", "critical"]
                    reason = "parent shows the team relies on this service, so its notices matter"
                else:
                    decisive = (f"it records that {service[0]} was decommissioned for {code}: "
                                f"John confirmed cancellation {ida} and said remaining "
                                f"automated mails from it can be ignored")
                    forbidden = ["decommission", "cancel", "ignored", "unsubscribed"]
                    reason = "parent shows the service was decommissioned, so its mail is noise"
                child["sender"] = f"{service[0]} <{service[1]}>"
                child["_sender_role"] = "automated system"
                child["_brief"] = _cc_child_brief_c(service[0], code)
                child["_parent"] = None
            parent["_brief"] = (parent["_brief"] +
                                f"; in this email the sender also refers to this whole "
                                f"workstream by its internal codename {code} (the email must "
                                f"use that codename naturally), and explicitly records: "
                                f"{decisive}. The email body MUST state this clearly and MUST "
                                f"include the exact tokens '{ida}' and '{name}'.")
            parent["_anchors"] = [ida, name]
            parent["class_critical_parent"] = True
            child["class_critical"] = True
            child["requires_context"] = True
            child["_cc_design"] = design
            child["_cc_code"] = code
            child["_category"] = f"cc_design_{design.lower()}"
            child["_forbidden_phrases"] = [p.lower() for p in forbidden + CC_COMMON_FORBIDDEN]
            child["context_note"] = (f"{parent['email_id']}: class depends on parent - {reason} "
                                     f"(design {design}, {code})")
            used_child.add(i)
            used_parent.add(j)
            used_parent_threads.add(parent["thread_id"])
            done += 1
        if done != count:
            raise RuntimeError(f"class-critical injection: side {design}/{child_label} "
                               f"filled {done}/{count}")
    for s in scen:
        s.setdefault("class_critical", False)
        s.setdefault("class_critical_parent", False)
    return scen


# --------------------------------------------------------------------------- #
# Pilot subset
# --------------------------------------------------------------------------- #

def pilot_ids(scenarios: list[dict]) -> list[str]:
    """Deterministic balanced 20-email pilot subset (8 IGNORE / 6 NOTIFY / 6 RESPOND).

    RESPOND slots come from the first two threads whose RESPOND parent has >=2
    context-dependent RESPOND children (parent + 2 children each), so the >=4
    context-dependent emails have their referenced parents inside the pilot.
    """
    by_thread: dict[str, list[dict]] = {}
    order: list[str] = []
    for s in scenarios:
        if s["thread_id"] not in by_thread:
            order.append(s["thread_id"])
        by_thread.setdefault(s["thread_id"], []).append(s)

    chosen: list[dict] = []
    used = 0
    for tid in order:
        mem = by_thread[tid]
        if len(mem) < 3 or mem[0]["label"] != "RESPOND":
            continue
        ctx_children = [m for m in mem[1:] if m["requires_context"] and m["label"] == "RESPOND"]
        if len(ctx_children) < 2:
            continue
        chosen.extend([mem[0]] + ctx_children[:2])
        used += 1
        if used == 2:
            break
    if used != 2:
        raise RuntimeError("pilot selection: not enough qualifying context threads")

    def spread(items: list[dict], k: int) -> list[dict]:
        n = len(items)
        return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]

    chosen += spread([s for s in scenarios
                      if s["label"] == "IGNORE" and not s.get("class_critical")], 8)
    chosen += spread([s for s in scenarios
                      if s["label"] == "NOTIFY" and not s.get("class_critical")], 6)
    chosen.sort(key=lambda s: (s["day"], s["ts"]))
    return [s["email_id"] for s in chosen]


def ccpilot_slice(scenarios: list[dict], n: int = 60, min_cc: int = 12) -> list[dict]:
    """Deterministic contiguous slice of ``n`` scenarios containing at least
    ``min_cc`` class-critical children whose parents are also inside the slice
    at distance >= 12 positions."""
    idx = {s["email_id"]: i for i, s in enumerate(scenarios)}

    def parent_index(s: dict) -> int | None:
        m = re.search(r"(syn-\d{4})", s.get("context_note") or "")
        return idx.get(m.group(1)) if m else None

    best_start, best_count = None, -1
    for start in range(0, len(scenarios) - n + 1):
        end = start + n
        count = 0
        for i in range(start, end):
            s = scenarios[i]
            if not s.get("class_critical"):
                continue
            j = parent_index(s)
            if j is not None and start <= j and i - j >= 12:
                count += 1
        if count > best_count:
            best_start, best_count = start, count
    if best_count < min_cc:
        raise RuntimeError(f"ccpilot: best window has only {best_count} in-slice "
                           f"class-critical pairs (need {min_cc})")
    return scenarios[best_start:best_start + n]


# --------------------------------------------------------------------------- #
# Pass 2: text generation
# --------------------------------------------------------------------------- #

def _make_batches(scenarios: list[dict], size: int = 5) -> list[list[dict]]:
    """Chronological batches; flush early when a scenario's parent is in the
    open batch so parent text is always generated before the child's call."""
    batches: list[list[dict]] = []
    cur: list[dict] = []
    for s in scenarios:
        if cur and s["_parent"] in {c["email_id"] for c in cur}:
            batches.append(cur)
            cur = []
        cur.append(s)
        if len(cur) == size:
            batches.append(cur)
            cur = []
    if cur:
        batches.append(cur)
    return batches


def _strip_re(subject: str) -> str:
    return re.sub(r"^(?:(?:re|fw|fwd)\s*:\s*)+", "", subject.strip(), flags=re.I)


def _batch_user_prompt(batch: list[dict], generated: dict[str, dict[str, str]]) -> str:
    items = []
    for s in batch:
        item: dict[str, Any] = {
            "email_id": s["email_id"], "date": s["ts"], "from": s["sender"],
            "from_role": s["_sender_role"], "label": s["label"],
            "category": s["_category"], "scenario": s["_brief"],
        }
        if s["_storyline"]:
            item["ongoing_storyline"] = s["_storyline"]
        if s.get("_anchors"):
            item["must_include_tokens"] = s["_anchors"]
        if s.get("_cc_code"):
            item["must_include_tokens"] = [s["_cc_code"]]
        if s.get("class_critical"):
            item["write_neutral"] = ("This email is deliberately ambiguous on its own: keep it "
                                     "brief and neutral, do NOT restate any decision, ownership, "
                                     "commitment, or subscription status from any earlier email.")
        if s["_parent"] is not None:
            if s["_parent"] in generated:
                pg = generated[s["_parent"]]
                item["in_reply_to"] = {"subject": pg["subject"], "body": pg["body"]}
            else:  # subset generation without the parent present
                item["in_reply_to"] = {"subject": "(earlier email in this thread)",
                                       "body": f"(summary) {s['_brief']}"}
        items.append(item)
    return GEN_INSTRUCTIONS + "\n\nSCENARIOS JSON:\n" + json.dumps({"emails": items})


def _validate_batch(out: dict, batch: list[dict]) -> dict[str, tuple[str, str]]:
    """Validate one LLM batch result; raise ValueError on any problem."""
    if not isinstance(out, dict) or not isinstance(out.get("emails"), list):
        raise ValueError("missing 'emails' list")
    got: dict[str, tuple[str, str]] = {}
    for e in out["emails"]:
        if not isinstance(e, dict):
            raise ValueError("non-dict email entry")
        eid, subj, body = e.get("email_id"), e.get("subject"), e.get("body")
        if not (isinstance(eid, str) and isinstance(subj, str) and isinstance(body, str)):
            raise ValueError("bad field types")
        if not subj.strip() or len(subj) > 200 or len(body.split()) < 15:
            raise ValueError(f"degenerate subject/body for {eid}")
        low = body.lower()
        if any(p in low for p in _PLACEHOLDER_SUBSTRINGS):
            raise ValueError(f"placeholder text in {eid}")
        got[eid] = (subj.strip(), body.strip())
    by_id = {s["email_id"]: s for s in batch}
    for eid, (subj, body) in got.items():
        s = by_id.get(eid)
        if s is None:
            continue
        low = (subj + "\n" + body).lower()
        for tok in s.get("_anchors") or []:
            if tok.lower() not in low:
                raise ValueError(f"anchor token '{tok}' missing from {eid}")
        for phrase in s.get("_forbidden_phrases") or []:
            if phrase in low:
                raise ValueError(f"forbidden phrase '{phrase}' leaked into {eid}")
        code = s.get("_cc_code")
        if code:
            head = (subj + "\n" + body[:200]).lower()
            if code.lower() not in head:
                raise ValueError(f"codename '{code}' missing from subject/opening of {eid}")
    want = [s["email_id"] for s in batch]
    if set(got) != set(want):
        raise ValueError(f"email_id mismatch: want {want}, got {sorted(got)}")
    return got


def generate_bodies(scenarios: list[dict], llm: JSONChat, progress: bool = False) -> list[dict]:
    """Pass 2: fill subject/body for every scenario via ``llm``; return final records.

    Follow-up subjects are normalized to ``Re: <parent subject>``. Records are
    returned in the input (chronological) order with exactly the pinned schema keys.
    """
    generated: dict[str, dict[str, str]] = {}
    batches = _make_batches(scenarios)
    for bi, batch in enumerate(batches):
        user = _batch_user_prompt(batch, generated)
        last_err: Exception | None = None
        got: dict[str, tuple[str, str]] | None = None
        for _ in range(3):
            try:
                got = _validate_batch(llm.complete_json(GEN_SYSTEM, user), batch)
                break
            except ValueError as err:
                last_err = err
        if got is None:
            # Some models return fewer emails than requested per call;
            # degrade to one-scenario-per-call for this batch.
            if progress:
                print(f"[gen] batch {bi + 1}: batch mode failed "
                      f"({last_err}); retrying scenarios individually")
            got = {}
            for s in batch:
                single_err: Exception | None = None
                for _ in range(3):
                    try:
                        one = _validate_batch(
                            llm.complete_json(
                                GEN_SYSTEM, _batch_user_prompt([s], generated)),
                            [s])
                        got.update(one)
                        break
                    except ValueError as err:
                        single_err = err
                else:
                    raise RuntimeError(
                        f"batch {bi}: scenario {s['email_id']} invalid after "
                        f"batch and single retries: {single_err}")
        for s in batch:
            subj, body = got[s["email_id"]]
            if s["_parent"] in generated:
                subj = "Re: " + _strip_re(generated[s["_parent"]]["subject"])
            generated[s["email_id"]] = {"subject": subj, "body": body}
        if progress:
            print(f"[gen] batch {bi + 1}/{len(batches)} done")
    records = []
    for s in scenarios:
        g = generated[s["email_id"]]
        records.append({"email_id": s["email_id"], "day": s["day"], "ts": s["ts"],
                        "thread_id": s["thread_id"], "sender": s["sender"],
                        "subject": g["subject"], "body": g["body"], "label": s["label"],
                        "requires_context": s["requires_context"],
                        "context_note": s["context_note"],
                        "class_critical": s.get("class_critical", False),
                        "class_critical_parent": s.get("class_critical_parent", False)})
    return records


# --------------------------------------------------------------------------- #
# Output + stats
# --------------------------------------------------------------------------- #

def write_jsonl(records: Iterable[dict], path: Path) -> None:
    """Write records as one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def corpus_stats(scenarios: list[dict]) -> dict:
    """Class counts, thread stats and context-dependency counts for stats.json."""
    sizes = Counter(s["thread_id"] for s in scenarios)
    multi = sorted((n for n in sizes.values() if n > 1), reverse=True)
    return {
        "n_emails": len(scenarios),
        "class_counts": dict(Counter(s["label"] for s in scenarios)),
        "n_threads": len(sizes),
        "n_multi_threads": len(multi),
        "multi_thread_emails": sum(multi),
        "max_thread_len": multi[0] if multi else 1,
        "n_context_dependent": sum(1 for s in scenarios if s["requires_context"]),
        "n_class_critical": sum(1 for s in scenarios if s.get("class_critical")),
        "class_critical_by_design": dict(Counter(
            f"{s.get('_cc_design')}-{s['label']}" for s in scenarios if s.get("class_critical"))),
        "day_min": min(s["day"] for s in scenarios),
        "day_max": max(s["day"] for s in scenarios),
        "pilot_ids": pilot_ids(scenarios),
    }


# --------------------------------------------------------------------------- #
# Selftest (no OpenAI)
# --------------------------------------------------------------------------- #

class FakeLLM:
    """Deterministic pass-2 stand-in; counts follow-ups that got parent text."""

    def __init__(self) -> None:
        self.calls = 0
        self.parented = 0

    def complete_json(self, system: str, user: str) -> dict:
        payload = json.loads(user.split("SCENARIOS JSON:\n", 1)[1])
        out = []
        for item in payload["emails"]:
            if "in_reply_to" in item:
                self.parented += 1
            extra = " ".join(item.get("must_include_tokens", []))
            out.append({"email_id": item["email_id"],
                        "subject": f"About {item['category']} {item['email_id']} {extra}".strip(),
                        "body": (f"{extra} Canned deterministic body for {item['email_id']} "
                                 "covering the scenario in enough words to pass every "
                                 "length validation check applied by the generator today.").strip()})
        self.calls += 1
        return {"emails": out}


def _public(s: dict) -> dict:
    return {k: v for k, v in s.items() if not k.startswith("_")} | {"_parent": s["_parent"]}


def _selftest() -> None:
    """Pass-1 checks + pass-2 with FakeLLM. No OpenAI calls."""
    s1 = build_scenarios(13)
    s2 = build_scenarios(13)
    assert json.dumps([_public(x) for x in s1]) == json.dumps([_public(x) for x in s2]), \
        "pass 1 not deterministic"

    assert len(s1) == 500
    counts = Counter(s["label"] for s in s1)
    assert counts == {"IGNORE": 200, "NOTIFY": 150, "RESPOND": 150}, counts
    assert all(1 <= s["day"] <= 84 for s in s1)
    keys = [(s["day"], s["ts"]) for s in s1]
    assert keys == sorted(keys), "not sorted by (day, ts)"
    assert [s["email_id"] for s in s1] == [f"syn-{i + 1:04d}" for i in range(500)]

    by_id = {s["email_id"]: s for s in s1}
    sizes = Counter(s["thread_id"] for s in s1)
    assert sum(1 for n in sizes.values() if n > 1) >= 40, "expected at least 40 multi threads"
    ctx = [s for s in s1 if s["requires_context"]]
    assert len(ctx) >= 100, len(ctx)
    for s in ctx:
        m = re.search(r"(syn-\d{4})", s["context_note"])
        assert m, f"context_note misses parent id: {s['context_note']}"
        parent = by_id[m.group(1)]
        if s["_parent"]:
            assert parent["thread_id"] == s["thread_id"]
        assert (parent["day"], parent["ts"]) < (s["day"], s["ts"]), "parent not earlier"

    # --- class-critical properties ---------------------------------------
    pos = {s["email_id"]: i for i, s in enumerate(s1)}
    cc = [s for s in s1 if s["class_critical"]]
    assert len(cc) == 44, len(cc)
    from collections import Counter as _C
    pairs = _C((s["_cc_design"], s["label"]) for s in cc)
    assert pairs == {("A", "RESPOND"): 10, ("A", "NOTIFY"): 10, ("B", "RESPOND"): 8,
                     ("B", "IGNORE"): 8, ("C", "NOTIFY"): 3, ("C", "IGNORE"): 5}, pairs
    for s in cc:
        m = re.search(r"(syn-\d{4})", s["context_note"])
        parent = by_id[m.group(1)]
        d = pos[s["email_id"]] - pos[parent["email_id"]]
        assert 12 <= d <= 120, (s["email_id"], d)
        assert parent["label"] == "RESPOND" and parent["class_critical_parent"]
        assert len(parent["_anchors"]) == 2
        for tok in parent["_anchors"]:
            assert tok.lower() not in s["_brief"].lower(), (s["email_id"], tok)
        assert s["_forbidden_phrases"]

    # forbidden-phrase validator: rejects a planted leak, passes clean text
    probe = next(s for s in cc if s["_cc_design"] == "A" and s["label"] == "NOTIFY")
    bad = {"emails": [{"email_id": probe["email_id"], "subject": "Re: x",
                       "body": ("I will handle everything from here so there is truly "
                                "no action needed from you on this item going forward, "
                                "just wanted to give a quick update about the timeline.")}]}
    try:
        _validate_batch(bad, [probe])
        raise AssertionError("forbidden-phrase leak not caught")
    except ValueError:
        pass
    ok = {"emails": [{"email_id": probe["email_id"],
                      "subject": f"Re: {probe['_cc_code']}",
                      "body": (f"{probe['_cc_code']} is moving along on our side. Quick note "
                               "to touch base and see if there is anything to sync on "
                               "regarding the timeline and the remaining items of the work.")}]}
    _validate_batch(ok, [probe])

    # ccpilot slice property
    csl = ccpilot_slice(s1, 140)
    cidx = {s["email_id"]: i for i, s in enumerate(csl)}
    in_cc = [s for s in csl if s["class_critical"]
             and re.search(r"(syn-\d{4})", s["context_note"]).group(1) in cidx
             and cidx[s["email_id"]] - cidx[re.search(r"(syn-\d{4})", s["context_note"]).group(1)] >= 12]
    assert len(in_cc) >= 12, len(in_cc)
    for s in s1:
        if s["_parent"]:
            assert by_id[s["_parent"]]["thread_id"] == s["thread_id"]

    pids = pilot_ids(s1)
    assert len(pids) == len(set(pids)) == 20
    psub = [by_id[i] for i in pids]
    pcounts = Counter(s["label"] for s in psub)
    assert pcounts == {"IGNORE": 8, "NOTIFY": 6, "RESPOND": 6}, pcounts
    pctx = [s for s in psub if s["requires_context"]]
    assert len(pctx) >= 4
    pset = set(pids)
    assert all(s["_parent"] in pset for s in pctx), "pilot context parent missing from pilot"

    fake = FakeLLM()
    records = generate_bodies(s1, fake)
    n_children = sum(1 for s in s1 if s["_parent"])
    assert fake.parented == n_children >= 60
    rec_by_id = {r["email_id"]: r for r in records}
    for s in s1:
        if s.get("_anchors"):
            body = rec_by_id[s["email_id"]]["body"].lower()
            assert all(t.lower() in body for t in s["_anchors"])
    assert len(records) == 500
    for r, s in zip(records, s1):
        assert tuple(r.keys()) == _SCHEMA_KEYS
        assert r["subject"] and len(r["body"].split()) >= 15
        if s["_parent"]:
            assert r["subject"].startswith("Re: "), r["subject"]
    records_b = generate_bodies(build_scenarios(13), FakeLLM())
    assert records == records_b, "pass 2 with fake LLM not deterministic"

    stats = corpus_stats(s1)
    assert stats["n_multi_threads"] >= 40 and stats["n_context_dependent"] >= 100
    assert stats["n_class_critical"] == 44
    print("SELFTEST OK generate_synthetic: 500 scenarios, "
          f"{stats['n_multi_threads']} multi-threads, "
          f"{stats['n_context_dependent']} context-dependent, "
          f"{stats['n_class_critical']} class-critical, pilot 20 balanced")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> None:
    """CLI entry point (see module docstring)."""
    ap = argparse.ArgumentParser(description="Synthetic email corpus generator")
    ap.add_argument("--pilot", action="store_true",
                    help="generate only the 20-email pilot subset (4 gpt-4o calls)")
    ap.add_argument("--out", type=str, default=None, help="output jsonl path")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ccpilot", type=int, default=None,
                    help="generate a contiguous slice of N emails containing >=12 "
                         "class-critical children with in-slice parents (pilot for "
                         "the class-critical evaluation)")
    ap.add_argument("--first", type=int, default=None,
                    help="keep only the first N scenarios chronologically "
                         "(parents always precede children, so the slice is "
                         "thread-consistent); used for validation splits")
    ap.add_argument("--selftest", action="store_true", help="run offline selftest (no OpenAI)")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return

    scenarios = build_scenarios(args.seed)
    ids = pilot_ids(scenarios)
    llm = OpenAIJSON(model="gpt-4o-mini", temperature=0.7, provider="openai")

    if args.ccpilot:
        subset = ccpilot_slice(scenarios, args.ccpilot)
        records = generate_bodies(subset, llm, progress=True)
        out = Path(args.out) if args.out else DATA_DIR / f"synthetic_ccpilot_{args.ccpilot}.jsonl"
        write_jsonl(records, out)
        ncc = sum(1 for r in records if r.get("class_critical"))
        print(f"[gen] wrote {len(records)} ccpilot emails ({ncc} class-critical) -> {out}")
        return

    if args.pilot:
        subset = [s for s in scenarios if s["email_id"] in set(ids)]
        records = generate_bodies(subset, llm, progress=True)
        out = Path(args.out) if args.out else DATA_DIR / "synthetic_pilot_20.jsonl"
        write_jsonl(records, out)
        print(f"[gen] wrote {len(records)} pilot emails -> {out}")
        return

    if args.first is not None:
        scenarios = scenarios[:args.first]
        records = generate_bodies(scenarios, llm, progress=True)
        out = Path(args.out) if args.out else DATA_DIR / f"synthetic_first{args.first}.jsonl"
        write_jsonl(records, out)
        print(f"[gen] wrote {len(records)} emails (first {args.first}) -> {out}")
        return

    records = generate_bodies(scenarios, llm, progress=True)
    out = Path(args.out) if args.out else DATA_DIR / "synthetic_500.jsonl"
    write_jsonl(records, out)
    pilot_out = out.parent / "synthetic_pilot_20.jsonl"
    idset = set(ids)
    write_jsonl([r for r in records if r["email_id"] in idset], pilot_out)
    stats_path = out.parent / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump({"seed": args.seed} | corpus_stats(scenarios), fh, indent=2)
    print(f"[gen] wrote {len(records)} emails -> {out}; pilot -> {pilot_out}; stats -> {stats_path}")


if __name__ == "__main__":
    main()
