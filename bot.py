import logging
import sys
from logging.handlers import RotatingFileHandler
import os
import platform
import random
import re
import asyncio
import apprise
import pandas as pd
import time

from pathlib import Path
from jobspy import scrape_jobs
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session as SessionType
from sqlalchemy import Column, Integer, String
from openai import (
    APIConnectionError,
    APITimeoutError,
    APIError,
    AuthenticationError,
    InternalServerError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    AsyncOpenAI,
)
from pydantic import BaseModel
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

# USER CUSTOMIZATION
# Edit these values to control which jobs are searched, filtered, and pinged.
SEARCH_SITES = ['linkedin', 'indeed', 'glassdoor']
SEARCH_KEYWORDS = [
    "computer science",
    "software",
    "devops",
    "developer",
]
SEARCH_QUERY = " OR ".join(f"({keyword})" if " " in keyword else keyword for keyword in SEARCH_KEYWORDS)

REQUIRED_TITLE_KEYWORDS = [
    "engineer", "technology", "developer", "software", "new grad", "entry level", "entry",
    "data", "sde", "it", "programmer", "machine learning", "ml", "ai", "firmware",
    "embedded", "cloud", "devops", "analyst", "cybersecurity", "automation",
]

BLOCKED_TITLE_KEYWORDS = {
    "senior",
    "lead",
    "manager",
    "director",
    "principal",
    "vp",
    "Sr.",
    "Sr",
    "Senior",
    "Lead",
    "Manager",
    "Director",
    "Principal",
    "VP",
    "sr.",
    "Snr",
    "II",
    "III",
    "president"
}

BLACKLISTED_COMPANIES = {
    'Team Remotely Inc',
    'HireMeFast LLC',
    'Get It Recruit - Information Technology',
    "Offered.ai",
    "4 Staffing Corp",
    "myGwork - LGBTQ+ Business Community",
    "Patterned Learning AI",
    "Mindpal",
    "Phoenix Recruiting",
    "SkyRecruitment",
    "Phoenix Recruitment",
    "Patterned Learning Career",
    "SysMind",
    "SysMind LLC",
    "Motion Recruitment",
    "DataAnnotation",
    "BeaconFire Inc.",
    "Helic & Co.",
    "ShrinQ Consulting Group Inc",
    "New Relic",
    "General Dynamics Mission Systems",
    "Jobs via Dice",
    "Lensa",
    "Jobright.ai",
}

# Pinger customization: jobs with matching locations get the mention prepended.
PING_MENTION = "@everyone"
PING_LOCATION_STATE_CODES = [
    "OR",
    "WA",
]
PING_LOCATION_NAMES = [
    "Oregon",
    "Washington",
]

OPENAI_BASE_URL = (
    os.getenv("OPENAI_BASE_URL")
    or os.getenv("OPENAI_API_BASE_URL")
    or os.getenv("API_URL")
)
gpt = AsyncOpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else AsyncOpenAI()
apobj = apprise.Apprise()
Base = declarative_base()

def _load_apprise_urls(apobj: apprise.Apprise, env_var: str, tag: str) -> None:
    raw = os.getenv(env_var, "")
    for url in raw.split(","):
        url = url.strip()
        if url:
            apobj.add(url, tag=tag)

_load_apprise_urls(apobj, "FT_APPRISE_URLS", "ft")

# Read Resume.pdf — exits with a clear message if missing or unreadable
try:
    _reader = PdfReader(str(BASE_DIR / "Resume.pdf"))
    RESUME = _reader.pages[0].extract_text() or ""
    del _reader
except FileNotFoundError:
    print("ERROR: Resume.pdf not found in the project directory. Please add it before running.", file=sys.stderr)
    sys.exit(1)
if not RESUME:
    print("ERROR: Resume.pdf could not be read (possibly an image-only PDF). Please use a text-based PDF.", file=sys.stderr)
    sys.exit(1)

CACHE_ID = str(time.time())

SYSTEM_PROMPT = f"""You are a Strict Application Auditor. Your task is to filter a specific candidate's resume against various job descriptions.

# PRIMARY DIRECTIVE: QUANTITATIVE AUDIT
You must evaluate the candidate based on the following strict rules. If any rule is violated, `apply` must be false.

1.  **YEARS OF EXPERIENCE (STRICT CALCULATION):**
    -   **FILTER STEP (CRITICAL):** You must IGNORE and EXCLUDE any experience entries labeled as:
        * "Independent" / "Independent Seller"
        * "Sole Proprietor" / "Self-Employed" / "Freelance"
        * "Founder" (unless for a Venture Backed startup)
    -   **CALCULATION:** Sum the years of *only* the remaining corporate/W2 employment roles.
    -   **COMPARE:** If (Valid Corporate Years) < (Required Years - 1), REJECT.
    -   *Reason format: "Mismatch: JD requires 5 years, Resume has [X] valid corporate years (excluded Independent role)."*

2.  **SENIORITY MISMATCH:**
    -   Reject if JD asks for Senior/Lead/Principal and Resume is Entry/Junior/Intern.
    -   *Reason format: "Mismatch: Seniority level (Junior vs Lead)."*

3.  **MANDATORY SKILLS:**
    -   Reject if a "Must Have" or "Required" hard skill is completely absent.
    -   *Reason format: "Missing core skill: Kubernetes."*

4.  **CLEARANCE/LEGAL:**
    -   Reject if JD requires Security Clearance or Citizenship and Resume does not specify it.
    -   *Reason format: "Missing mandatory Security Clearance."*

# OUTPUT FORMAT
You must return a single JSON object. Do not add markdown formatting.
{{
  "apply": boolean,
  "reason": string
}}

### CANDIDATE RESUME:
\"\"\"
{RESUME}
\"\"\"
"""

class ApplicationAnalyzer(BaseModel):
    apply: bool
    reason: str

class FullTimeJob(Base):
    __tablename__ = "full_time_jobs"

    id = Column(Integer, primary_key=True)
    description = Column(String)
    job_id = Column(String, unique=True)
    application_url = Column(String)
    job_title = Column(String)
    company_name = Column(String)
    company_url = Column(String)
    location = Column(String)

class LoggingFormatter(logging.Formatter):
    black = "\x1b[30m"
    red = "\x1b[31m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    blue = "\x1b[34m"
    gray = "\x1b[38m"
    reset = "\x1b[0m"
    bold = "\x1b[1m"

    COLORS = {
        logging.DEBUG: gray + bold,
        logging.INFO: blue + bold,
        logging.WARNING: yellow + bold,
        logging.ERROR: red,
        logging.CRITICAL: red + bold,
    }

    _formatter_cache: dict = {}

    def format(self, record):
        levelno = record.levelno
        if levelno not in self._formatter_cache:
            log_color = self.COLORS.get(levelno, self.reset)
            fmt = "(black){asctime}(reset) (levelcolor){levelname:<8}(reset) (green){name}(reset) {message}"
            fmt = fmt.replace("(black)", self.black + self.bold)
            fmt = fmt.replace("(reset)", self.reset)
            fmt = fmt.replace("(levelcolor)", log_color)
            fmt = fmt.replace("(green)", self.green + self.bold)
            self._formatter_cache[levelno] = logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S", style="{")
        return self._formatter_cache[levelno].format(record)

logger = logging.getLogger("discord_bot")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(LoggingFormatter())
file_handler = RotatingFileHandler(filename=str(BASE_DIR / "discord.log"), encoding="utf-8", maxBytes=10*1024*1024, backupCount=3)
file_handler_formatter = logging.Formatter(
    "[{asctime}] [{levelname:<8}] {name}: {message}", "%Y-%m-%d %H:%M:%S", style="{"
)
file_handler.setFormatter(file_handler_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

engine = create_engine(f"sqlite:///{BASE_DIR / 'jobs.db'}", echo=False)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

class JobScraperApp:
    def __init__(self, apobj: apprise.Apprise) -> None:
        self.logger = logger
        self.apobj = apobj

        self.full_time_configured = bool(os.getenv("FT_APPRISE_URLS", "").strip())
        if not self.full_time_configured:
            self.logger.warning("No Apprise URLs configured for full-time jobs; skipping related task.")

    async def _call_llm(self, user_content: str, job_title: str, company: str) -> tuple[bool, str, str]:
        """Call the OpenAI-compatible LLM. Returns (post, reason, log_suffix). Fails open on any error."""
        post = True
        reason = "LLM response unavailable."
        log_suffix = ""
        try:
            openai_query = await gpt.responses.parse(
                model=os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07"),
                reasoning={"effort": "low"},
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                text_format=ApplicationAnalyzer,
                prompt_cache_key=CACHE_ID,
            )
            if openai_query.output_parsed is not None:
                post = openai_query.output_parsed.apply
                reason = openai_query.output_parsed.reason
            if openai_query.usage is not None:
                self.logger.info("OpenAI usage: %s", openai_query.usage)
                cached_details = openai_query.usage.input_tokens_details
                cached = (cached_details.cached_tokens if cached_details is not None else 0) or 0
                log_suffix = f". Used {openai_query.usage.input_tokens - cached}, cached {cached}"
        except RateLimitError as e:
            self.logger.warning("OpenAI rate limit; skipping check for %s @ %s", job_title, company, exc_info=True)
            reason = f"LLM check skipped (rate limit). Error: {e}"
        except (PermissionDeniedError, AuthenticationError) as e:
            self.logger.error("OpenAI auth/permission error; skipping check for %s @ %s", job_title, company, exc_info=True)
            reason = f"LLM check skipped (auth/permission error). Error: {e}"
        except (APITimeoutError, APIConnectionError, InternalServerError, APIError) as e:
            self.logger.warning("OpenAI API error; skipping check for %s @ %s", job_title, company, exc_info=True)
            reason = f"LLM check skipped (API error). Error: {e}"
        except OpenAIError as e:
            self.logger.warning("OpenAI error; skipping check for %s @ %s", job_title, company, exc_info=True)
            reason = f"LLM check skipped (OpenAI error). Error: {e}"
        except Exception as e:
            self.logger.exception("Unexpected error calling LLM for %s @ %s", job_title, company)
            reason = f"LLM check skipped (unexpected error). Error: {e}"
        return post, reason, log_suffix

    def _persist_job(
        self,
        session: SessionType,
        job_model: type,
        row: dict,
        application_url: str,
        company_url: str,
    ) -> None:
        try:
            session.add(job_model(
                job_id=row['id'],
                application_url=application_url,
                job_title=row['title'],
                company_name=row['company'],
                company_url=company_url,
            ))
            session.commit()
        except Exception:
            self.logger.exception("DB commit failed for job_id=%s; rolling back", row['id'])
            session.rollback()

    async def _notify_then_persist(
        self,
        session: SessionType,
        job_model: type,
        row: dict,
        message: str,
        tag: str,
        application_url: str,
        company_url: str,
    ) -> None:
        """Notify the user first; only persist to DB if notification succeeded."""
        try:
            notified = await asyncio.to_thread(self.apobj.notify, body=message, tag=tag)
        except Exception:
            self.logger.exception(
                "Apprise notification failed for %s @ %s; job will be retried next run",
                row['title'], row['company'],
            )
            return
        if not notified:
            self.logger.warning(
                "Apprise notify returned False for %s @ %s; job will be retried next run",
                row['title'], row['company'],
            )
            return
        self._persist_job(session, job_model, row, application_url, company_url)

    async def post_jobs(self, jobs: pd.DataFrame) -> None:
        job_model = FullTimeJob
        channel_name = "Full-Time Jobs"
        needed_cols = ['company', 'title', 'job_url', 'company_url', 'id', 'description', 'location']
        available_cols = [c for c in needed_cols if c in jobs.columns]
        required_cols = {'company', 'title', 'job_url', 'company_url', 'id'}
        if not required_cols.issubset(set(available_cols)):
            missing = required_cols - set(available_cols)
            self.logger.warning("Scrape result missing required columns %s for full-time jobs; skipping", missing)
            return
        job_rows = jobs[available_cols].to_dict('records')
        del jobs

        session = SessionLocal()
        try:
            for row in job_rows:
                if row['company'] in BLACKLISTED_COMPANIES:
                    self.logger.info("Skipping blacklisted company: %s in channel: %s", row['company'], channel_name)
                    continue

                if not any(term.lower() in row['title'].lower() for term in REQUIRED_TITLE_KEYWORDS):
                    self.logger.info("Skipping title '%s' (no required terms) in channel: %s", row['title'], channel_name)
                    continue

                if any(term.lower() in row['title'].lower() for term in BLOCKED_TITLE_KEYWORDS):
                    self.logger.info("Skipping bad role title: %s in channel: %s", row['title'], channel_name)
                    continue

                application_url = row['job_url']
                company_url = row['company_url']
                if application_url and isinstance(application_url, str) and 'linkedin.com' in application_url:
                    application_url = re.sub(r'//[^/]*\.linkedin\.com', '//linkedin.com', application_url)
                if company_url and isinstance(company_url, str) and 'linkedin.com' in company_url:
                    company_url = re.sub(r'//[^/]*\.linkedin\.com', '//linkedin.com', company_url)

                if session.query(job_model).filter(job_model.job_id == row['id']).first() is not None:
                    continue

                user_content = (
                    f"### JOB DESCRIPTION for {row['title']} at {row['company']}:\n"
                    f"\"\"\"\n{row['description']}\n\"\"\"\n\n"
                )
                post, reason, log_suffix = await self._call_llm(user_content, row['title'], row['company'])

                if not post:
                    self.logger.info("%s @ %s flagged by LLM as not compatible: %s", row['title'], row['company'], application_url)
                    self._persist_job(session, job_model, row, application_url, company_url)
                    continue

                loc = row.get('location')
                location_str = "" if (loc is None or (isinstance(loc, float) and pd.isna(loc))) else str(loc)
                should_ping = self._should_ping_for_location(location_str)
                reason_display = "LLM check skipped." if reason.startswith("LLM check skipped") else reason
                job_info = (
                    f">>> ## {''.join(random.choices(['🎉', '👏', '💼', '🔥', '💻'], k=1))} "
                    f"[{row['company']}](<{company_url}>) just posted a new job!\n\n"
                    f"### **Role:**\n[**{row['title']}**](<{application_url}>)\n\n"
                    f"### **Location:**\n{location_str}\n\n"
                    f"### **Reason:**\n{reason_display}"
                )
                message = f"{PING_MENTION}\n{job_info}" if should_ping else job_info
                self.logger.info("Posting job: %s to channel: %s (tag: ft)%s", row['title'], channel_name, log_suffix)
                await self._notify_then_persist(session, job_model, row, message, "ft", application_url, company_url)
        finally:
            session.close()

    def _should_ping_for_location(self, location: str) -> bool:
        if not PING_MENTION:
            return False

        state_code_match = False
        if PING_LOCATION_STATE_CODES:
            escaped_codes = "|".join(re.escape(code) for code in PING_LOCATION_STATE_CODES)
            state_code_match = bool(re.search(rf"(?:,\s*|\b)(?:{escaped_codes})\b", location))

        state_name_match = False
        if PING_LOCATION_NAMES:
            name_patterns = []
            for name in PING_LOCATION_NAMES:
                escaped_name = re.escape(name)
                if name.lower() == "washington":
                    escaped_name = rf"{escaped_name}\b(?!\s*(?:,|\s)\s*D\.?C\.?\b)"
                name_patterns.append(escaped_name)
            state_name_match = bool(re.search(rf"\b(?:{'|'.join(name_patterns)})\b", location, flags=re.IGNORECASE))

        return state_code_match or state_name_match

    async def full_time_job_task(self) -> None:
        if not self.full_time_configured:
            return
        full_time_jobs = await self.get_jobs(search_term=SEARCH_QUERY, results_wanted=50, sites=SEARCH_SITES)
        await self.post_jobs(full_time_jobs)

    async def get_jobs(
        self,
        sites: list[str] | dict[str, str] | None = None,
        search_term: str = SEARCH_QUERY,
        location: str = 'United States',
        results_wanted: int = 15,
        hours_old: int = 1,
    ) -> pd.DataFrame:
        if sites is None:
            sites = SEARCH_SITES

        if isinstance(sites, list):
            try:
                return await asyncio.to_thread(
                    scrape_jobs,
                    site_name=sites,
                    search_term=search_term,
                    location=location,
                    results_wanted=results_wanted,
                    hours_old=hours_old,
                    country_indeed="USA",
                    linkedin_fetch_description=True,
                )
            except Exception:
                self.logger.exception("Job scraping failed for sites=%s", sites)
                return pd.DataFrame()

        all_results = []
        for site, term in sites.items():
            try:
                jobs = await asyncio.to_thread(
                    scrape_jobs,
                    site_name=site,
                    search_term=term,
                    location=location if site != "glassdoor" else None,
                    results_wanted=results_wanted,
                    hours_old=hours_old,
                    country_indeed="USA",
                    linkedin_fetch_description=True,
                )
            except Exception:
                self.logger.exception("Job scraping failed for site=%s", site)
                continue
            if not jobs.empty:
                all_results.append(jobs)

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)

    async def run(self) -> None:
        self.logger.info("Python version: %s", platform.python_version())
        self.logger.info("Running on: %s %s (%s)", platform.system(), platform.release(), os.name)
        self.logger.info("-------------------")
        while True:
            try:
                await self.full_time_job_task()
                self.logger.info("Job posting task completed.")
            except Exception:
                self.logger.exception("job_posting_task failed; will retry next loop tick")
            await asyncio.sleep(60)

app = JobScraperApp(apobj=apobj)
asyncio.run(app.run())
