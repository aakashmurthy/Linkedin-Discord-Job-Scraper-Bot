# LinkedIn / Discord Job Scraper Bot

This project continuously scrapes recent job listings using AI-selected resume keywords, filters them against the resume in `Resume.pdf` with an OpenAI-compatible model, and sends matching jobs to one or more notification targets through [Apprise](https://github.com/caronc/apprise).

Despite the repository name, the current implementation does not log in as a Discord bot. Discord delivery is handled through Apprise, typically with a Discord webhook URL.

## Requirements

- Python 3.11
- A text-based `Resume.pdf` in the project root
- An OpenAI API key in your environment (custom providers must support the Responses API and structured outputs)
- At least one Apprise notification target if you want full-time job alerts

## What It Does

- Scrapes recent jobs from LinkedIn, Indeed, Glassdoor, and ZipRecruiter with `python-jobspy`
- Selects relevant search phrases and title keywords from the full resume once at startup
- Skips blacklisted companies, then lets the LLM judge fit without title keyword vetoes
- Sends only jobs rated good or excellent against the full resume
- Sends accepted jobs through Apprise using the `ft` notification tag
- Stores seen jobs in `jobs.db` so they are not posted repeatedly
- Repeats the scrape loop every 60 seconds

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/Linkedin-Discord-Job-Scraper-Bot.git
cd Linkedin-Discord-Job-Scraper-Bot
pyenv install -s 3.11.17
pyenv local 3.11.17
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install [pyenv](https://github.com/pyenv/pyenv#installation) and its Python build
dependencies first, and enable pyenv in your shell. The project pins Python
3.11.17 in `.python-version`; `.venv` isolates the bot's dependencies.

## Configuration

Copy the example config and edit it:

```bash
cp .env.example .env
```

The example file currently includes:

```dotenv
# Only needed here if not already set globally:
# OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5-nano-2025-08-07
OPENAI_REASONING_EFFORT=low
FT_APPRISE_URLS=https://discord.com/api/webhooks/...
```

At minimum, set `OPENAI_API_KEY` in your environment or `.env`. Existing environment variables take precedence over `.env`. Set `FT_APPRISE_URLS` if you want notifications enabled.

Custom providers must implement `/responses` with structured outputs. Chat Completions compatibility alone is insufficient.

Supported environment variables used by the current code:

- `OPENAI_API_KEY`: credential used by the OpenAI Python SDK
- `OPENAI_MODEL`: optional model override; defaults to `gpt-5-nano-2025-08-07`
- `OPENAI_REASONING_EFFORT`: defaults to `low`; set to an empty string for models that do not support reasoning options
- `OPENAI_PROMPT_CACHE_MODE`: `auto` (default), `explicit`, or `implicit`. Auto uses explicit caching for GPT-6 models on `api.openai.com`; other models and custom providers retain their default implicit behavior. Set `explicit` only when the configured model/provider supports explicit cache options and breakpoints.
- `OPENAI_BASE_URL`: optional custom Responses-compatible API base URL
- `OPENAI_API_BASE_URL`: alternate base URL variable name
- `API_URL`: another fallback base URL variable name
- `FT_APPRISE_URLS`: comma-separated Apprise URLs for full-time job notifications

## Current Architecture

The project is currently a single-process Python app centered around [`bot.py`](bot.py).

Core components:

- Job scraping: `jobspy.scrape_jobs(...)`
- Resume parsing: `pypdf.PdfReader`
- AI keyword selection and LLM filtering: `openai.AsyncOpenAI().responses.parse(...)`
- Notifications: `apprise.Apprise`
- Persistence: SQLite via SQLAlchemy in `jobs.db`
- Logging: console output plus rotating log files in `discord.log`

## Notification Setup

Apprise supports many backends. For Discord, the simplest setup is usually a webhook URL.

See the Apprise services catalog for the full list of supported notification backends:
[https://appriseit.com/services/](https://appriseit.com/services/)

Example:

```dotenv
FT_APPRISE_URLS=https://discord.com/api/webhooks/...
```

You can send to multiple destinations by separating URLs with commas:

```dotenv
FT_APPRISE_URLS=https://discord.com/api/webhooks/...,ntfy://topic-name?format=markdown
```

If `FT_APPRISE_URLS` is missing, the app still runs but skips the full-time notification task.

## Resume Requirements

The application reads every page of `Resume.pdf` at startup and exits immediately if:

- the file is missing
- the PDF is unreadable
- the PDF contains no extractable text

Use a normal text-based PDF, not an image-only scan.

## How Matching Works

Before the first scrape, the AI selects up to eight search phrases based on demonstrated skills, experience, projects, education, and seniority. It also selects role/title terms for logging; these do not reject listings. Queries and title terms are cleaned, deduplicated, and logged. The company blacklist still applies.

The plan is reused for the process lifetime; restart after changing the resume. If analysis fails, is refused, or returns unusable terms, the bot uses `SEARCH_KEYWORDS` and `REQUIRED_TITLE_KEYWORDS` from `bot.py` for that run. No keyword request is made when notifications are disabled.

Both AI tasks use `AsyncOpenAI.responses.parse` with Pydantic schemas, `text_format`, and `output_parsed`, following the [official Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs). Responses are requested with `store=False`.

With explicit caching, job evaluation places one breakpoint after the fixed audit instructions and full resume. Each job description follows that breakpoint, so changing job text is not written to the prompt cache. The one-off keyword request uses explicit mode without breakpoints, avoiding cache reads and writes for that task. Implicit mode omits these controls for compatibility with older models and custom providers.

Cache keys are deterministic hashes of the task's stable prompt and structured-output schema, so restarting the bot does not change them. A cold or evicted cache can still write the reusable prefix; warm requests can read it when the provider's caching requirements are met. No job or resume content is truncated. See [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).

Both AI tasks log input, cached-read, cache-write, ordinary-input, and output tokens. Ordinary input is total input minus cached reads and cache writes. Missing read/write metrics are reported as `unavailable`, and ordinary input is unavailable if either is missing. Compare these logs across different jobs during normal operation to assess savings; mocked tests verify request structure rather than actual provider cache hits.

The LLM rates each job as poor, fair, good, or excellent. Only good and excellent
fits are sent. It considers relevant work (including freelance/self-employed work),
projects, education, transferable skills, core responsibilities, and seniority.
Missing exact keywords, preferred skills, or small experience gaps do not cause
automatic rejection; material qualification gaps and insufficient job details do.

Failed, refused, incomplete, or invalid evaluations send no notification and are
not saved as rejections, allowing a later scrape to retry them.

## Running

Start the bot:

```bash
python3 bot.py
```

Run it persistently with PM2 from the project directory. If your API key is
exported in `.bashrc`, run these commands from a shell that has loaded it:

```bash
pm2 start ecosystem.config.cjs
pm2 save
```

The PM2 configuration uses `.venv/bin/python`, runs one instance, and restarts
the bot after failures. Runtime logs are in `discord.log` and rotate automatically.
The virtual environment uses the persistent Python installation managed by
pyenv at `~/.pyenv/versions/3.11.17`. PM2 uses the virtual environment's absolute
interpreter path, so it does not depend on shell initialization at boot.

To restore the saved process list after reboot, run `pm2 startup` and follow
its instructions, then run `pm2 save`.

Useful management commands:

```bash
pm2 status
pm2 restart linkedin-job-bot
pm2 stop linkedin-job-bot
tail -f discord.log
```

After changing an exported environment variable, restart from the shell that
has loaded the new value with `pm2 restart linkedin-job-bot --update-env`, then
run `pm2 save` again. Changes to `.env` are read on restart; existing exported
variables still take precedence.

## Data Files

- [`bot.py`](bot.py): main application
- [`requirements.txt`](requirements.txt): Python dependencies
- `Resume.pdf`: source resume used by the filtering prompt
- `jobs.db`: SQLite database of already-seen jobs
- `discord.log`: rotating runtime logs

## Search Behavior

The current search configuration in code is:

- Sites: `linkedin`, `indeed`, `glassdoor`, `zip_recruiter`
- Search query: resume-derived phrases joined with `OR`
- Fallback query: `(computer science) OR software OR devops OR developer`
- Location: `United States`
- Freshness window: last 1 hour
- Batch size: 50 jobs per loop for the full-time task

JobSpy is pinned to 1.3.0 for its updated Glassdoor endpoints and parsing.
Descriptions are fetched for both LinkedIn and Glassdoor using `fetch_description=True`.
Sites run concurrently with separate error handling, so a failed site does not discard
jobs from the others. Location filters also apply to Glassdoor when using per-site queries.
JobSpy warnings and errors are captured in `discord.log`, including under PM2.

Before LLM evaluation, listings are deduplicated by job ID and normalized company,
title, and location. The copy with the longest description is evaluated first.
Successfully notified jobs and rejected jobs are remembered across runs; failed
notifications can retry on the next run. Missing locations are not matched across
sites. Older database entries without a saved location still match by job ID,
but cannot match by company/title/location. This conservative match does not merge
location aliases or differently worded titles; identical company/title/location
listings are treated as one opening.

The app tags accepted jobs with `@everyone` only for configured states (`WA`/`Washington` and `OR`/`Oregon`) or remote jobs. Remote status uses JobSpy metadata when available, otherwise the location text. Other accepted jobs are posted without a mention.

## Tests

Run `python3 -m unittest discover -s tests -v` after installing dependencies. Tests use a mocked HTTP transport with the real OpenAI SDK; they do not call OpenAI, scrape live sites, or send notifications.

## Known Gaps

- The implementation is a single large script with hardcoded search and filtering rules
- Secrets should not be committed in `.env` or other tracked files

## Troubleshooting

`Resume.pdf not found in the project directory`

- Add `Resume.pdf` to the repository root

`Resume.pdf could not be read`

- Export a text-based PDF instead of a scanned image PDF

No jobs are being posted

- Check that `FT_APPRISE_URLS` is set
- Check `discord.log` for scrape failures or notification errors
- Verify your OpenAI credentials are valid

OpenAI request failures in logs

- Confirm `OPENAI_API_KEY`
- If using a custom provider, confirm it supports `/responses` and structured outputs before setting `OPENAI_BASE_URL`, `OPENAI_API_BASE_URL`, or `API_URL`
- If the model does not accept reasoning options, set `OPENAI_REASONING_EFFORT=`
- Upgrade dependencies with `pip install -r requirements.txt` if `responses.parse` is unavailable
- Reduce request frequency or change models if you are hitting rate limits

## Security Note

This project uses environment variables for credentials, but local `.env` files and webhook URLs should be treated as secrets. Do not commit real API keys or live webhook endpoints.
