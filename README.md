# LinkedIn / Discord Job Scraper Bot

This project continuously scrapes recent software-related job listings, filters them against the resume in `Resume.pdf` with an OpenAI-compatible model, and sends matching jobs to one or more notification targets through [Apprise](https://github.com/caronc/apprise).

Despite the repository name, the current implementation does not log in as a Discord bot. Discord delivery is handled through Apprise, typically with a Discord webhook URL.

## Requirements

- Python 3.11
- A text-based `Resume.pdf` in the project root
- An OpenAI-compatible API key in your environment
  - [NVIDIA NIM](https://build.nvidia.com/explore/discover) is a free-tier option for an OpenAI-compatible provider.
- At least one Apprise notification target if you want full-time job alerts

## What It Does

- Scrapes recent jobs from LinkedIn, Indeed, and Glassdoor with `python-jobspy`
- Searches for software, IT, data, ML, cloud, firmware, automation, and related roles
- Rejects obvious mismatches before the LLM step:
  - blacklisted companies
  - senior/lead/manager-level titles
  - titles that do not contain required keywords
- Reads `Resume.pdf` and asks an OpenAI-compatible model whether a job is worth applying to
- Sends accepted jobs through Apprise using the `ft` notification tag
- Stores seen jobs in `jobs.db` so they are not posted repeatedly
- Repeats the scrape loop every 60 seconds

## Installation

```bash
git clone https://github.com/haydenthai/Linkedin-Discord-Job-Scraper-Bot.git
cd Linkedin-Discord-Job-Scraper-Bot
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Copy the example config and edit it:

```bash
cp .env.example .env
```

The example file currently includes:

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5-nano-2025-08-07
OPENAI_BASE_URL=
FT_APPRISE_URLS=https://discord.com/api/webhooks/...
```

At minimum, set `OPENAI_API_KEY`. Set `FT_APPRISE_URLS` if you want notifications enabled.

If you want a low-cost OpenAI-compatible provider, [NVIDIA NIM](https://build.nvidia.com/explore/discover) is a reasonable option to try because it offers a free tier. In that case, keep `OPENAI_API_KEY` set to your NVIDIA key and point `OPENAI_BASE_URL` at the NIM-compatible endpoint you want to use.

Supported environment variables used by the current code:

- `OPENAI_API_KEY`: credential used by the OpenAI Python SDK
- `OPENAI_MODEL`: optional model override; defaults to `gpt-5-nano-2025-08-07`
- `OPENAI_BASE_URL`: optional custom OpenAI-compatible API base URL
- `OPENAI_API_BASE_URL`: alternate base URL variable name
- `API_URL`: another fallback base URL variable name
- `FT_APPRISE_URLS`: comma-separated Apprise URLs for full-time job notifications

## Current Architecture

The project is currently a single-process Python app centered around [`bot.py`](bot.py).

Core components:

- Job scraping: `jobspy.scrape_jobs(...)`
- Resume parsing: `pypdf.PdfReader`
- LLM filtering: `openai.AsyncOpenAI().responses.parse(...)`
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

The application reads `Resume.pdf` at startup and exits immediately if:

- the file is missing
- the PDF is unreadable
- the PDF contains no extractable text

Use a normal text-based PDF, not an image-only scan.

## How Matching Works

The LLM prompt is intentionally strict. It tries to reject jobs when:

- required years of experience are above the candidate's qualifying corporate experience
- the role is too senior
- required hard skills are missing
- clearance or citizenship requirements are missing

If the LLM call fails because of rate limits, auth issues, or API errors, the app fails open and treats the job as postable while logging the reason.

## Running

Start the bot:

```bash
python3 bot.py
```

Run it in the background:

```bash
nohup python3 bot.py &
```

## Data Files

- [`bot.py`](bot.py): main application
- [`requirements.txt`](requirements.txt): Python dependencies
- `Resume.pdf`: source resume used by the filtering prompt
- `jobs.db`: SQLite database of already-seen jobs
- `discord.log`: rotating runtime logs

## Search Behavior

The current search configuration in code is:

- Sites: `linkedin`, `indeed`, `glassdoor`
- Search query: `(computer science) OR software OR devops OR developer`
- Location: `United States`
- Freshness window: last 1 hour
- Batch size: 50 jobs per loop for the full-time task

The app also highlights Pacific Northwest jobs by tagging messages with `@everyone` when the location matches `WA`, `OR`, `Washington`, or `Oregon`.

## Known Gaps

- The repository has no real automated test suite yet
- `requirements.txt` is incomplete relative to `bot.py`
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
- Verify your OpenAI-compatible credentials are valid

OpenAI request failures in logs

- Confirm `OPENAI_API_KEY`
- If using a non-OpenAI provider, set `OPENAI_BASE_URL`, `OPENAI_API_BASE_URL`, or `API_URL`
- Reduce request frequency or change models if you are hitting rate limits

## Security Note

This project uses environment variables for credentials, but local `.env` files and webhook URLs should be treated as secrets. Do not commit real API keys or live webhook endpoints.
