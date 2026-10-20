import asyncio
import hashlib
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pandas as pd
from openai import AsyncOpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import bot


def response_body(data=None, *, status="completed", refusal=False):
    content = ({"type": "refusal", "refusal": "Cannot analyze this input."} if refusal else
               {"type": "output_text", "text": json.dumps(data), "annotations": []})
    return {
        "id": "resp_test", "object": "response", "created_at": 1,
        "model": "gpt-5-nano-2025-08-07", "status": status,
        "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                    "status": "completed", "content": [content]}],
        "usage": {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40,
                  "input_tokens_details": {"cached_tokens": 20},
                  "output_tokens_details": {"reasoning_tokens": 0}},
    }


class ResumeTests(unittest.TestCase):
    def test_all_pages_including_empty_first_page(self):
        pages = [Mock(), Mock(), Mock()]
        for page, text in zip(pages, [None, "Experience", "Python and SQL"]):
            page.extract_text.return_value = text
        with patch.object(bot, "PdfReader", return_value=SimpleNamespace(pages=pages)):
            self.assertEqual(bot.load_resume(Path("fake.pdf")), "Experience\n\nPython and SQL")

    def test_empty_resume_is_rejected(self):
        with patch.object(bot, "PdfReader", return_value=SimpleNamespace(pages=[])):
            with self.assertRaises(ValueError):
                bot.load_resume(Path("fake.pdf"))

    def test_keyword_cleanup_bounds_and_boolean_syntax(self):
        terms = [" Data   Analyst ", "data analyst", "", "*", "(engineer)",
                 "developer OR manager", "x" * 81, "C++", "C#", "Python"]
        self.assertEqual(bot.clean_keywords(terms, 3), ["Data Analyst", "C++", "C#"])


class LoggingTests(unittest.TestCase):
    def test_jobspy_errors_reach_rotating_log(self):
        scraper_logger = logging.getLogger("JobSpy:Glassdoor")
        watched = [bot.logger, scraper_logger]
        # configure_logging attaches the file handler to all imported boards.
        watched.extend(value for name, value in logging.Logger.manager.loggerDict.copy().items()
                       if name.startswith("JobSpy:") and isinstance(value, logging.Logger)
                       and value is not scraper_logger)
        original = {value: list(value.handlers) for value in watched}
        try:
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(bot, "BASE_DIR", Path(directory)):
                    bot.configure_logging()
                scraper_logger.error("test Glassdoor service failure")
                self.assertIn("test Glassdoor service failure",
                              (Path(directory) / "discord.log").read_text())
        finally:
            added = {handler for value in watched for handler in value.handlers
                     if handler not in original[value]}
            for value, handlers in original.items():
                value.handlers = handlers
            for handler in added:
                handler.close()


class ScraperTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.environment = patch.dict(os.environ, {
            "FT_APPRISE_URLS": "test", "OPENAI_MODEL": "gpt-5-nano-2025-08-07",
            "OPENAI_REASONING_EFFORT": "low",
            "OPENAI_PROMPT_CACHE_MODE": "auto",
            "OPENAI_MAX_CONCURRENT_EVALUATIONS": "5",
            "OPENAI_SEARCH_MODEL": "gpt-6.1-sol",
            "OPENAI_SEARCH_REASONING_EFFORT": "medium",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.requests = []
        self.reply = response_body({
            "keywords": [" technical writer ", "Technical Writer", "documentation specialist"],
            "title_keywords": ["writer", "documentation"],
            "negative_keywords": ["registered nurse", "chief executive officer"],
        })
        self.http_status = 200

        def handle(request):
            self.requests.append(request)
            return httpx.Response(self.http_status, json=self.reply)

        self.client = AsyncOpenAI(
            api_key="test-only-not-a-real-key", base_url="https://mock.example/v1",
            max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        )
        # Keep SDK platform detection out of the executor in these offline tests.
        self.client._platform = "Linux"
        self.addAsyncCleanup(self.client.close)
        self.engine = create_engine("sqlite:///:memory:")
        bot.Base.metadata.create_all(self.engine)
        self.addCleanup(self.engine.dispose)
        self.app = bot.JobScraperApp(
            Mock(), self.client, "Technical writing experience.\nDocumentation projects.",
            sessionmaker(bind=self.engine),
        )
        self.app.logger = Mock()

    def duplicate_listing(self, job_id, **changes):
        return {
            "id": job_id, "company": "Example Inc.", "title": "Software Engineer",
            "location": "Portland, OR", "description": "Build software.",
            "job_url": f"https://example.com/{job_id}", "company_url": "https://example.com",
            **changes,
        }

    async def test_concurrent_evaluations_are_bounded_and_processed_as_completed(self):
        self.app.max_concurrent_evaluations = 2
        started = [asyncio.Event() for _ in range(4)]
        release = [asyncio.Event() for _ in range(4)]
        persisted = asyncio.Event()
        active = peak = 0

        async def evaluate(content, title, company):
            nonlocal active, peak
            index = int(title)
            active += 1
            peak = max(peak, active)
            started[index].set()
            try:
                await release[index].wait()
                if index == 3:
                    raise RuntimeError("One failed request")
                return False, "No match", ""
            finally:
                active -= 1

        persist = self.app._persist_job

        def record(*args):
            persist(*args)
            persisted.set()

        self.app._call_llm = AsyncMock(side_effect=evaluate)
        self.app._persist_job = Mock(side_effect=record)
        rows = [self.duplicate_listing(str(i), title=str(i)) for i in range(4)]
        # Deduplication must happen before starting concurrent requests.
        task = asyncio.create_task(self.app.post_jobs(pd.DataFrame(rows + [rows[0]])))
        try:
            await asyncio.wait_for(started[0].wait(), 2)
            await asyncio.wait_for(started[1].wait(), 2)
            self.assertFalse(started[2].is_set())
            release[1].set()
            await asyncio.wait_for(started[2].wait(), 2)
            await asyncio.wait_for(persisted.wait(), 2)
            self.assertFalse(task.done())
            self.assertFalse(release[0].is_set())
            for event in release:
                event.set()
            await asyncio.wait_for(task, 2)
            self.assertEqual(peak, 2)
            self.assertEqual(self.app._call_llm.await_count, 4)
            with self.app.session_factory() as session:
                self.assertEqual(session.query(bot.FullTimeJob).count(), 3)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_cancelling_batch_cleans_up_evaluations(self):
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def evaluate(*args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        self.app._call_llm = AsyncMock(side_effect=evaluate)
        task = asyncio.create_task(self.app.post_jobs(pd.DataFrame([self.duplicate_listing("cancel")])))
        try:
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            self.assertTrue(stopped.is_set())
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_cross_site_duplicates_use_fullest_description_before_llm(self):
        self.app._call_llm = AsyncMock(return_value=(False, "Not a match", ""))
        await self.app.post_jobs(pd.DataFrame([
            self.duplicate_listing("li-1"),
            self.duplicate_listing("zr-2", company=" EXAMPLE  INC ",
                                   title="software engineer", location="Portland OR",
                                   description="A much more complete job description."),
        ]))
        self.app._call_llm.assert_awaited_once()
        self.assertIn("A much more complete", self.app._call_llm.call_args.args[0])

    async def test_search_plan_survives_database_reopen_without_api_call(self):
        with tempfile.TemporaryDirectory() as directory:
            url = f"sqlite:///{Path(directory) / 'jobs.db'}"
            engine = create_engine(url)
            try:
                bot.Base.metadata.create_all(engine)
                self.app.session_factory = sessionmaker(bind=engine)
                await self.app.initialize_search()
                self.assertEqual(len(self.requests), 1)
                with self.app.session_factory() as session:
                    cached = session.get(bot.CachedResumeSearchPlan,
                                         hashlib.sha256(self.app.resume.encode()).hexdigest())
                    self.assertIsNotNone(cached)
                    self.assertEqual(json.loads(cached.plan_json)["keywords"],
                                     ["technical writer", "documentation specialist"])
            finally:
                engine.dispose()

            engine = create_engine(url)
            try:
                other = bot.JobScraperApp(Mock(), self.client, self.app.resume,
                                          sessionmaker(bind=engine))
                other.logger = Mock()
                other._parse_response = AsyncMock(side_effect=AssertionError("Cache should avoid API"))
                await other.initialize_search()
                other._parse_response.assert_not_awaited()
                self.assertEqual(other.search_query, self.app.search_query)
                self.assertEqual(other.required_title_keywords, self.app.required_title_keywords)
                self.assertEqual(other.negative_keywords, self.app.negative_keywords)
            finally:
                engine.dispose()

    async def test_changed_resume_generates_separate_plan(self):
        await self.app.initialize_search()
        other = bot.JobScraperApp(Mock(), self.client, self.app.resume + "\nNew skills",
                                  self.app.session_factory)
        other.logger = Mock()
        await other.initialize_search()
        self.assertEqual(len(self.requests), 2)
        with self.app.session_factory() as session:
            self.assertEqual(session.query(bot.CachedResumeSearchPlan).count(), 2)

    async def test_invalid_cached_search_plan_is_regenerated(self):
        resume_hash = hashlib.sha256(self.app.resume.encode()).hexdigest()
        for value in ("invalid json", json.dumps({"keywords": ["***"],
                      "title_keywords": ["writer"], "negative_keywords": []})):
            with self.subTest(value=value):
                with self.app.session_factory() as session:
                    session.merge(bot.CachedResumeSearchPlan(resume_hash=resume_hash, plan_json=value))
                    session.commit()
                self.app._search_initialized = False
                count = len(self.requests)
                await self.app.initialize_search()
                self.assertEqual(len(self.requests), count + 1)
                with self.app.session_factory() as session:
                    cached = session.get(bot.CachedResumeSearchPlan, resume_hash)
                    self.assertEqual(json.loads(cached.plan_json)["keywords"][0], "technical writer")

    async def test_cache_database_failure_keeps_generated_plan(self):
        self.app.session_factory = Mock(side_effect=RuntimeError("Database unavailable"))
        await self.app.initialize_search()
        self.assertEqual(self.app.search_query, "(technical writer) OR (documentation specialist)")
        self.assertTrue(self.app.negative_keywords)
        self.assertEqual(len(self.requests), 1)

    async def test_cross_site_dedup_survives_new_app_instance(self):
        self.app._call_llm = AsyncMock(return_value=(False, "Not a match", ""))
        await self.app.post_jobs(pd.DataFrame([self.duplicate_listing("li-1")]))
        other = bot.JobScraperApp(Mock(), self.client, self.app.resume, self.app.session_factory)
        other._call_llm = AsyncMock(return_value=(False, "Not a match", ""))
        await other.post_jobs(pd.DataFrame([self.duplicate_listing("zr-2")]))
        other._call_llm.assert_not_awaited()

    async def test_distinct_and_missing_locations_are_not_merged(self):
        self.app._call_llm = AsyncMock(return_value=(False, "Not a match", ""))
        rows = [self.duplicate_listing(str(index), location=location)
                for index, location in enumerate(["Portland, OR", "Seattle, WA", None, float("nan")])]
        rows += [self.duplicate_listing("other-company", company="Another Company"),
                 self.duplicate_listing("other-title", title="Data Engineer")]
        await self.app.post_jobs(pd.DataFrame(rows))
        self.assertEqual(self.app._call_llm.await_count, len(rows))

    async def test_failed_notification_is_retried_next_run(self):
        self.app._call_llm = AsyncMock(return_value=(True, "Match", ""))
        self.app.apobj.notify.return_value = False
        jobs = pd.DataFrame([self.duplicate_listing("li-1"), self.duplicate_listing("zr-2")])
        await self.app.post_jobs(jobs)
        self.app._call_llm.assert_awaited_once()
        self.app.apobj.notify.return_value = True
        await self.app.post_jobs(jobs)
        self.assertEqual(self.app._call_llm.await_count, 2)
        await self.app.post_jobs(pd.DataFrame([self.duplicate_listing("gd-3")]))
        self.assertEqual(self.app._call_llm.await_count, 2)

    async def test_existing_job_id_without_location_still_skips_llm(self):
        with self.app.session_factory() as session:
            session.add(bot.FullTimeJob(job_id="old-job", company_name="Example Inc.",
                                        job_title="Software Engineer"))
            session.commit()
        self.app._call_llm = AsyncMock(return_value=(False, "Not a match", ""))
        await self.app.post_jobs(pd.DataFrame([self.duplicate_listing("old-job")]))
        self.app._call_llm.assert_not_awaited()

    async def test_glassdoor_failure_preserves_other_sites(self):
        def scrape(**kwargs):
            site = kwargs["site_name"]
            if site == "glassdoor":
                raise ValueError("Glassdoor location request failed")
            return pd.DataFrame([{"site": site, "title": "Software Engineer"}])

        with patch.object(bot, "scrape_jobs", side_effect=scrape):
            jobs = await self.app.get_jobs(sites=["linkedin", "glassdoor", "indeed"])
        self.assertEqual(jobs["site"].tolist(), ["linkedin", "indeed"])
        self.app.logger.exception.assert_called_once_with("Job scraping failed for site=%s", "glassdoor")

    async def test_search_options_and_descriptions_for_lists_and_mappings(self):
        for sites in (["glassdoor"], {"glassdoor": "data analyst"}):
            with self.subTest(sites=sites):
                with patch.object(bot, "scrape_jobs", return_value=pd.DataFrame()) as scrape:
                    jobs = await self.app.get_jobs(sites=sites, search_term="software engineer",
                                                   location="Portland, OR", results_wanted=3, hours_old=6)
                self.assertTrue(jobs.empty)
                scrape.assert_called_once_with(
                    site_name="glassdoor",
                    search_term="data analyst" if isinstance(sites, dict) else "software engineer",
                    location="Portland, OR", results_wanted=3, hours_old=6,
                    country_indeed="USA", fetch_description=True,
                )

    async def test_all_sites_failed_returns_empty_frame(self):
        with patch.object(bot, "scrape_jobs", side_effect=RuntimeError("service unavailable")):
            jobs = await self.app.get_jobs()
        self.assertTrue(jobs.empty)

    async def test_plan_reaches_scraper_once_and_uses_responses_schema(self):
        self.app.post_jobs = AsyncMock()
        async def inline_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (patch.object(bot, "scrape_jobs", return_value=pd.DataFrame()) as scrape,
              patch.object(bot.asyncio, "to_thread", side_effect=inline_thread)):
            await self.app.full_time_job_task()
            await self.app.full_time_job_task()
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0].url.path, "/v1/responses")
        request = json.loads(self.requests[0].content)
        self.assertEqual(request["input"][1]["content"], self.app.resume)
        self.assertEqual(request["model"], "gpt-6.1-sol")
        self.assertEqual(request["service_tier"], "default")
        self.assertEqual(request["reasoning"], {"effort": "medium"})
        self.assertTrue(request["store"])
        self.assertNotIn("response_format", request)
        fmt = request["text"]["format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertTrue(fmt["strict"])
        self.assertFalse(fmt["schema"]["additionalProperties"])
        self.assertEqual(set(fmt["schema"]["required"]), {"keywords", "title_keywords", "negative_keywords"})
        for call in scrape.call_args_list:
            self.assertEqual(call.kwargs["search_term"], "(technical writer) OR (documentation specialist)")
        self.assertIn("writer", self.app.required_title_keywords)

    async def test_generated_title_can_pass_filter(self):
        await self.app.initialize_search()
        title = "Documentation Specialist"
        self.assertFalse(any(term.lower() in title.lower() for term in bot.REQUIRED_TITLE_KEYWORDS))
        self.app._call_llm = AsyncMock(return_value=(True, "Matches", ""))
        self.app._notify_then_persist = AsyncMock()
        await self.app.post_jobs(pd.DataFrame([{
            "id": "writer-job", "company": "Example", "title": title,
            "job_url": "https://example.com/job", "company_url": "https://example.com",
            "description": "Write documentation", "location": "Remote",
        }]))
        self.app._call_llm.assert_awaited_once()
        self.app._notify_then_persist.assert_awaited_once()

    async def test_failed_or_unusable_plans_keep_defaults_without_repeated_calls(self):
        cases = [
            (200, response_body(refusal=True)),
            (200, response_body({"keywords": [], "title_keywords": ["writer"], "negative_keywords": []})),
            (200, response_body({"keywords": ["***"], "title_keywords": ["writer"], "negative_keywords": []})),
            (200, response_body({"keywords": ["writer"], "title_keywords": [], "negative_keywords": []})),
            (200, response_body({"keywords": "not an array", "title_keywords": [], "negative_keywords": []})),
            (200, response_body({"keywords": ["writer"], "title_keywords": ["writer"], "negative_keywords": []}, status="incomplete")),
            (429, {"error": {"message": "Too many requests", "type": "rate_limit_error"}}),
            (401, {"error": {"message": "Unauthorized", "type": "authentication_error"}}),
        ]
        for code, body in cases:
            with self.subTest(code=code, body=body):
                self.http_status, self.reply = code, body
                self.app._search_initialized = False
                count = len(self.requests)
                await self.app.initialize_search()
                await self.app.initialize_search()
                self.assertEqual(len(self.requests), count + 1)
                self.assertEqual(self.app.search_query, bot.SEARCH_QUERY)
                self.assertEqual(self.app.required_title_keywords, bot.REQUIRED_TITLE_KEYWORDS)
                self.assertEqual(self.app.negative_keywords, [])
                with self.app.session_factory() as session:
                    self.assertEqual(session.query(bot.CachedResumeSearchPlan).count(), 0)

    async def test_evaluation_parses_decision_and_usage(self):
        self.reply = response_body({"fit": "poor", "reason": "Missing required skill"})
        post, reason, suffix = await self.app._call_llm("Job description", "Writer", "Example")
        self.assertFalse(post)
        self.assertEqual(reason, "Missing required skill")
        self.assertEqual(suffix, ". Input 30, cached-read 20, cache-write unavailable, ordinary-input unavailable, output 10")
        request = json.loads(self.requests[0].content)
        self.assertIn(self.app.resume, request["input"][0]["content"])
        self.assertEqual(request["text"]["format"]["name"], "ApplicationAnalyzer")
        self.assertEqual(request["service_tier"], "flex")

    async def test_evaluation_refusal_defers_notification(self):
        self.reply = response_body(refusal=True)
        post, reason, _ = await self.app._call_llm("Job", "Writer", "Example")
        self.assertIsNone(post)
        self.assertTrue(reason.startswith("LLM check skipped"))

    async def test_only_good_or_excellent_fits_are_sent(self):
        for index, fit in enumerate(["poor", "fair", "good", "excellent"]):
            with self.subTest(fit=fit):
                self.reply = response_body({"fit": fit, "reason": "Resume evidence"})
                self.app._notify_then_persist = AsyncMock()
                await self.app.post_jobs(pd.DataFrame([self.duplicate_listing(
                    str(index), title=f"Unlisted Role {index}")]))
                self.assertEqual(self.app._notify_then_persist.await_count,
                                 int(fit in {"good", "excellent"}))

    async def test_title_keywords_do_not_veto_llm_review(self):
        self.app._call_llm = AsyncMock(return_value=(False, "Fair fit", ""))
        for index, title in enumerate(["Technical Writer", "Engineer II", "Lead Developer",
                                       "Office Assistant", "Senior Analyst"]):
            await self.app.post_jobs(pd.DataFrame([self.duplicate_listing(str(index), title=title)]))
        self.assertEqual(self.app._call_llm.await_count, 5)

    async def test_unavailable_evaluation_is_not_sent_or_persisted(self):
        cases = [response_body(refusal=True),
                 response_body({"fit": "good", "reason": "match"}, status="incomplete"),
                 response_body({"fit": "unknown", "reason": "match"})]
        self.app._notify_then_persist = AsyncMock()
        jobs = pd.DataFrame([self.duplicate_listing("retry")])
        for reply in cases:
            self.reply = reply
            await self.app.post_jobs(jobs)
        self.http_status = 429
        self.reply = {"error": {"message": "Rate limited", "type": "rate_limit_error"}}
        await self.app.post_jobs(jobs)
        self.app._notify_then_persist.assert_not_awaited()
        with self.app.session_factory() as session:
            self.assertEqual(session.query(bot.FullTimeJob).count(), 0)
        self.http_status = 200
        self.reply = response_body({"fit": "good", "reason": "Matches"})
        await self.app.post_jobs(jobs)
        self.app._notify_then_persist.assert_awaited_once()

    def test_ping_locations_and_remote_metadata(self):
        for location, remote, expected in [
            ("Portland, OR", None, True), ("Seattle, WA", False, True),
            ("Oregon", None, True), ("Washington", None, True),
            ("Washington, DC", None, False), ("Boston, MA", None, False),
            ("Remote", None, True), ("US (REMOTE)", None, True),
            ("Boston, MA", True, True), ("Remote", False, False),
            ("Non-remote", None, False), ("Not remote", None, False),
            ("", float("nan"), False), ("", None, False),
        ]:
            with self.subTest(location=location, remote=remote):
                self.assertEqual(self.app._should_ping_for_location(location, remote), expected)

    async def test_notifications_ping_only_states_or_remote(self):
        self.app._call_llm = AsyncMock(return_value=(True, "Good fit", ""))
        self.app._notify_then_persist = AsyncMock()
        for index, (location, remote, ping) in enumerate([
            ("Boston, MA", False, False), ("Boston, MA", True, True),
            ("Portland, OR", False, True), ("Remote", None, True),
        ]):
            await self.app.post_jobs(pd.DataFrame([self.duplicate_listing(
                str(index), location=location, is_remote=remote)]))
            message = self.app._notify_then_persist.call_args.args[3]
            self.assertEqual(message.startswith(bot.PING_MENTION), ping)

    async def test_reasoning_can_be_omitted_for_other_models(self):
        with patch.dict(os.environ, {"OPENAI_SEARCH_REASONING_EFFORT": ""}):
            await self.app.initialize_search()
        self.assertNotIn("reasoning", json.loads(self.requests[0].content))

    async def test_negative_keywords_skip_titles_before_individual_evaluation(self):
        await self.app.initialize_search()
        self.app._call_llm = AsyncMock(return_value=(False, "No match", ""))
        await self.app.post_jobs(pd.DataFrame([
            self.duplicate_listing("excluded", title="REGISTERED-NURSE (RN)"),
            self.duplicate_listing("allowed", title="Software Engineer",
                                   description="Build tools for registered nurses."),
            self.duplicate_listing("boundary", title="Registered Nursery Engineer"),
        ]))
        self.assertEqual(self.app._call_llm.await_count, 2)
        reviewed = {call.args[1] for call in self.app._call_llm.call_args_list}
        self.assertEqual(reviewed, {"Software Engineer", "Registered Nursery Engineer"})
        with self.app.session_factory() as session:
            self.assertEqual(session.query(bot.FullTimeJob).count(), 2)

    async def test_planner_overrides_and_empty_negative_keywords(self):
        self.reply = response_body({"keywords": ["writer"], "title_keywords": ["writer"],
                                    "negative_keywords": []})
        with patch.dict(os.environ, {"OPENAI_SEARCH_MODEL": "custom-planner",
                                    "OPENAI_SEARCH_REASONING_EFFORT": "high"}):
            await self.app.initialize_search()
        request = json.loads(self.requests[0].content)
        self.assertEqual(request["model"], "custom-planner")
        self.assertEqual(request["reasoning"], {"effort": "high"})
        self.assertEqual(self.app.negative_keywords, [])
        self.reply = response_body({"fit": "poor", "reason": "No match"})
        await self.app._call_llm("Job", "Writer", "Example")
        evaluation = json.loads(self.requests[1].content)
        self.assertEqual(evaluation["model"], "gpt-5-nano-2025-08-07")
        self.assertEqual(evaluation["reasoning"], {"effort": "low"})

    async def test_no_keyword_call_when_notifications_disabled(self):
        self.app.full_time_configured = False
        await self.app.full_time_job_task()
        self.assertEqual(self.requests, [])

    async def test_explicit_evaluation_caches_only_stable_prefix(self):
        self.reply = response_body({"fit": "good", "reason": "Matches"})
        self.client.base_url = "https://api.openai.com/v1"
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-6-luna"}):
            await self.app._call_llm("First distinct job", "Writer", "Example")
            await self.app._call_llm("Second distinct job", "Writer", "Example")
        first, second = [json.loads(r.content) for r in self.requests]
        self.assertEqual(first["prompt_cache_options"], {"mode": "explicit"})
        self.assertEqual(first["input"][0], second["input"][0])
        self.assertEqual(first["input"][0], {"role": "developer", "content": [{
            "type": "input_text", "text": self.app.system_prompt,
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }]})
        self.assertEqual(first["input"][1], {"role": "user", "content": "First distinct job"})
        self.assertEqual(second["input"][1]["content"], "Second distinct job")
        self.assertEqual(first["prompt_cache_key"], second["prompt_cache_key"])
        self.assertTrue(first["store"])
        self.assertEqual(first["text"]["format"]["name"], "ApplicationAnalyzer")

    async def test_explicit_keyword_request_has_no_cache_breakpoints(self):
        with patch.dict(os.environ, {"OPENAI_PROMPT_CACHE_MODE": "explicit"}):
            await self.app.initialize_search()
        request = json.loads(self.requests[0].content)
        self.assertEqual(request["prompt_cache_options"], {"mode": "explicit"})
        self.assertNotIn("prompt_cache_breakpoint", json.dumps(request))
        self.assertEqual(request["input"][1]["content"], self.app.resume)
        self.app.logger.info.assert_any_call(
            "OpenAI usage (%s)%s", "ResumeSearchPlan",
            ". Input 30, cached-read 20, cache-write unavailable, ordinary-input unavailable, output 10",
        )

    async def test_cache_mode_compatibility(self):
        cases = [
            ("https://api.openai.com/v1", "gpt-6-luna", "auto", True),
            ("https://api.openai.com/v1", "gpt-6.1-sol", "auto", True),
            ("https://api.openai.com/v1", "gpt-5-nano-2025-08-07", "auto", False),
            ("https://mock.example/v1", "gpt-6-luna", "auto", False),
            ("https://mock.example/v1", "gpt-6-luna", "explicit", True),
            ("https://api.openai.com/v1", "gpt-6-luna", "implicit", False),
        ]
        for url, model, mode, explicit in cases:
            with self.subTest(url=url, model=model, mode=mode):
                self.client.base_url = url
                with patch.dict(os.environ, {"OPENAI_MODEL": model, "OPENAI_PROMPT_CACHE_MODE": mode}):
                    await self.app._parse_response(bot.KEYWORD_PROMPT, self.app.resume, bot.ResumeSearchPlan)
                request = json.loads(self.requests[-1].content)
                self.assertEqual("prompt_cache_options" in request, explicit)

    def test_cache_key_stable_across_instances_and_sensitive_to_prefix_and_schema(self):
        other = bot.JobScraperApp(Mock(), self.client, self.app.resume, self.app.session_factory)
        key = self.app._cache_key(self.app.system_prompt, bot.ApplicationAnalyzer)
        self.assertEqual(key, other._cache_key(other.system_prompt, bot.ApplicationAnalyzer))
        self.assertNotEqual(key, self.app._cache_key(self.app.system_prompt + "changed", bot.ApplicationAnalyzer))
        self.assertNotEqual(key, self.app._cache_key(self.app.system_prompt, bot.ResumeSearchPlan))

    async def test_usage_reports_writes_and_ordinary_tokens(self):
        self.reply = response_body({"fit": "good", "reason": "Matches"})
        self.reply["usage"]["input_tokens_details"]["cache_write_tokens"] = 7
        _, _, suffix = await self.app._call_llm("Job", "Writer", "Example")
        self.assertEqual(suffix, ". Input 30, cached-read 20, cache-write 7, ordinary-input 3, output 10")

    async def test_missing_usage_details_and_usage_are_safe(self):
        self.reply = response_body({"fit": "good", "reason": "Matches"})
        self.reply["usage"]["input_tokens_details"] = None
        post, _, suffix = await self.app._call_llm("Job", "Writer", "Example")
        self.assertTrue(post)
        self.assertIn("cached-read unavailable, cache-write unavailable, ordinary-input unavailable", suffix)
        self.reply["usage"] = None
        post, reason, suffix = await self.app._call_llm("Job", "Writer", "Example")
        self.assertEqual((post, reason, suffix), (True, "Matches", ""))


if __name__ == "__main__":
    unittest.main()
