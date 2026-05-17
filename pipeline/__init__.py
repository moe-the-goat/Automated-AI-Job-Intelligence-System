"""
Job-scraping pipeline — core modules.

  core_search    — job fetchers (JobSpy + 7 public APIs + ATS)
  core_filter    — deterministic filter gauntlet + JobTracker
  core_ai        — Gemini evaluation, pre-screen, DDG web search
  core_embedding — CV/job similarity pre-ranking
  core_ats       — direct ATS API integration + Jina fallback
  core_notify    — email + GitHub Issue rendering and dispatch
"""
