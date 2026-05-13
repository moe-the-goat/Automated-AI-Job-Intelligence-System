import requests
import pandas as pd
from jobspy import scrape_jobs

"""
CORE SEARCH MODULE
------------------
This module is responsible for hunting down jobs across the internet. 
It talks directly to JobSpy (for LinkedIn, Glassdoor, Indeed) and connects 
to completely free APIs (Remotive, Arbeitnow, Jobicy, RemoteOK) to ensure 
we have the largest pool of jobs possible before filtering.
"""

def fetch_remotive_jobs():
    """Fetches purely remote software development jobs from Remotive's public API."""
    try:
        url = "https://remotive.com/api/remote-jobs?category=software-dev"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        jobs = response.json().get("jobs", [])
        
        parsed_jobs = []
        for j in jobs:
            parsed_jobs.append({
                "title": j.get("title", ""),
                "company": j.get("company_name", ""),
                "location": j.get("candidate_required_location", "Remote"),
                "job_url": j.get("url", ""),
                "date_posted": j.get("publication_date", "")
            })
        return pd.DataFrame(parsed_jobs)
    except Exception as e:
        print(f"Failed to fetch Remotive jobs: {e}")
        return pd.DataFrame()

def fetch_arbeitnow_jobs():
    """Fetches jobs from Arbeitnow, strictly filtering for those marked as remote."""
    try:
        url = "https://www.arbeitnow.com/api/job-board-api"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        jobs = response.json().get("data", [])
        
        parsed_jobs = []
        for j in jobs:
            if j.get("remote"):
                parsed_jobs.append({
                    "title": j.get("title", ""),
                    "company": j.get("company_name", ""),
                    "location": "Remote",
                    "job_url": j.get("url", ""),
                    "date_posted": str(j.get("created_at", ""))
                })
        return pd.DataFrame(parsed_jobs)
    except Exception as e:
        print(f"Failed to fetch Arbeitnow jobs: {e}")
        return pd.DataFrame()

def fetch_jobicy_jobs():
    """Fetches programming-specific remote jobs from the Jobicy API."""
    try:
        url = "https://jobicy.com/api/v2/remote-jobs?count=50&tag=python,software,backend,ai,data,ml"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        jobs = response.json().get("jobs", [])
        
        parsed_jobs = []
        for j in jobs:
            parsed_jobs.append({
                "title": j.get("jobTitle", ""),
                "company": j.get("companyName", ""),
                "location": j.get("jobGeo", "Remote"),
                "job_url": j.get("url", ""),
                "description": j.get("jobDescription", ""),
                "date_posted": j.get("pubDate", "")
            })
        return pd.DataFrame(parsed_jobs)
    except Exception as e:
        print(f"Failed to fetch Jobicy jobs: {e}")
        return pd.DataFrame()

def fetch_remoteok_jobs():
    """Fetches tech jobs from RemoteOK. Bypasses their legal notice header."""
    try:
        url = "https://remoteok.com/api"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        jobs = response.json()
        
        parsed_jobs = []
        for j in jobs:
            if "legal" in j:
                continue
            parsed_jobs.append({
                "title": j.get("position", ""),
                "company": j.get("company", ""),
                "location": j.get("location", "Remote"),
                "job_url": j.get("url", ""),
                "description": j.get("description", ""),
                "date_posted": j.get("date", "")
            })
        return pd.DataFrame(parsed_jobs)
    except Exception as e:
        print(f"Failed to fetch RemoteOK jobs: {e}")
        return pd.DataFrame()
