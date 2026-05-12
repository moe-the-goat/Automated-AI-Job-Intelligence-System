import json
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from jobspy import scrape_jobs
import pandas as pd
import requests
from datetime import datetime
import google.generativeai as genai
from bs4 import BeautifulSoup
import time
from duckduckgo_search import DDGS
def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def format_email_html(internships_df, jobs_df):
    html = "<h2>Automated AI Job Alerts</h2>"
    
    # --- Internships Table ---
    html += "<h3>🎓 Internships (AI & SWE)</h3>"
    if internships_df.empty:
        html += "<p>No relevant internships found today.</p>"
    else:
        html += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        html += "<tr><th>Title</th><th>Company</th><th>Location</th><th>AI Verdict</th><th>Match %</th><th>Link</th></tr>"
        for _, row in internships_df.iterrows():
            title = row.get("title", "N/A")
            company = row.get("company", "N/A")
            location = row.get("location", "Remote/Unspecified")
            verdict = row.get("ai_verdict", "")
            match_pct = row.get("match_percentage", "N/A")
            if match_pct != "N/A": match_pct = f"{match_pct}%"
            job_url = row.get("job_url", "#")
            html += f"<tr><td>{title}</td><td>{company}</td><td>{location}</td><td>{verdict}</td><td>{match_pct}</td><td><a href='{job_url}'>Apply</a></td></tr>"
        html += "</table><br>"
        
    # --- Full-Time Jobs Table ---
    html += "<h3>💼 Junior & Entry-Level Jobs (CV-Matched)</h3>"
    if jobs_df.empty:
        html += "<p>No relevant full-time jobs found today.</p>"
    else:
        html += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        html += "<tr><th>Title</th><th>Company</th><th>Location</th><th>AI Verdict</th><th>Match %</th><th>Link</th></tr>"
        for _, row in jobs_df.iterrows():
            title = row.get("title", "N/A")
            company = row.get("company", "N/A")
            location = row.get("location", "Remote/Unspecified")
            verdict = row.get("ai_verdict", "")
            match_pct = row.get("match_percentage", "N/A")
            if match_pct != "N/A": match_pct = f"{match_pct}%"
            job_url = row.get("job_url", "#")
            html += f"<tr><td>{title}</td><td>{company}</td><td>{location}</td><td>{verdict}</td><td>{match_pct}</td><td><a href='{job_url}'>Apply</a></td></tr>"
        html += "</table>"
        
    return html

def fetch_remotive_jobs():
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
    try:
        url = "https://jobicy.com/api/v2/remote-jobs?jobCategory=programming"
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

def filter_api_jobs(df, hours_old):
    if df.empty:
        return df
    
    role_keywords = ['software', 'developer', 'engineer', 'ai', 'data', 'machine learning', 'backend', 'frontend', 'fullstack', 'python', 'java']
    
    title_lower = df['title'].str.lower()
    has_role = title_lower.str.contains('|'.join(role_keywords), na=False)
    # Removed the 'has_exp' double filter here as recommended in Tier 1. Let AI decide seniority.
    df = df[has_role].copy()
    
    # 2. Filter by recency (hours_old)
    try:
        df['date_posted_dt'] = pd.to_datetime(df['date_posted'], utc=True, errors='coerce')
        now = pd.Timestamp.utcnow()
        cutoff = now - pd.Timedelta(hours=hours_old)
        df = df[df['date_posted_dt'].isna() | (df['date_posted_dt'] >= cutoff)]
        df = df.drop(columns=['date_posted_dt'])
    except Exception as e:
        print(f"Date filtering error: {e}")
        
    return df

def send_email(subject, html_content, email_settings):
    sender_email = os.environ.get("SENDER_EMAIL")
    app_password = os.environ.get("EMAIL_APP_PASSWORD")
    
    if not sender_email or not app_password:
        print("Error: SENDER_EMAIL or EMAIL_APP_PASSWORD environment variables are not set.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = ", ".join(email_settings["receiver_emails"])

    msg.attach(MIMEText(html_content, "html"))

    try:
        # Connect to SMTP server (using SSL for port 465)
        with smtplib.SMTP_SSL(email_settings["smtp_server"], email_settings["smtp_port"]) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

def format_github_markdown(internships_df, jobs_df):
    md = "## Automated AI Job Alerts\n\n"
    
    # --- Internships Table ---
    md += "### 🎓 Internships (AI & SWE)\n\n"
    if internships_df.empty:
        md += "No relevant internships found today.\n\n"
    else:
        md += "| Title | Company | Location | AI Verdict | Match % | Link |\n"
        md += "|---|---|---|---|---|---|\n"
        for _, row in internships_df.iterrows():
            title = row.get("title", "N/A")
            company = row.get("company", "N/A")
            location = row.get("location", "Remote/Unspecified")
            verdict = row.get("ai_verdict", "")
            match_pct = row.get("match_percentage", "N/A")
            if match_pct != "N/A": match_pct = f"{match_pct}%"
            job_url = row.get("job_url", "#")
            md += f"| {title} | {company} | {location} | {verdict} | {match_pct} | [Apply]({job_url}) |\n"
        md += "\n"
        
    # --- Full-Time Jobs Table ---
    md += "### 💼 Junior & Entry-Level Jobs (CV-Matched)\n\n"
    if jobs_df.empty:
        md += "No relevant full-time jobs found today.\n\n"
    else:
        md += "| Title | Company | Location | AI Verdict | Match % | Link |\n"
        md += "|---|---|---|---|---|---|\n"
        for _, row in jobs_df.iterrows():
            title = row.get("title", "N/A")
            company = row.get("company", "N/A")
            location = row.get("location", "Remote/Unspecified")
            verdict = row.get("ai_verdict", "")
            match_pct = row.get("match_percentage", "N/A")
            if match_pct != "N/A": match_pct = f"{match_pct}%"
            job_url = row.get("job_url", "#")
            md += f"| {title} | {company} | {location} | {verdict} | {match_pct} | [Apply]({job_url}) |\n"
            
    return md

def create_github_issue(title, body):
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    
    if not token or not repo:
        print("Error: GITHUB_TOKEN or GITHUB_REPOSITORY environment variables are not set.")
        return
        
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "title": title,
        "body": body
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(f"GitHub Issue created successfully: {response.json().get('html_url')}")
    except Exception as e:
        print(f"Failed to create GitHub Issue: {e}")

def get_full_job_description(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64 AppleWebKit/537.36)'}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            if "authwall" in res.url.lower() or "sign in to linkedin" in res.text.lower():
                return "[DESCRIPTION TRUNCATED BY LINKEDIN LOGIN WALL]"
            soup = BeautifulSoup(res.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator=' ')
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return '\n'.join(chunk for chunk in chunks if chunk)
    except:
        pass
    return ""

def search_company_remote_policy(company_name, job_title):
    print(f"Deep web search triggered for {company_name} ({job_title}) remote policy...")
    snippets = []
    try:
        # Query 1: Position specific remote rules
        q1 = f"{company_name} \"{job_title}\" remote eligible countries"
        res1 = DDGS().text(q1, max_results=2)
        snippets.extend([r.get('body', '') for r in res1])
        
        # Query 2: General company hiring in Middle East / Palestine
        q2 = f"{company_name} hire remote Middle East Palestine EMEA"
        res2 = DDGS().text(q2, max_results=2)
        snippets.extend([r.get('body', '') for r in res2])
        
        return " ".join(snippets)
    except Exception as e:
        print(f"Web search failed: {e}")
        return " ".join(snippets)

def evaluate_job_with_ai(row, cv_text, api_key):
    if not api_key:
        return "No API Key provided", True, "N/A"
        
    genai.configure(api_key=api_key)
    # Using Gemini 3.1 Flash Lite which has 500 RPD and 15 RPM limits
    model = genai.GenerativeModel('gemini-3.1-flash-lite')
    
    title = str(row.get("title", ""))
    company = str(row.get("company", ""))
    job_type = str(row.get("job_type", "")).lower()
    description = str(row.get("description", ""))
    if pd.isna(description) or len(description) < 100:
        description = get_full_job_description(str(row.get("job_url", "")))
        if not description:
            description = "[NO DESCRIPTION AVAILABLE - SCRAPING BLOCKED]"
        
    is_internship = 'intern' in title.lower() or 'internship' in job_type
    
    web_search_context = ""
    web_search_triggers = [
        "eligible countries", "selected countries", "certain countries", "based in", 
        "residents of", "remote in", "must be located", "work authorization", 
        "within the united states", "us only", "us-based", "uk only", "eu only"
    ]
    if any(trigger in description.lower() for trigger in web_search_triggers):
        search_data = search_company_remote_policy(company, title)
        if search_data:
            web_search_context = f"\n\n[LIVE WEB SEARCH RESULTS FOR '{company}' REMOTE POLICY]:\n{search_data}\n\nUse this live web data to determine if Palestine/Middle East is explicitly excluded from their remote eligible countries."
            
    prompt = f"""You are an expert technical recruiter. 
Candidate's CV Summary:
{cv_text[:3000]}

Job Title: {title}
Is this an Internship?: {is_internship}
Job Description:
{description[:5000]}

Evaluate based on these STRICT rules:
1. REMOTE LOCATION CHECK: Deeply analyze the remote policy. {web_search_context}
   - If the description or web search explicitly restricts remote work to specific regions/countries (e.g. "Remote in US/UK/EU", "Must be resident of...") and does NOT include Palestine, EMEA, or Middle East, it FAILS.
   - If the description says "Eligible countries" but the web search data reveals Palestine/Middle East is NOT eligible, it FAILS.
   - If it explicitly says "Worldwide", "Global", "EMEA", or simply "Remote" with absolutely no geographic restrictions found, it PASSES.
   - If it is ambiguous but there is NO evidence excluding Palestine/Middle East, assume it is PASSABLE but note the ambiguity in the verdict.
2. If this is an Internship, it MUST be strictly related to Software Engineering, Machine Learning, Data, or AI. If it is an HR, Marketing, or random internship, it FAILS.
3. If this is a Full-Time job, evaluate if the candidate's CV matches for an Entry-level/Junior role. Allow leniency if they have strong general ML/Python/FastAPI background.
4. MATCH PERCENTAGE: Mathematically calculate a realistic match percentage (0-100) based strictly on how the candidate's skills and experience in the CV align with the job description's requirements. Deduct points proportionally for missing core requirements. Output as a clean integer (e.g. 88, not 88.23).
5. LIMITED INFO PROTOCOL: If the description says [DESCRIPTION TRUNCATED...] or [NO DESCRIPTION AVAILABLE...], rely solely on the job title, company, and web search results to make your decision, and note the missing description in your verdict.

Reply ONLY with valid JSON in this exact format, with no markdown formatting:
{{"is_valid": true/false, "verdict": "A 1-sentence reason for your decision", "match_percentage": 85}}
"""
    try:
        # Sleep for 4 seconds to strictly stay under the 15 Requests Per Minute free tier limit
        time.sleep(4)
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
        import json
        result = json.loads(text)
        return result.get("verdict", "AI Approved"), result.get("is_valid", True), result.get("match_percentage", "N/A")
    except Exception as e:
        print(f"AI evaluation failed for {title}: {str(e)}")
        error_msg = str(e).replace('"', "'")
        return f"AI Error: {error_msg[:100]}...", False, "N/A"

def main():
    config = load_config()
    all_jobs_dfs = []
    
    print("Starting job scrape...")
    for search in config.get("searches", []):
        print(f"Scraping for: {search.get('search_term')} in {search.get('location')}...")
        try:
            jobs = scrape_jobs(
                site_name=search.get("site_name", ["linkedin", "indeed", "glassdoor"]),
                search_term=search.get("search_term"),
                location=search.get("location"),
                distance=search.get("distance", 50),
                job_type=search.get("job_type"),
                is_remote=search.get("is_remote", False),
                results_wanted=search.get("results_wanted", 20),
                hours_old=search.get("hours_old", 24),
                country_indeed=search.get("country_indeed", "USA")
            )
            print(f"Found {len(jobs)} jobs for this search.")
            all_jobs_dfs.append(jobs)
        except Exception as e:
            print(f"Error scraping for {search.get('search_term')}: {e}")
            
    # Determine the maximum hours_old from config to use for APIs
    max_hours = 24
    if config.get("searches"):
        max_hours = max([s.get("hours_old", 24) for s in config.get("searches", [])])

    # Fetch from secondary APIs
    print("Fetching from Remotive API...")
    remotive_df = fetch_remotive_jobs()
    if not remotive_df.empty:
        remotive_df = filter_api_jobs(remotive_df, hours_old=max_hours)
        print(f"Found {len(remotive_df)} relevant jobs from Remotive.")
        if not remotive_df.empty:
            all_jobs_dfs.append(remotive_df)
            
    print("Fetching from Arbeitnow API...")
    arbeitnow_df = fetch_arbeitnow_jobs()
    if not arbeitnow_df.empty:
        arbeitnow_df = filter_api_jobs(arbeitnow_df, hours_old=max_hours)
        print(f"Found {len(arbeitnow_df)} relevant jobs from Arbeitnow.")
        if not arbeitnow_df.empty:
            all_jobs_dfs.append(arbeitnow_df)
            
    print("Fetching from Jobicy API...")
    jobicy_df = fetch_jobicy_jobs()
    if not jobicy_df.empty:
        jobicy_df = filter_api_jobs(jobicy_df, hours_old=max_hours)
        print(f"Found {len(jobicy_df)} relevant jobs from Jobicy.")
        if not jobicy_df.empty:
            all_jobs_dfs.append(jobicy_df)

    print("Fetching from RemoteOK API...")
    remoteok_df = fetch_remoteok_jobs()
    if not remoteok_df.empty:
        remoteok_df = filter_api_jobs(remoteok_df, hours_old=max_hours)
        print(f"Found {len(remoteok_df)} relevant jobs from RemoteOK.")
        if not remoteok_df.empty:
            all_jobs_dfs.append(remoteok_df)
            
    if all_jobs_dfs:
        combined_jobs = pd.concat(all_jobs_dfs, ignore_index=True)
        
        # Drop duplicates by URL
        if "job_url" in combined_jobs.columns:
            combined_jobs = combined_jobs.drop_duplicates(subset=["job_url"])
            
        # Smarter deduplication: Drop duplicates by normalized Title + Company
        if "title" in combined_jobs.columns and "company" in combined_jobs.columns:
            combined_jobs['norm_title'] = combined_jobs['title'].astype(str).str.replace(r'\s*\(.*?\)', '', regex=True).str.strip().str.lower()
            combined_jobs['norm_company'] = combined_jobs['company'].astype(str).str.strip().str.lower()
            combined_jobs = combined_jobs.drop_duplicates(subset=["norm_title", "norm_company"])
            combined_jobs = combined_jobs.drop(columns=['norm_title', 'norm_company'])
            
        # Language Pre-filter: Reject Chinese/Korean/Japanese titles
        import re
        if "title" in combined_jobs.columns:
            combined_jobs = combined_jobs[~combined_jobs['title'].astype(str).str.contains(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', na=False)]
            
        # Location Pre-filter: Drop clearly location-locked jobs that don't say remote
        if "location" in combined_jobs.columns:
            explicit_non_remote = ['shanghai', 'beijing', 'mumbai', 'bangalore', 'moscow', 'tx', 'ca', 'ny', 'california', 'texas', 'new york', 'india', 'china', 'russia']
            pattern = '|'.join([rf'\b{loc}\b' for loc in explicit_non_remote])
            remote_in_loc = combined_jobs['location'].astype(str).str.lower().str.contains('remote')
            remote_in_title = combined_jobs['title'].astype(str).str.lower().str.contains('remote')
            bad_loc = combined_jobs['location'].astype(str).str.lower().str.contains(pattern, na=False)
            combined_jobs = combined_jobs[~(bad_loc & ~remote_in_loc & ~remote_in_title)]
            
        # Filter out senior/lead roles
        exclude_words = ['senior', 'sr', 'sr.', 'lead', 'principal', 'manager', 'director', 'staff', 'head', 'vp', 'president']
        if "title" in combined_jobs.columns:
            pattern = '|'.join([rf'\b{w}\b' for w in exclude_words])
            combined_jobs = combined_jobs[~combined_jobs['title'].str.lower().str.contains(pattern, na=False)]
            
        # Ensure job title is highly relevant (must contain at least one tech keyword)
        role_keywords = ['software', 'developer', 'engineer', 'ai', 'data', 'machine learning', 'backend', 'frontend', 'fullstack', 'web', 'python', 'java', 'c\\+\\+', 'c#', 'programmer']
        if "title" in combined_jobs.columns:
            pattern = '|'.join([rf'{w}' for w in role_keywords])
            combined_jobs = combined_jobs[combined_jobs['title'].str.lower().str.contains(pattern, na=False)]
            
        print(f"Total unique jobs found before AI filtering: {len(combined_jobs)}")
        
        if not combined_jobs.empty:
            # --- AI EVALUATION ---
            gemini_key = os.environ.get("GEMINI_API_KEY", "")
            try:
                with open("cv_text.txt", "r", encoding="utf-8") as f:
                    cv_text = f.read()
            except:
                cv_text = "Computer Engineering student, strong in Python, PyTorch, FastAPI, RAG, ML, and Backend Development."
            
            print("Running AI Job Validation 1-by-1 (this may take a while)...")
            verdicts = []
            valid_mask = []
            match_pcts = []
            for idx, row in combined_jobs.iterrows():
                verdict, is_valid, match_pct = evaluate_job_with_ai(row, cv_text, gemini_key)
                verdicts.append(verdict)
                valid_mask.append(is_valid)
                match_pcts.append(match_pct)
            
            combined_jobs['ai_verdict'] = verdicts
            combined_jobs['match_percentage'] = match_pcts
            combined_jobs = combined_jobs[valid_mask]
            print(f"Total jobs remaining after AI validation: {len(combined_jobs)}")
            
            # Split into internships and jobs
            intern_mask = combined_jobs['title'].str.lower().str.contains('intern') | (combined_jobs.get('job_type', pd.Series(dtype=str)).astype(str).str.lower().str.contains('internship'))
            internships_df = combined_jobs[intern_mask]
            jobs_df = combined_jobs[~intern_mask]
            
            output_config = config.get("output", {"use_email": True, "use_github_issue": False})
            
            if output_config.get("use_email"):
                html_content = format_email_html(internships_df, jobs_df)
                send_email("Your Automated AI Job Alerts", html_content, config.get("email_settings", {}))
                
            if output_config.get("use_github_issue"):
                md_content = format_github_markdown(internships_df, jobs_df)
                today = datetime.now().strftime("%Y-%m-%d")
                create_github_issue(f"Automated AI Job Alerts - {today}", md_content)
        else:
            print("No new jobs found.")
    else:
        print("No job data collected.")

if __name__ == "__main__":
    main()
