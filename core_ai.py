import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from google import genai
from ddgs import DDGS
import json

"""
CORE AI MODULE
--------------
The brain of the operation. Uses DuckDuckGo to deep-search the web for remote policies 
and leverages Google's Gemini AI to mathematically evaluate the candidate's CV against 
the specific job requirements, yielding a precise verdict and Match %.
"""

def get_full_job_description(url):
    """Fallback scraper for job URLs when the API only returns a truncated description."""
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
    """
    Executes a dual web-search using DuckDuckGo to find specific geographic restrictions 
    for the company and the position.
    """
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
    """
    Sends the job description and candidate CV to Gemini 3.1 Flash Lite to 
    determine geographic eligibility and calculate a mathematical Match %.
    """
    if not api_key:
        return "No API Key provided", False, "N/A"

    client = genai.Client(api_key=api_key)
    
    title = str(row.get("title", ""))
    company = str(row.get("company", ""))
    job_type = str(row.get("job_type", "")).lower()
    description = str(row.get("description", ""))
    
    # If description is missing or too short, attempt to scrape it directly
    if pd.isna(description) or len(description) < 100:
        description = get_full_job_description(str(row.get("job_url", "")))
        if not description:
            description = "[NO DESCRIPTION AVAILABLE - SCRAPING BLOCKED]"
        
    is_internship = 'intern' in title.lower() or 'internship' in job_type
    
    # Check if we need to do a deep web search for remote policy ambiguity
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
        time.sleep(4)
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt
        )
        text = response.text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
        result = json.loads(text)
        return result.get("verdict", "AI Approved"), result.get("is_valid", True), result.get("match_percentage", "N/A")
    except Exception as e:
        print(f"AI evaluation failed for {title}: {str(e)}")
        error_msg = str(e).replace('"', "'")
        # Flip to False to prevent garbage jobs from passing on AI errors
        return f"AI Error: {error_msg[:100]}...", False, "N/A"
