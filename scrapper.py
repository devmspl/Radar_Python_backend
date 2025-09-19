import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from typing import List, Dict
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Blog, ScrapeJob
import uuid


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

def scrape_listing_page(listing_url: str) -> List[str]:
    """Extract all sub-page links from a listing page."""
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        response = session.get(listing_url, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch {listing_url}")
            return []

        soup = BeautifulSoup(response.content, "html.parser")
        links = []

        base_domain = urlparse(listing_url).netloc

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(listing_url, href)
            if urlparse(full_url).netloc == base_domain and full_url not in links:
                links.append(full_url)

        return links
    except Exception as e:
        print(f"Error fetching {listing_url}: {e}")
        return []

def scrape_page(url: str) -> Dict:
    """Scrape content from a single page."""
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            return {}

        soup = BeautifulSoup(response.content, "html.parser")

        title = soup.find("h1")
        title = title.get_text().strip() if title else soup.title.string if soup.title else "No title"

        description = soup.find("meta", {"name": "description"})
        description = description.get("content", "").strip() if description else "No description"

        content_selectors = ["article", ".blog-content", ".post-content", ".content", "main"]
        content = ""
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                content = element.get_text(separator="\n\n", strip=True)
                break
        if not content:
            content = soup.get_text(separator="\n\n", strip=True)

        return {
            "title": title,
            "description": description,
            "content": content,
            "url": url
        }

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {}

def scrape_any_website(listing_url: str, category: str = "general") -> List[Dict]:
    """Scrape a given URL dynamically and classify pages under a category."""
    results = []
    idx = 1

    print(f"Scraping listing page: {listing_url}")
    links = scrape_listing_page(listing_url)
    print(f"Found {len(links)} links")

    for link in links:
        data = scrape_page(link)
        if data:
            data["id"] = idx
            data["website"] = urlparse(listing_url).netloc
            data["category"] = category
            results.append(data)
            idx += 1
            time.sleep(1)  # polite delay

    return results

def save_any_website_to_db(listing_url: str, category: str = "general"):
    """Scrape any given URL and save to DB."""
    db: Session = SessionLocal()
    try:
        scraped_data = scrape_any_website(listing_url, category)

        for item in scraped_data:
            existing = db.query(Blog).filter(Blog.url == item["url"]).first()
            if existing:
                continue

            blog = Blog(
                website=item["website"],
                category=item["category"],
                title=item["title"],
                description=item["description"],
                content=item["content"],
                url=item["url"],
                job_uid=job.uid
            )
            db.add(blog)

        db.commit()
    finally:
        db.close()
def start_scraping_job(db: Session, url: str, category: str = "general"):
    job_uid = str(uuid.uuid4())
    job = ScrapeJob(uid=job_uid, website=urlparse(url).netloc, url=url, status="inprocess")
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        data = scrape_any_website(url, category)
        for item in data:
            existing = db.query(Blog).filter(Blog.url == item["url"]).first()
            if existing:
                continue
            blog = Blog(
                website=item["website"],
                category=item["category"],
                title=item["title"],
                description=item["description"],
                content=item["content"],
                url=item["url"]
            )
            db.add(blog)
        db.commit()
        job.status = "done"
    except Exception as e:
        job.status = "failed"
    finally:
        db.commit()

    return job.uid