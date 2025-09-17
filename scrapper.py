import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from typing import List, Dict
from sqlalchemy.orm import Session
from database import SessionLocal  # ✅ import SessionLocal directly
from models import Blog            # ✅ import Blog model

# Define headers
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

# Define all websites with their main listing pages
WEBSITE_URLS = {
    "outreach": {
        "blogs": "https://www.outreach.io/resources/blog",
        "webinars": "https://www.outreach.io/resources/webinars-videos/role/revenue-operations",
        "reports": "https://www.outreach.io/resources/reports-guides",
        "stories": "https://www.outreach.io/resources/stories",
        "tools": "https://www.outreach.io/resources/tools",
        "product_updates": "https://www.outreach.io/product-updates"
    },
    "xactly": {
        "blogs": "https://www.xactlycorp.com/blog",
        "webinars": "https://www.xactlycorp.com/resources/webinars",
        "infographics": "https://www.xactlycorp.com/resources/infographics"
    },
    "gong": {
        "blogs": "https://www.gong.io/blog/",
        "resources": "https://www.gong.io/resources/",
        "guides": "https://www.gong.io/resources/guides/what-is-revenue-intelligence/",
        "webinars": "https://www.gong.io/resources/webinars/",
        "labs": "https://www.gong.io/resources/labs/",
        "podcast": "https://www.gong.io/podcast/"
    },
    "ziphq": {
        "blogs": "https://ziphq.com/blog?blog-topic=Industry+trends",
        "webinars": "https://ziphq.com/webinars",
        "events": "https://ziphq.com/events",
        "reports": "https://ziphq.com/research-reports-guides",
        "resources": "https://ziphq.com/resources"
    }
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

        # Collect all links that belong to the same website
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(listing_url, href)
            if full_url.startswith(listing_url) or any(
                domain in full_url for domain in ["outreach.io", "xactlycorp.com", "gong.io", "ziphq.com"]
            ):
                if full_url not in links:
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


def scrape_website(website: str) -> List[Dict]:
    """Scrape all resources from the selected website, classified by category."""
    website = website.lower()
    if website not in WEBSITE_URLS:
        raise ValueError(f"Unsupported website: {website}")

    results = []
    idx = 1

    for category, listing_page in WEBSITE_URLS[website].items():
        print(f"Scraping {category} page: {listing_page}")
        links = scrape_listing_page(listing_page)
        print(f"Found {len(links)} links in {category}")

        for link in links:
            data = scrape_page(link)
            if data:
                data["id"] = idx
                data["website"] = website
                data["category"] = category   # ✅ classify here
                results.append(data)
                idx += 1
                time.sleep(1)  # polite delay

    return results


def save_blogs_to_db(website: str):
    """Scrape and persist data into DB with classification."""
    db: Session = SessionLocal()
    try:
        scraped_data = scrape_website(website)

        for item in scraped_data:
            existing = db.query(Blog).filter(Blog.url == item["url"]).first()
            if existing:
                continue

            blog = Blog(
                website=item["website"],
                category=item["category"],  # ✅ store category
                title=item["title"],
                description=item["description"],
                content=item["content"],
                url=item["url"]
            )
            db.add(blog)

        db.commit()
    finally:
        db.close()
