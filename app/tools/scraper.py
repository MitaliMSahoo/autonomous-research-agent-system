from langchain_core.tools import tool
import httpx
import trafilatura

@tool
def scrape_url(url: str, timeout: int = 15) -> str:
    """Fetch a URL and extract clean readable text (strips nav, ads, boilerplate)."""
    try:
        # A more robust set of headers makes the request look like a legitimate browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }
        
        # Using a client instance handles cookies and protocol negotiation better
        with httpx.Client(headers=headers, http2=True) as client:
            resp = client.get(url, timeout=timeout, follow_redirects=True)
            resp.raise_for_status()
            
        text = trafilatura.extract(resp.text)
        return text[:8000] if text else "Could not extract content from this page."
        
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"




if __name__ == "__main__":
    content = scrape_url.invoke({"url": "https://en.wikipedia.org/wiki/Nuclear_fusion"})
    print(content[:1000])
    content = scrape_url.invoke({"url": "https://en.wikipedia.org/wiki/List_of_video_games_released_in_2026"})
    print(content[:1000])