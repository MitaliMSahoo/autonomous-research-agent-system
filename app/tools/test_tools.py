import asyncio
from app.tools.search import tavily_search
from app.tools.scraper import scrape_url

if __name__ == "__main__":
    # Test search
    print("=== Test Tavily Search ===")
    results = tavily_search.invoke({"query": "nuclear fusion latest", "max_results": 3})
    for r in results:
        print(f"\n🔗 {r['title']}")
        print(f"   {r['url']}")

    # Test scraper
    print("\n=== Test Scraper ===")
    if results:
        url = results[0]["url"]
        content = scrape_url.invoke({"url": url})
        print(f"\n📄 {url}\n")
        print(content[:800])