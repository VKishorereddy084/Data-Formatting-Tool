import asyncio
import nest_asyncio
import os
import re
import hashlib
import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher


# ---- Helper Components ----

def create_pruning_filter():
    return PruningContentFilter(
        threshold=0.1,
        threshold_type="dynamic",
        min_word_threshold=5
    )


def create_markdown_generator(prune_filter):
    return DefaultMarkdownGenerator(
        content_filter=prune_filter,
        options={
            "ignore_links": True,
            "escape_html": False,
            "body_width": 50,
            "ignore_images": False,
            "ignore_tables": True
        }
    )


def create_dispatcher():
    return MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=10
    )


def create_crawler_config(md_generator):
    return CrawlerRunConfig(
        markdown_generator=md_generator,
        word_count_threshold=15,
        exclude_external_images=True,
        exclude_external_links=True,
        process_iframes=True,
        remove_overlay_elements=True,
        exclude_social_media_links=True,
        check_robots_txt=False,
        wait_for_images=True,
        semaphore_count=3,
        cache_mode=CacheMode.BYPASS
    )


def make_file_safe(url: str, max_len: int = 50) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace('.', '_')
    path = parsed.path.rstrip('/')
    slug = os.path.basename(path) or "homepage"
    combined = f"{domain}_{slug}"
    safe = re.sub(r'[^A-Za-z0-9_-]', '_', combined)
    if len(safe) > max_len:
        h = hashlib.sha1(url.encode('utf-8')).hexdigest()[:8]
        safe = f"{safe[: max_len - 9]}_{h}"
    return safe


# ---- Sitemap & Link Discovery ----

async def fetch_sitemap(url):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url)
        return result.html if result.success else None


async def extract_urls_from_sitemap(sitemap_url):
    xml_content = await fetch_sitemap(sitemap_url)
    if not xml_content:
        return []

    try:
        root = ET.fromstring(xml_content)
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        return [url.text for url in root.findall(".//ns:loc", ns)]
    except ET.ParseError:
        print(f" Sitemap at {sitemap_url} is invalid. Skipping...")
        return []


def extract_internal_links(base_url):
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        base_domain = urlparse(base_url).netloc
        filtered_links = []
        seen_urls = set()

        exclusion_keywords = ['#', 'signup', 'login', 'contact', 'help', 'terms', 'privacy', 'copyright', 'contrib']

        for link in soup.find_all('a'):
            relative_url = link.get('href')
            if relative_url:
                absolute_url = urljoin(base_url, relative_url)
                parsed_url = urlparse(absolute_url)
                if (
                    parsed_url.netloc == base_domain
                    and absolute_url not in seen_urls
                    and not any(keyword in absolute_url for keyword in exclusion_keywords)
                ):
                    filtered_links.append(absolute_url)
                    seen_urls.add(absolute_url)

        return filtered_links

    except requests.exceptions.RequestException as e:
        print(f" Error fetching {base_url}: {e}")
        return []


async def discover_urls(base_url):
    sitemap_url = f"{base_url}/sitemap.xml"
    sitemap_links = await extract_urls_from_sitemap(sitemap_url)

    if not sitemap_links:
        print(" No sitemap found! Extracting internal links instead (ordered)...")
        sitemap_links = extract_internal_links(base_url)

    print(f"🔍 Total Pages Found: {len(sitemap_links)}")
    return sitemap_links


# ---- Core Crawling ----

async def process_url(crawler, url, config, dispatcher, output_dir):
    result = await crawler.arun(url=url, config=config, dispatcher=dispatcher)

    if result.success:
        file_safe_url = make_file_safe(url)
        raw_file = os.path.join(output_dir, f"result_raw_{file_safe_url}.md")
        filtered_file = os.path.join(output_dir, f"result_filtered_{file_safe_url}.md")

        images_info = result.media.get("images", [])
        tables = result.media.get("tables", [])

        print(f"Found {len(images_info)} images.")
        print(f"Found {len(tables)} tables.")

        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(result.markdown.raw_markdown)
            print(f" Raw markdown saved to: {raw_file}")

        with open(filtered_file, "w", encoding="utf-8") as f:
            f.write(result.markdown.fit_markdown)
            print(f" Filtered markdown saved to: {filtered_file}")

        return {"url": url, "raw": raw_file, "filtered": filtered_file}
    else:
        print(f"❌ FAILED for {url} | Error: {result.error_message}")
        return None


async def crawl_urls(selected_urls, output_dir: Path):
    browser_config = BrowserConfig(verbose=True)
    prune_filter = create_pruning_filter()
    md_generator = create_markdown_generator(prune_filter)
    dispatcher = create_dispatcher()
    config = create_crawler_config(md_generator)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        tasks = [
            process_url(crawler, url, config, dispatcher, output_dir)
            for url in selected_urls
        ]
        completed = await asyncio.gather(*tasks)

    return [res for res in completed if res]


# ---- Flask Integration Wrappers ----

def get_discovered_urls(base_url: str) -> list:
    nest_asyncio.apply()
    return asyncio.run(discover_urls(base_url))


def run_crawl_on_selected_urls(url_list: list, output_dir: Path) -> list:
    nest_asyncio.apply()
    return asyncio.run(crawl_urls(url_list, output_dir))
