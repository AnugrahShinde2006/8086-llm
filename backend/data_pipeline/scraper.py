import os
import requests
from bs4 import BeautifulSoup
import re

def clean_text(text):
    text = re.sub(r'\[\d+\]', '', text) # Remove Wikipedia style citations [1]
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def scrape_article(url):
    print(f"Scraping {url}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract paragraphs
        paragraphs = soup.find_all('p')
        text_content = ""
        for p in paragraphs:
            cleaned = clean_text(p.get_text())
            if len(cleaned) > 30:
                text_content += cleaned + "\n"
                
        # Extract code blocks if any (pre tags)
        code_blocks = soup.find_all('pre')
        code_content = ""
        for pre in code_blocks:
            code = pre.get_text().strip()
            if code:
                code_content += f"<|assembly|>\n{code}\n<|endoftext|>\n"
                
        # Format for our dataset
        dataset_entry = f"<|user|>\nExtract knowledge from this article: {url}\n<|assistant|>\n{text_content}\n<|endoftext|>\n"
        if code_content:
            dataset_entry += code_content
            
        return dataset_entry
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return ""

def crawl_domain(start_url, max_pages=10):
    print(f"\n--- Starting Web Spider at {start_url} ---")
    visited = set()
    to_visit = [start_url]
    scraped_content = []
    
    # Simple regex to get the base domain so we don't crawl the whole internet
    domain_match = re.search(r'https?://([^/]+)', start_url)
    base_domain = domain_match.group(1) if domain_match else ""
    
    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        
        if current_url in visited:
            continue
            
        visited.add(current_url)
        print(f"[{len(visited)}/{max_pages}] Crawling: {current_url}")
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(current_url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 1. Extract text and code like before
            paragraphs = soup.find_all('p')
            text_content = ""
            for p in paragraphs:
                cleaned = clean_text(p.get_text())
                if len(cleaned) > 30:
                    text_content += cleaned + "\n"
                    
            code_blocks = soup.find_all('pre')
            code_content = ""
            for pre in code_blocks:
                code = pre.get_text().strip()
                if code:
                    code_content += f"<|assembly|>\n{code}\n<|endoftext|>\n"
                    
            if text_content:
                dataset_entry = f"<|user|>\nExtract knowledge from this article: {current_url}\n<|assistant|>\n{text_content}\n<|endoftext|>\n"
                if code_content:
                    dataset_entry += code_content
                scraped_content.append(dataset_entry)
                
            # 2. Find new links to add to the queue (Spider behavior)
            valid_keywords = ['8086', 'x86', 'microprocessor', 'assembly', 'intel', 'processor', 'architecture']
            for link in soup.find_all('a', href=True):
                href = link['href']
                # Reconstruct relative URLs
                if href.startswith('/'):
                    # Wikipedia specific cleanup: skip Special, Talk, User pages, and index.php actions
                    if any(skip in href for skip in [':', 'index.php']):
                        continue
                    href = f"https://{base_domain}{href}"
                    
                # Only follow links on the same domain
                if base_domain in href and href not in visited and href not in to_visit:
                    # Filter: Only add the URL if it seems relevant to our project
                    if any(keyword in href.lower() for keyword in valid_keywords):
                        to_visit.append(href)
                    
        except Exception as e:
            print(f" Failed: {e}")
            
    return scraped_content

def generate_scraped_dataset(start_urls, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for url in start_urls:
            # Let the spider crawl up to 5 related pages per starting URL
            contents = crawl_domain(url, max_pages=5)
            for content in contents:
                f.write(content + "\n")
                
    print(f"\nScraping complete. Saved data to {output_file}")

if __name__ == "__main__":
    # The spider will start here and automatically click links!
    urls_to_scrape = [
        "https://en.wikipedia.org/wiki/Intel_8086"
    ]
    
    # Save properly relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DATASET = os.path.join(script_dir, "../data/raw/scraped_data.txt")
    generate_scraped_dataset(urls_to_scrape, OUTPUT_DATASET)
