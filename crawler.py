import requests
from bs4 import BeautifulSoup
import pdfkit
import os

wkhtmltopdf_path = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)


visited_urls = set()  # Keep track of visited URLs to avoid repetition
urls_to_crawl = ["https://www.healthunit.com/"]  # Starting URL

def fetch_html(url):
    """Fetch the HTML content of a URL."""
    try:
        response = requests.get(url)
        content_type = response.headers.get('Content-Type')

        if 'text/html' in content_type:
            #soup = BeautifulSoup(response.content, 'html.parser')
            response.raise_for_status()  # Checks if the request was successful
            return response.text
        else:
            print(f"Error fetching {url}: Skipping non-HTML content: {content_type}")
            return None        
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def crawl_and_save(start_url):
    """Crawl the website starting from the given URL and save each page as a PDF."""
    while urls_to_crawl:
        current_url = urls_to_crawl.pop(0)
        if current_url not in visited_urls:
            print(f"Crawling: {current_url}")
            visited_urls.add(current_url)
            html_content = fetch_html(current_url)
            if html_content:
                # Convert and save the current page to PDF
                pdf_filename = f"output/{current_url.replace('https://www.healthunit.com/', '').replace('/', '_') or 'index'}.pdf"
                option = {
                    'enable-local-file-access': None,
                    'no-images': None,  # Add this if you want to disable loading images
                    'disable-javascript': None
                    }
                pdfkit.from_string(html_content, pdf_filename,options=option ,configuration=config)
                print(f"Saved {pdf_filename}")

                # Parse the HTML for links and add them to the crawl queue
                soup = BeautifulSoup(html_content, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.startswith('/'):
                        full_url = f"https://www.healthunit.com/{href}"
                        if full_url not in visited_urls:
                            urls_to_crawl.append(full_url)

# Create output directory if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

# Start crawling from the main page
crawl_and_save("https://www.healthunit.com/")
