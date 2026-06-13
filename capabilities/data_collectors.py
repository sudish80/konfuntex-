"""
Phases 11-20 — Data Collection from various sources.

Provides:
  - YouTubeAudioDownloader  download + transcribe audio    (11)
  - TwitterScraper          scrape tweets by query          (12)
  - PDFKnowledgeExtractor   parse + chunk PDFs / ArXiv     (13)
  - GitHubReadmeCollector   collect repo READMEs           (14)
  - WikipediaDumpStreamer   stream Wikipedia dumps         (15)
  - SlackExportParser       parse Slack export ZIPs         (16)
  - ArxivPaperFetcher       search + download ArXiv papers (17)
  - RedditAPICollector      collect Reddit posts/comments  (18)
  - CommonCrawlExtractor    extract from CommonCrawl WARC  (19)
  - SeleniumDynamicScraper  scrape JS-rendered pages       (20)
"""
import os


# ==================================================================== #
#  11 — YouTubeAudioDownloader
# ==================================================================== #

class YouTubeAudioDownloader:
    """
    Download audio from YouTube videos using yt-dlp,
    convert to 16 kHz mono WAV, transcribe with Whisper,
    and produce a HuggingFace Dataset.
    """

    SUPPORTED_FORMATS = ["opus", "m4a", "wav"]
    WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

    def __init__(self, output_dir: str = "./youtube_audio",
                 whisper_model: str = "tiny",
                 sample_rate: int = 16000):
        self.output_dir = output_dir
        self.whisper_model = whisper_model
        self.sample_rate = sample_rate
        os.makedirs(output_dir, exist_ok=True)

    def download_code(self, url: str, format: str = "wav") -> str:
        format_flag = {"opus": "251", "m4a": "140", "wav": "bestaudio/best"}.get(format, "bestaudio/best")
        postprocessor = ""
        if format == "wav":
            postprocessor = f"""
# Convert to 16kHz mono WAV using ffmpeg
import subprocess
input_path = f"{{audio_path}}"
output_path = input_path.rsplit('.', 1)[0] + '.wav'
subprocess.run([
    "ffmpeg", "-i", input_path,
    "-ar", "{self.sample_rate}", "-ac", "1",
    output_path, "-y"
], check=True, capture_output=True)
audio_path = output_path
"""
        return f"""
import yt_dlp, subprocess, os

url = "{url}"
output_template = os.path.join("{self.output_dir}", "%(title)s.%(ext)s")

ydl_opts = {{
    "format": "{format_flag}",
    "outtmpl": output_template,
    "quiet": True,
    "no_warnings": True,
}}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=True)
    audio_path = ydl.prepare_filename(info)
    print(f"Downloaded: {{info['title']}} ({{info.get('duration',0)}}s)")

{postprocessor}

# Transcribe
import whisper
model = whisper.load_model("{self.whisper_model}")
result = model.transcribe(audio_path, language="en")
print(f"Transcribed: {{len(result['segments'])}} segments")

# Create dataset
from datasets import Dataset
data = {{
    "text": [seg["text"].strip() for seg in result["segments"]],
    "start_time": [seg["start"] for seg in result["segments"]],
    "end_time": [seg["end"] for seg in result["segments"]],
    "video_url": [url] * len(result["segments"]),
}}
dataset = Dataset.from_dict(data)
dataset = dataset.filter(lambda x: len(x["text"]) > 10)
print(f"Dataset: {{len(dataset)}} samples")
dataset.save_to_disk("{self.output_dir}/dataset")
"""

    @staticmethod
    def clean_transcript_code() -> str:
        return """
import re

def clean_transcript(text: str) -> str:
    # Remove [Music], [Applause], etc.
    text = re.sub(r'\\[[^\\]]*\\]', '', text)
    # Fix capitalization
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text

dataset = dataset.map(lambda x: {"text": clean_transcript(x["text"])})
"""


# ==================================================================== #
#  12 — TwitterScraper
# ==================================================================== #

class TwitterScraper:
    """
    Scrape tweets using snscrape (no auth required).
    """

    def __init__(self, output_dir: str = "./twitter_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def scrape_code(self, query: str, limit: int = 100,
                    since: str = "2025-01-01", until: str = "2026-06-01",
                    lang: str = "en", min_faves: int = 10) -> str:
        return f"""
import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "{query} lang:{lang} min_faves:{min_faves} since:{since} until:{until}"
tweets = []
rate_limit_delay = 2  # seconds between requests

for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i >= {limit}:
        break
    tweets.append({{
        "id": tweet.id,
        "username": tweet.user.username,
        "text": tweet.rawContent,
        "like_count": tweet.likeCount,
        "retweet_count": tweet.retweetCount,
        "reply_count": tweet.replyCount,
        "timestamp": str(tweet.date),
        "url": tweet.url,
    }})
    if i % 10 == 0:
        import time; time.sleep(rate_limit_delay)

df = pd.DataFrame(tweets)

# Clean text
df["cleaned_text"] = df["text"].apply(lambda t: (
    t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
))
# Remove retweets
df = df[~df["cleaned_text"].str.startswith("RT @")]
# Remove URLs
df["cleaned_text"] = df["cleaned_text"].str.replace(
    r"http\\S+|www\\S+|https\\S+", "", case=False, regex=True)
# Remove mentions
df["cleaned_text"] = df["cleaned_text"].str.replace(r"@\\w+", "", regex=True)

print(f"Collected {{len(df)}} tweets")
df.to_parquet("{self.output_dir}/tweets.parquet")
print(f"Saved to {{self.output_dir}}/tweets.parquet")
"""


# ==================================================================== #
#  13 — PDFKnowledgeExtractor
# ==================================================================== #

class PDFKnowledgeExtractor:
    """
    Parse PDFs, extract text, chunk, deduplicate, and generate Q&A pairs.
    """

    def __init__(self, output_dir: str = "./pdf_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def parse_code(self, pdf_path: str, use_arxiv: bool = False,
                   arxiv_id: str = "") -> str:
        source_block = ""
        if use_arxiv and arxiv_id:
            source_block = f"""
import arxiv
search = arxiv.Search(id_list=["{arxiv_id}"])
paper = next(search.results())
paper.download_pdf(dirpath="{self.output_dir}")
pdf_path = os.path.join("{self.output_dir}", f"{{paper.get_short_id()}}.pdf")
print(f"Downloaded: {{paper.title}}")
"""
        return f"""
import os, json
{source_block}

# Parse PDF
import pypdf
reader = pypdf.PdfReader(pdf_path)
full_text = " ".join(page.extract_text() for page in reader.pages[:50])
print(f"Extracted {{len(full_text)}} chars from {{len(reader.pages)}} pages")

# Chunk with overlap
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, separators=["\\n\\n", "\\n", ".", " "],
)
chunks = splitter.split_text(full_text)
print(f"Created {{len(chunks)}} chunks")

# Save as Parquet
import pandas as pd
df = pd.DataFrame({{"text": chunks, "chunk_id": range(len(chunks))}})
df.to_parquet("{self.output_dir}/chunks.parquet")

# Generate summaries
from transformers import pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summaries = []
for i, chunk in enumerate(chunks[:5]):  # Limit to 5 for speed
    s = summarizer(chunk[:1024], max_length=80, min_length=20)[0]["summary_text"]
    summaries.append({{"chunk_id": i, "summary": s, "original": chunk[:100]}})
    print(f"  Summary {{i+1}}: {{s[:60]}}...")

with open("{self.output_dir}/summaries.json", "w") as f:
    json.dump(summaries, f, indent=2)
"""

    @staticmethod
    def dedup_code() -> str:
        return """
from datasketch import MinHash, MinHashLSH
import pandas as pd

df = pd.read_parquet("./pdf_data/chunks.parquet")
lsh = MinHashLSH(threshold=0.85, num_perm=128)
kept = []

for i, row in df.iterrows():
    m = MinHash(num_perm=128)
    for word in row["text"].split():
        m.update(word.encode())
    if not lsh.query(m):
        lsh.insert(f"doc_{i}", m)
        kept.append(i)

df_dedup = df.iloc[kept]
print(f"Dedup: {len(df)} -> {len(df_dedup)} chunks")
df_dedup.to_parquet("./pdf_data/chunks_dedup.parquet")
"""


# ==================================================================== #
#  14 — GitHubReadmeCollector
# ==================================================================== #

class GitHubReadmeCollector:
    """
    Search GitHub repos and collect READMEs via PyGithub.
    """

    def __init__(self, output_dir: str = "./github_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def collect_code(self, query: str, language: str = "python",
                     min_stars: int = 100, max_repos: int = 50) -> str:
        return f"""
import os, json
from github import Github

g = Github(os.environ.get("GITHUB_TOKEN", None))
query = "{query} language:{language} stars:>={min_stars}"
repos = g.search_repositories(query, sort="stars", order="desc")

results = []
count = 0
for repo in repos:
    if count >= {max_repos}:
        break
    try:
        readme = repo.get_readme().decoded_content.decode("utf-8")
        # Remove badges
        readme_clean = re.sub(r'!\\[.*?\\]\\(.*?\\)', '', readme)
        # Remove HTML comments
        readme_clean = re.sub(r'<!--.*?-->', '', readme_clean, flags=re.DOTALL)
        results.append({{
            "repo": repo.full_name,
            "description": repo.description,
            "topics": repo.get_topics(),
            "language": repo.language,
            "stars": repo.stargazers_count,
            "readme": readme_clean[:5000],
        }})
        count += 1
    except Exception as e:
        print(f"Could not fetch {{repo.full_name}}: {{e}}")

# Save as JSONL
with open("{self.output_dir}/readmes.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")

print(f"Collected {{len(results)}} repos")
"""


# ==================================================================== #
#  15 — WikipediaDumpStreamer
# ==================================================================== #

class WikipediaDumpStreamer:
    """
    Stream Wikipedia XML dump, extract articles, save as Parquet.
    """

    def __init__(self, output_dir: str = "./wikipedia_data",
                 batch_size: int = 10000):
        self.output_dir = output_dir
        self.batch_size = batch_size
        os.makedirs(output_dir, exist_ok=True)

    def stream_code(self, dump_path: str = "",
                    sample_ratio: float = 1.0,
                    resume_from: str = "") -> str:
        resume_block = ""
        if resume_from:
            resume_block = f"""
# Load checkpoint
import json
if os.path.exists("{self.output_dir}/checkpoint.json"):
    with open("{self.output_dir}/checkpoint.json") as f:
        cp = json.load(f)
    resume_id = cp["last_article_id"]
    print(f"Resuming from article {{resume_id}}")
else:
    resume_id = None
"""
        return f"""
import os, json, bz2
import xml.etree.ElementTree as ET
import pandas as pd

dump_path = "{dump_path}" or "enwiki-latest-pages-articles.xml.bz2"
{resume_block}

def strip_wiki_markup(text):
    import re
    text = re.sub(r'{{{{.*?}}}}', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'\\[\\[File:[^\\]]*\\]\\]', '', text)
    text = re.sub(r"'''", '', text)
    return text.strip()

batch = []
article_count = 0
with bz2.open(dump_path, "rt", encoding="utf-8") as f:
    for event, elem in ET.iterparse(f, events=("end",)):
        if elem.tag.endswith("}}page"):
            ns = elem.find(".{{http://www.mediawiki.org/xml/export-0.10/}}ns")
            title_elem = elem.find(".{{http://www.mediawiki.org/xml/export-0.10/}}title")
            text_elem = elem.find(".{{http://www.mediawiki.org/xml/export-0.10/}}revision/{{http://www.mediawiki.org/xml/export-0.10/}}text")

            if ns is not None and ns.text == "0" and title_elem is not None and text_elem is not None:
                article_id = elem.find(".{{http://www.mediawiki.org/xml/export-0.10/}}id")
                aid = article_id.text if article_id is not None else ""
                text = text_elem.text or ""

                if len(text) >= 500:
                    if resume_id and aid <= resume_id:
                        elem.clear(); continue
                    cleaned = strip_wiki_markup(text)
                    if len(cleaned) >= 500:
                        batch.append({{
                            "title": title_elem.text,
                            "text": cleaned[:10000],
                            "article_id": aid,
                        }})
                        article_count += 1

                        if len(batch) >= {self.batch_size}:
                            df = pd.DataFrame(batch)
                            df.to_parquet(f"{{self.output_dir}}/articles_{{article_count//{self.batch_size}}}.parquet")
                            with open("{self.output_dir}/checkpoint.json", "w") as cp:
                                json.dump({{"last_article_id": aid}}, cp)
                            batch = []
                            print(f"Processed {{article_count}} articles")

            elem.clear()

if batch:
    df = pd.DataFrame(batch)
    df.to_parquet(f"{{self.output_dir}}/articles_final.parquet")

print(f"Total articles extracted: {{article_count}}")
"""


# ==================================================================== #
#  16 — SlackExportParser
# ==================================================================== #

class SlackExportParser:
    """
    Parse Slack export ZIP files into a HuggingFace Dataset.
    """

    def __init__(self, output_dir: str = "./slack_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def parse_code(self, zip_path: str) -> str:
        return f"""
import zipfile, json, os, re
from datasets import Dataset
import pandas as pd

zip_path = "{zip_path}"
output_dir = "{self.output_dir}"
os.makedirs(output_dir, exist_ok=True)

records = []
with zipfile.ZipFile(zip_path, "r") as z:
    for name in z.namelist():
        if name.endswith(".json") and "/" in name:
            channel = name.split("/")[0]
            data = json.loads(z.read(name))
            if not isinstance(data, list):
                continue
            for msg in data:
                text = msg.get("text", "")
                if msg.get("subtype") in ("bot_message", "message_deleted"):
                    continue
                # Redact emails
                text = re.sub(r"[\\w\\.-]+@[\\w\\.-]+\\.\\w+", "[EMAIL]", text)
                # Redact phone
                text = re.sub(r"\\b\\d{{3}}[-.]?\\d{{3}}[-.]?\\d{{4}}\\b", "[PHONE]", text)
                # Redact user IDs
                text = re.sub(r"<@[A-Z0-9]+>", "[USER]", text)

                record = {{
                    "channel": channel,
                    "user": msg.get("user", ""),
                    "text": text,
                    "timestamp": msg.get("ts", ""),
                    "thread_ts": msg.get("thread_ts", ""),
                }}
                # Collect thread replies
                if "thread_ts" in msg and "replies" in msg:
                    record["replies"] = json.dumps(msg["replies"])
                records.append(record)

df = pd.DataFrame(records)
print(f"Parsed {{len(df)}} messages from {{df['channel'].nunique()}} channels")
df.to_parquet(os.path.join(output_dir, "slack_messages.parquet"))
"""


# ==================================================================== #
#  17 — ArxivPaperFetcher
# ==================================================================== #

class ArxivPaperFetcher:
    """
    Search ArXiv, download PDFs, extract summaries, generate Q&A.
    """

    def __init__(self, output_dir: str = "./arxiv_papers"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def fetch_code(self, query: str = "cat:cs.LG",
                   max_results: int = 50,
                   sort_by: str = "SubmittedDate") -> str:
        return f"""
import arxiv, os, json
from transformers import pipeline

output_dir = "{self.output_dir}"
os.makedirs(output_dir, exist_ok=True)

search = arxiv.Search(
    query="{query}",
    max_results={max_results},
    sort_by=arxiv.SortCriterion.{sort_by},
)

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
papers = []

for paper in search.results():
    # Download PDF
    pdf_path = os.path.join(output_dir, f"{{paper.get_short_id()}}.pdf")
    paper.download_pdf(dirpath=output_dir)

    # Extract abstract + conclusion heuristic
    abstract = paper.summary.replace("\\n", " ")
    metadata = {{
        "arxiv_id": paper.get_short_id(),
        "title": paper.title,
        "authors": [a.name for a in paper.authors],
        "abstract": abstract[:1000],
        "categories": list(paper.categories),
        "published": str(paper.published),
        "pdf_path": pdf_path,
        "url": paper.entry_id,
    }}

    # Generate summary
    try:
        summary = summarizer(abstract[:1024], max_length=100, min_length=30)[0]["summary_text"]
        metadata["summary"] = summary
    except Exception:
        metadata["summary"] = abstract[:200]

    # Generate Q&A
    qa_prompt = f"Based on this abstract: {{abstract[:500]}}\\nWhat problem does this paper solve?"
    metadata["qa_prompt"] = qa_prompt

    papers.append(metadata)
    print(f"[{{len(papers)}}/{{max_results}}] {{paper.title[:60]}}...")

with open(os.path.join(output_dir, "papers.json"), "w") as f:
    json.dump(papers, f, indent=2, default=str)

print(f"Saved {{len(papers)}} papers to {{output_dir}}")
"""


# ==================================================================== #
#  18 — RedditAPICollector
# ==================================================================== #

class RedditAPICollector:
    """
    Collect Reddit posts + comments using PRAW.
    """

    def __init__(self, output_dir: str = "./reddit_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def collect_code(self, subreddit: str = "machinelearning",
                     query: str = "",
                     limit: int = 100,
                     time_filter: str = "month") -> str:
        return f"""
import praw, os, json

reddit = praw.Reddit(
    client_id=os.environ.get("REDDIT_CLIENT_ID", ""),
    client_secret=os.environ.get("REDDIT_CLIENT_SECRET", ""),
    user_agent="MyColabAgent/1.0",
)

sub = reddit.subreddit("{subreddit}")
query = "{query}"
limit = {limit}

posts = []
if query:
    submissions = sub.search(query, limit=limit, sort="top", time_filter="{time_filter}")
else:
    submissions = sub.top(time_filter="{time_filter}", limit=limit)

for submission in submissions:
    submission.comments.replace_more(limit=0)
    top_comments = []
    for comment in submission.comments[:10]:
        if hasattr(comment, "body"):
            top_comments.append({{
                "body": comment.body[:500],
                "score": comment.score,
            }})

    posts.append({{
        "id": submission.id,
        "title": submission.title,
        "selftext": (submission.selftext or "")[:1000],
        "score": submission.score,
        "num_comments": submission.num_comments,
        "created": str(submission.created_utc),
        "url": submission.url,
        "top_comments": top_comments,
    }})

print(f"Collected {{len(posts)}} posts")
with open("{self.output_dir}/reddit_posts.json", "w") as f:
    json.dump(posts, f, indent=2, default=str)
"""


# ==================================================================== #
#  19 — CommonCrawlExtractor
# ==================================================================== #

class CommonCrawlExtractor:
    """
    Download and extract text from CommonCrawl WARC files.
    """

    def __init__(self, output_dir: str = "./commoncrawl_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_code(self, max_pages: int = 1000,
                     crawl_id: str = "CC-MAIN-2025-18") -> str:
        return f"""
import os, json, gzip
from warcio import ArchiveIterator
import trafilatura
import langdetect
import pandas as pd

output_dir = "{self.output_dir}"
max_pages = {max_pages}
crawl_id = "{crawl_id}"

# Get WARC paths
import requests
index_url = f"https://data.commoncrawl.org/crawl-data/{{crawl_id}}/warc.paths.gz"
resp = requests.get(index_url)
paths = gzip.decompress(resp.content).decode().strip().split("\\n")
print(f"Found {{len(paths)}} WARC files")

# Use first WARC file
warc_url = f"https://data.commoncrawl.org/{{paths[0]}}"
print(f"Streaming: {{warc_url}}")

records = []
count = 0
resp = requests.get(warc_url, stream=True)
for record in ArchiveIterator(resp.raw):
    if record.rec_type == "response" and record.http_headers:
        url = record.rec_headers.get_header("WARC-Target-URI", "")
        html = record.content_stream().read().decode("utf-8", errors="ignore")

        # Extract text
        text = trafilatura.extract(html)
        if not text or len(text) < 500:
            continue

        # Language detection
        try:
            lang = langdetect.detect(text)
        except Exception:
            lang = "unknown"
        if lang != "en":
            continue

        records.append({{"url": url, "text": text[:5000], "lang": lang}})
        count += 1

        if count % 100 == 0:
            print(f"  Extracted {{count}} pages")

        if count >= max_pages:
            break

df = pd.DataFrame(records)
df.to_parquet(os.path.join(output_dir, "commoncrawl_sample.parquet"))
print(f"Saved {{len(df)}} pages to {{output_dir}}")
"""


# ==================================================================== #
#  20 — SeleniumDynamicScraper
# ==================================================================== #

class SeleniumDynamicScraper:
    """
    Scrape JavaScript-rendered pages using Selenium.
    Limited use in Colab due to resource constraints.
    """

    def __init__(self, output_dir: str = "./selenium_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def scrape_code(self, url: str = "https://example.com",
                    max_pages: int = 5,
                    scroll: bool = False) -> str:
        scroll_js = ""
        if scroll:
            scroll_js = """
# Scroll to load infinite content
import time
last_height = driver.execute_script("return document.body.scrollHeight")
for _ in range(3):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height
"""
        return f"""
import os, json, time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# Setup headless Chrome
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1920x1080")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options,
)

url = "{url}"
driver.get(url)

# Wait for body
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.TAG_NAME, "body"))
)

{scroll_js}

# Extract page content
from bs4 import BeautifulSoup
soup = BeautifulSoup(driver.page_source, "lxml")

# Remove nav, footer, ads
for tag in soup.find_all(["nav", "footer", "script", "style", "noscript"]):
    tag.decompose()

text = soup.get_text(separator="\\n", strip=True)
print(f"Extracted {{len(text)}} chars")

# Save
output = {{
    "url": url,
    "title": driver.title,
    "text": text[:10000],
    "timestamp": time.time(),
}}
with open("{self.output_dir}/page.json", "w") as f:
    json.dump(output, f, indent=2)

driver.quit()
print(f"Saved to {{self.output_dir}}/page.json")
"""
