from flask import Flask, request, render_template, send_file, redirect, url_for, session
from pathlib import Path
import os
import time
import logging
import asyncio
import atexit

from apscheduler.schedulers.background import BackgroundScheduler

from converters.convert_pdf import convert_pdf_to_markdown, clean_markdown
from converters.crawl_url import get_discovered_urls, run_crawl_on_selected_urls
from converters.qa_generator import generate_qa_from_markdown
from converters.summarizer import generate_summary

# ─────────────────────────────────────────────────────────────
# ⚙️ Configuration
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = Path("static/converted_files")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

latest_files = []

# ─────────────────────────────────────────────────────────────
# 🔁 Background Cleanup Job – Deletes old files every 10 minutes
# ─────────────────────────────────────────────────────────────
def cleanup_old_files(folder: Path, age_minutes: int = 30):
    now = time.time()
    cutoff = now - age_minutes * 60
    for file in folder.glob("*"):
        if file.is_file() and file.stat().st_mtime < cutoff:
            print(f"🧹 Deleting old file: {file.name}")
            file.unlink()

scheduler = BackgroundScheduler()
scheduler.add_job(lambda: cleanup_old_files(UPLOAD_FOLDER, age_minutes=30), 'interval', minutes=10)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# ─────────────────────────────────────────────────────────────
# 🧠 Flask Routes
# ─────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    global latest_files
    _log.info("📥 Request received at /")
    latest_files = []

    if request.method == 'POST':
        pdf = request.files.get('pdf')
        md_upload = request.files.get('markdown_file')
        summary_upload = request.files.get('summary_file')
        url = request.form.get('url')
        source = request.form.get('source')
        crawl_mode = request.form.get('crawl_mode', 'yes')
        apply_preprocess = request.form.get('pdf_preprocess', 'yes') == 'yes'
        selected_model = request.form.get('llm_model', 'meta-llama/Llama-4-Scout-17B-16E-Instruct')

        # 🟡 Summarization
        if source == "summarize" and summary_upload and summary_upload.filename.endswith('.md'):
            summary_path = UPLOAD_FOLDER / summary_upload.filename
            summary_upload.save(summary_path)

            try:
                with open("my_key") as f:
                    api_key = f.read().strip()
            except FileNotFoundError:
                return "❌ API key file 'my_key' not found.", 500

            summary_md_path = generate_summary(str(summary_path), str(UPLOAD_FOLDER), api_key, selected_model)

            with open(summary_md_path, "r", encoding="utf-8") as f:
                content = f.read()

            latest_files.append({
                "url": None,
                "raw": summary_md_path,
                "filtered": summary_md_path,
                "qa": False,
                "summary": True
            })

            return render_template('index.html', converted=True, markdown_content=content, latest_files=latest_files)

        # 🟢 Q&A Pair Generation
        if source == "qa" and md_upload and md_upload.filename.endswith('.md'):
            md_path = UPLOAD_FOLDER / md_upload.filename
            md_upload.save(md_path)

            try:
                with open("my_key") as f:
                    api_key = f.read().strip()
            except FileNotFoundError:
                return "❌ API key file 'my_key' not found.", 500

            qa_md_path = generate_qa_from_markdown(str(md_path), str(UPLOAD_FOLDER), api_key, selected_model)

            with open(qa_md_path, "r", encoding="utf-8") as f:
                content = f.read()

            latest_files.append({
                "url": None,
                "raw": qa_md_path,
                "filtered": qa_md_path,
                "qa": True,
                "summary": False
            })

            return render_template('index.html', converted=True, markdown_content=content, latest_files=latest_files)

        # 🔵 PDF to Markdown
        elif source == "pdf" and pdf and pdf.filename != '':
            pdf_path = UPLOAD_FOLDER / pdf.filename
            pdf.save(str(pdf_path))

            markdown_file = convert_pdf_to_markdown(pdf_path, UPLOAD_FOLDER)

            if apply_preprocess:
                cleaned = clean_markdown(markdown_file)
                with open(cleaned, 'r', encoding='utf-8') as f:
                    content = f.read()
                latest_files.append({
                    "url": None,
                    "raw": markdown_file,
                    "filtered": cleaned,
                    "qa": False,
                    "summary": False,
                    "source": "pdf",
                    "preprocess": "yes"
                })
            else:
                with open(markdown_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                latest_files.append({
                    "url": None,
                    "raw": markdown_file,
                    "filtered": markdown_file,
                    "qa": False,
                    "summary": False,
                    "source": "pdf",
                    "preprocess": "no"
                })

            return render_template('index.html', converted=True, markdown_content=content, latest_files=latest_files)

        
        # 🔗 URL to Markdown
        elif url:
            if crawl_mode == 'yes':
                # Discover internal links
                discovered = get_discovered_urls(url)
                session['base_url'] = url
                session['discovered'] = discovered
                return render_template('select_urls.html', urls=discovered)
            else:
                # Just convert the base URL
                results = run_crawl_on_selected_urls([url], UPLOAD_FOLDER)
                latest_files.clear()
                latest_files.extend(results)

                preview = ""
                for res in results:
                    with open(res["filtered"], 'r', encoding='utf-8') as f:
                        preview += f"# {res['url']}\n\n" + f.read() + "\n\n"

                return render_template('index.html', converted=True, markdown_content=preview, latest_files=latest_files)

    return render_template('index.html', converted=False)



@app.route('/process_selected', methods=['POST'])
def process_selected():
    global latest_files
    latest_files = []

    # Grab the URLs directly (no more int conversions!)
    selected_urls = request.form.getlist('selected_urls')
    if not selected_urls:
        return redirect(url_for('index'))

    # Crawl each one
    results = run_crawl_on_selected_urls(selected_urls, UPLOAD_FOLDER)
    latest_files = results

    # Build the preview HTML just like you did before
    preview = ""
    for res in results:
        with open(res["filtered"], 'r', encoding='utf-8') as f:
            preview += f"# {res['url']}\n\n" + f.read() + "\n\n"

    return render_template('index.html', converted=True, markdown_content=preview, latest_files=latest_files)


@app.route('/download/<version>/<int:index>')
def download_file(version, index):
    global latest_files
    if 0 <= index < len(latest_files):
        file_path = latest_files[index].get(version)
        if file_path and Path(file_path).exists():
            filename = os.path.basename(file_path)
            return send_file(file_path, as_attachment=True, download_name=filename)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
