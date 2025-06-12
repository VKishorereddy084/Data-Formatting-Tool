import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

# -----------------------------
# Prompt Templates
# -----------------------------
SUMMARY_PROMPT_TEMPLATE = """
Task: You are an AI assistant tasked with summarizing the following text in a clear, concise, and formal academic tone.

Instructions:

1. Review the provided input:

<text>
{text}
</text>

2. Your task:

- Identify the most important facts, arguments, and conclusions.
- Write a well-structured summary covering the key ideas.
- Keep it factual and objective.
- Length: 1-7 paragraphs depending on input size.

Summary:
"""

REFINE_PROMPT_TEMPLATE = """
You are refining an academic summary.

Current Summary:
<summary>
{summary}
</summary>

New Text:
<text>
{text}
</text>

Instructions:
- Integrate new information from the text into the existing summary.
- Maintain a clear, formal academic tone.
- Avoid redundancy.
- Keep the updated summary concise and coherent.

Refined Summary:
"""

# -----------------------------
# Config
# -----------------------------
SUMMARIZATION_MODE = "map_reduce"  # or "refine"


# -----------------------------
# Core Functions
# -----------------------------
def load_markdown(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def split_into_chapters(md: str) -> Dict[str, str]:
    lines = md.split('\n')
    chapters: Dict[str, str] = {}
    current_title = None
    current_content: List[str] = []

    header_pattern = re.compile(r'^(#{1,6})\s*Chapter\s+(\d+)(?::\s*(.*))?', re.IGNORECASE)

    for line in lines:
        m = header_pattern.match(line)
        if m:
            if current_title:
                chapters[current_title] = "\n".join(current_content).strip()
                current_content = []
            number = m.group(2)
            subtitle = m.group(3) or ''
            current_title = f"Chapter {number}" + (f": {subtitle}" if subtitle else "")
        elif current_title:
            current_content.append(line)

    if current_title and current_content:
        chapters[current_title] = "\n".join(current_content).strip()

    return chapters or {"Full Document": md}


def chunk_chapter(title: str, text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


def summarize_chunk(client: OpenAI, chunk: str, model_name: str) -> str:
    prompt = SUMMARY_PROMPT_TEMPLATE.replace("{text}", chunk)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def refine_summary(client: OpenAI, summary: str, chunk: str, model_name: str) -> str:
    prompt = REFINE_PROMPT_TEMPLATE.replace("{summary}", summary).replace("{text}", chunk)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def summarize_with_refine(client: OpenAI, chunks: List[str], model_name: str) -> str:
    summary = summarize_chunk(client, chunks[0], model_name)
    for chunk in chunks[1:]:
        summary = refine_summary(client, summary, chunk, model_name)
    return summary


def combine_summaries_map_reduce(client: OpenAI, summaries: List[str], model_name: str) -> str:
    combined_input = "\n\n".join(summaries)
    prompt = f"""
You are an academic assistant. Combine the following chunk-level summaries into one coherent summary for an entire chapter or document.
Avoid repetition and ensure a logical flow.

Summaries:
{combined_input}

Final Chapter Summary:
"""
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def process_chapter(
    client: OpenAI,
    title: str,
    text: str,
    model_name: str
) -> Tuple[str, Tuple[str, str]]:
    print(f"\n📚 Summarizing {title} using '{SUMMARIZATION_MODE}' mode...")
    chunks = chunk_chapter(title, text)

    if SUMMARIZATION_MODE == "map_reduce":
        chunk_summaries = [summarize_chunk(client, chunk, model_name) for chunk in chunks]
        final_summary = combine_summaries_map_reduce(client, chunk_summaries, model_name)
    elif SUMMARIZATION_MODE == "refine":
        final_summary = summarize_with_refine(client, chunks, model_name)
    else:
        raise ValueError("Invalid summarization mode. Choose 'map_reduce' or 'refine'.")

    md_section = f"## {title} Summary\n\n{final_summary}\n"
    return md_section, (title, final_summary)


def chunked_items(d: Dict[str, str], n: int) -> List[List[Tuple[str, str]]]:
    items = list(d.items())
    return [items[i:i + n] for i in range(0, len(items), n)]


# -----------------------------
# Main Flask-compatible function
# -----------------------------
def generate_summary(md_path: str, output_dir: str, api_key: str, model_name: str) -> str:
    client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=api_key)
    md_text = load_markdown(md_path)
    chapters = split_into_chapters(md_text)

    output_md = "# 📘 Chapter Summaries\n\n"
    output_rows: List[Tuple[str, str]] = []

    BATCH_SIZE = 5
    for batch_idx, batch in enumerate(chunked_items(chapters, BATCH_SIZE), start=1):
        titles = [t for t, _ in batch]
        print(f"\n⏳ Processing batch {batch_idx}: {titles}")

        for title, content in batch:
            sec_md, sec_row = process_chapter(client, title, content, model_name)
            output_md += sec_md + "\n"
            output_rows.append(sec_row)

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / (Path(md_path).stem + "_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_md)

    print(f" Summary saved to {out_path}")
    return str(out_path)
