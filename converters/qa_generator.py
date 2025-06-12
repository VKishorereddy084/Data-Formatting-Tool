import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

# -----------------------------
# Constants & Prompt
# -----------------------------

QUESTION_PROMPT_TEMPLATE = """
Task: You are an AI assistant tasked with generating only the top 10 most important questions from the following chapter. Do not include answers.

Instructions:

1. Review the provided input:

<chapter_text>
{text}
</chapter_text>

2. Your task:

- Read the chapter text carefully.
- Identify and extract the most important facts or concepts.
- Do NOT include answers.
- For each, generate a concise and natural-sounding question that can be fully answered using the chapter text.
- Focus on main ideas—not minor details or trivia.
- Do not include more than 10 questions.
- Do not generate any question if the information is unclear or not directly supported in the text.

3. Guidelines:

- Questions should be fact-based, clear, and context-specific.
- Avoid generic or overly broad questions.
- Do not reference “this text” or “the paragraph” or “the chapter.”
- Keep questions self-contained.

Text:
------------
{text}
------------
Questions:
"""

# -----------------------------
# Helper Functions
# -----------------------------
def load_markdown(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def split_into_chapters(md: str) -> Dict[str, str]:
    lines = md.split('\n')
    chapters = {}
    current_title = None
    current_content = []

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

def build_vector_store(chapters: Dict[str, str]) -> Chroma:
    docs = [
        Document(page_content=text, metadata={"chapter": title})
        for title, text in chapters.items()
    ]
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return Chroma.from_documents(docs, embeddings)

def get_chapter_retriever(chapter_title: str, vector_store: Chroma):
    return vector_store.as_retriever(
        search_kwargs={"filter": {"chapter": chapter_title}, "k": 2}
    )

def generate_questions(client: OpenAI, text: str, model_name: str) -> List[str]:
    prompt = QUESTION_PROMPT_TEMPLATE.replace("{text}", text)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content or ""

    questions = []
    for line in raw.strip().split("\n"):
        if re.match(r"^\s*(\d+\.)?\s*(What|Why|How|When|Where|Who)\b", line.strip(), re.IGNORECASE):
            cleaned = re.sub(r"^\s*\d+\.\s*", "", line).strip()
            questions.append(cleaned)

    return questions


def generate_answer(client: OpenAI, question: str, retriever, model_name: str) -> str:
    docs = retriever.get_relevant_documents(question)
    context = "\n".join(doc.page_content for doc in docs)
    prompt = f"Answer the following question based only on the text below:\n\nText:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def process_chapter(
    client: OpenAI,
    title: str,
    text: str,
    vector_store: Chroma
) -> Tuple[str, List[Tuple[str, str, str]]]:
    print(f" Processing {title}...")
    md_section = f"## {title}\n\n"
    rows = []

    questions = generate_questions(client, text)
    if not questions:
        return "", []

    retriever = get_chapter_retriever(title, vector_store)

    for i, q in enumerate(questions, 1):
        try:
            ans = generate_answer(client, q, retriever)
        except Exception as e:
            ans = f"[Error] {e}"
        md_section += f"**Q{i}:** {q}\n\n**A{i}:** {ans}\n\n"
        rows.append((title, q, ans))

    return md_section, rows

def chunked_items(d: Dict[str, str], n: int) -> List[List[Tuple[str, str]]]:
    items = list(d.items())
    return [items[i:i + n] for i in range(0, len(items), n)]

# -----------------------------
# Main Q&A Pipeline Function
# -----------------------------
def generate_qa_from_markdown(md_path: str, output_dir: str, api_key: str, model_name: str) -> str:
    client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=api_key)
    md_text = load_markdown(md_path)
    chapters = split_into_chapters(md_text)

    if not chapters:
        chapters = {"Full Document": md_text}

    output_md = "#  Q&A Pairs\n\n"
    for title, content in chapters.items():
        questions = generate_questions(client, content, model_name)
        if not questions:
            continue

        retriever = get_chapter_retriever(title, build_vector_store({title: content}))

        output_md += f"## {title}\n\n"
        for i, q in enumerate(questions, 1):
            try:
                ans = generate_answer(client, q, retriever, model_name)
            except Exception as e:
                ans = f"[Error] {e}"
            output_md += f"**Q{i}:** {q}\n\n**A{i}:** {ans}\n\n"

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / (Path(md_path).stem + "_qa.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_md)

    return str(out_path)
