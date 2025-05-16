import json
import logging
import time
import pymupdf as fitz
import pandas as pd
import argparse
from pathlib import Path
from transformers import pipeline
from langdetect import detect
import re
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import FigureElement, InputFormat, Table
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.document import ConversionResult
from docling.models.tesseract_ocr_model import TesseractOcrOptions
from huggingface_hub import snapshot_download


_log = logging.getLogger(__name__)


def convert_pdf_to_markdown(input_pdf_path: Path, output_dir: Path) -> Path:
    """ Convert PDF to Markdown using Docling """
    try:
        _log.info(f"🔁 Starting PDF conversion: {input_pdf_path}")

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options.lang = ["en", "de"]
        pipeline_options.do_code_enrichment = True
        pipeline_options.do_formula_enrichment = True
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.CUDA
    )

        doc_converter = DocumentConverter(
          format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

        start_time = time.time()
        conv_result = doc_converter.convert(input_pdf_path)
        if not conv_result or not conv_result.document:
            _log.error("❌ Conversion returned empty result or failed silently.")
            raise ValueError("Empty conversion result for PDF.")
        logging.info(f"Document converted in {time.time() - start_time:.2f} seconds.")

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = input_pdf_path.stem
        markdown_path = output_dir / f"{stem}.md"
        markdown_path.write_text(conv_result.document.export_to_markdown(), encoding="utf-8")
        return markdown_path
    
    except Exception as e:
        _log.exception("❌ PDF conversion failed")
        raise



UNWANTED = {
    "en": {
        "Table of Contents": ["Table of Contents", "Contents", "Index"],
        "Preface": ["Preface"],
        "Acknowledgments": ["Acknowledgments", "Acknowledgements", "Credits", "Funding"],
        "Copyright": ["Copyright", "Legal Notice", "Printing History"],
        "Appendix": ["Appendix", "Supplementary Material"],
        "Glossary": ["Definitions", "Glossary", "Terminology"],
        "List of Tables": ["List of Tables"],
        "List of Figures": ["List of Figures"],
        "Nomenclature": ["Nomenclature", "Symbols Used"],
        "References": ["References", "Bibliography", "Sources", "Citations", "Works Cited", "Further Reading"],
    },
    "de": {
        "Inhaltsverzeichnis": ["Inhaltsverzeichnis", "Index"],
        "Vorwort": ["Vorwort", "Einleitung", "Über dieses Buch"],
        "Danksagung": ["Danksagung", "Credits", "Funding"],
        "Copyright": ["Copyright", "Druckvermerk", "Rechtlicher Hinweis"],
        "Anhang": ["Anhang", "Supplement"],
        "Glossar": ["Begriffe", "Glossar", "Terminologie"],
        "Tabellenverzeichnis": ["Tabellenverzeichnis"],
        "Abbildungsverzeichnis": ["Abbildungsverzeichnis"],
        "Literaturverzeichnis": ["Literaturverzeichnis", "Quellen", "Zitate"],
    },
}

# Sections we never remove
# … keep all your imports and convert_pdf_to_markdown() the same …

# Add "Kapitel" to the list of protected German sections
PROTECTED = {
    "en": ["Abstract", "Introduction", "Discussion", "Results", "Conclusion"],
    "de": ["Abstract", "Einleitung", "Diskussion", "Ergebnisse", "Fazit", "Kapitel"],
}

# Broaden the chapter-start anchor to any "Kapitel <number>"
CHAPTER_START_PATTERNS = {
    "en": r"^(##\s*(Abstract|Chapter\s+1))",
    "de": r"^(##\s*(Abstract|Einleitung|Kapitel\s+\d+))",
}



def clean_markdown(md_path: Path) -> Path:
    try:
        _log.info(f"🧼 Cleaning markdown file: {md_path}")
        text = md_path.read_text(encoding="utf-8")

    # 1) detect language
        sample = "\n".join([h for _, h in re.findall(r"(##+)\s+(.+)", text)][:5])
        _log.info(f"🧾 Heading sample for detection: {sample}")
        lang = "de" if sample.lower().startswith("kapitel") or sample.startswith("Einleitung") or detect(sample).startswith("de") else "en"
        _log.info(f"🌍 Detected language: {lang}")

    # 2) chapter‐start anchoring (now catches "Kapitel 1", "Kapitel 2", etc.)
        start_re = re.search(CHAPTER_START_PATTERNS[lang], text, flags=re.MULTILINE)
        if start_re:
            _log.info(f"🔗 Found chapter start pattern at: {start_re.start()}")
            text = text[start_re.start():]

        else:
            _log.warning("⚠️ No chapter anchor found in markdown.")

    # 3) load NLI classifier once
        _log.info(f"🤖 Loading zero-shot model for lang: {lang}")
        if lang == "en":
            classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        else:
            classifier = pipeline("zero-shot-classification", model="svalabs/gbert-large-zeroshot-nli")

        _log.info("✅ Classifier loaded successfully.")


    # flatten unwanted-section labels
        labels = [lbl for variants in UNWANTED[lang].values() for lbl in variants] + ["Body Text"]
        threshold = 0.30

    # 4) iterate **all** headings (no early return)
        for level, heading in re.findall(r"(##+)\s+(.+)", text):
        # skip any fully protected headings (including Kapitel X)
            if any(heading.strip().startswith(p) for p in PROTECTED[lang]):
                continue

        # also skip any German "Kapitel <numb er>" explicitly
            if lang == "de" and re.match(r"Kapitel\s+\d+", heading):
                continue

        # classify!
            res = classifier(heading, candidate_labels=labels)
            top_label, top_score = res["labels"][0], res["scores"][0]
            if top_score > threshold and top_label in sum(UNWANTED[lang].values(), []):
            # remove section up to next heading
                 pattern = rf"{re.escape(level)}\s*{re.escape(heading)}.*?(?=\n## |\Z)"
                 text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # 5) final regex-nuke pass for stragglers
            final_regex = "|".join(rf"##\s*{re.escape(v)}"
                            for variants in UNWANTED[lang].values()
                            for v in variants)
            text = re.sub(rf"({final_regex}).*?(?=\n##|\Z)", "", text, flags=re.IGNORECASE | re.DOTALL)

    # write out cleaned file
        out_path = md_path.parent / f"cleaned_{md_path.stem}.md"
        out_path.write_text(text.strip(), encoding="utf-8")
        logging.info(f"✅ Cleaned Markdown at {out_path}")
        return out_path

    except Exception as e:
        _log.exception("❌ Markdown cleaning failed")
        raise

