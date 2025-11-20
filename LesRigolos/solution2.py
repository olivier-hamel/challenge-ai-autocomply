# LesRigolos/solution2.py
import fitz
import requests
import json
import re
import base64
import time
from difflib import get_close_matches
from pathlib import Path
from typing import List, Dict, Tuple, Optional


API_URL = "https://ai-models.autocomply.ca"
API_KEY = "sk-ac-7f8e9d2c4b1a6e5f3d8c7b9a2e4f6d1c"
MODEL = "gemini-2.5-flash"  # used for /ask
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

SECTIONS = [
    "Articles & Amendments",
    "By Laws",
    "Unanimous Shareholder Agreement",
    "Minutes & Resolutions",
    "Directors Register",
    "Officers Register",
    "Shareholder Register",
    "Securities Register",
    "Share Certificates",
    "Ultimate Beneficial Owner Register"
]


SAMPLE_RATE = 3  
BIN_SEARCH_WINDOW = 20  
MAX_CONTEXT_PAGES = 5

def safe_json_extract(raw: str) -> dict:
    match = re.search(r"\{.*?\}", raw, flags=re.S)
    if not match:
        return {"section": "Unknown", "position": "unknown", "confidence": 0.0}
    try:
        j = json.loads(match.group())
        return {
            "section": j.get("section", "Unknown"),
            "position": j.get("position", "unknown"),
            "confidence": float(j.get("confidence", 0.0))
        }
    except Exception:
        return {"section": "Unknown", "position": "unknown", "confidence": 0.0}

def map_to_official(name: str) -> str:

    if name in SECTIONS:
        return name
    candidates = get_close_matches(name, SECTIONS, n=1, cutoff=0.6)
    if candidates:
        return candidates[0]
    name_low = name.lower()
    for s in SECTIONS:
        if all(word.lower() in name_low for word in s.split()[:2]):
            return s
    return "Unknown"

def extract_texts(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        txt = page.get_text("text").strip()
        pages.append(txt)
    doc.close()
    return pages

def ocr_page_image_b64(doc: fitz.Document, page_index: int) -> Optional[str]:
    try:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        return base64.b64encode(pix.tobytes("png")).decode("utf-8")
    except Exception:
        return None


def ask_text_classification(text_context: str) -> dict:
    """Call /ask with a strict prompt and return parsed dict."""
    prompt = f"""
You are a classifier for MINUTE BOOK SECTIONS. Given the TEXT of a page (or a small window
of consecutive pages), classify the content into exactly ONE of the following 10 section names:

1. Articles & Amendments
2. By Laws
3. Unanimous Shareholder Agreement
4. Minutes & Resolutions
5. Directors Register
6. Officers Register
7. Shareholder Register
8. Securities Register
9. Share Certificates
10. Ultimate Beneficial Owner Register

Return ONLY a JSON object EXACTLY in this format (no extra commentary):

{{ "section": "<one of the 10 EXACT names above>", "position": "start|middle|end|unknown", "confidence": 0.0-1.0 }}

If you are unsure, choose "Unknown" as section with low confidence.
Text to classify:
\"\"\"{text_context[:2000]}\"\"\"
"""
    payload = {"query": prompt, "model": MODEL}
    try:
        r = requests.post(f"{API_URL}/ask", json=payload, headers=HEADERS, timeout=10)
        r.raise_for_status()
        raw = r.json().get("result", "")
    except Exception as e:
        print("API call failed:", e)
        raw = ""
    parsed = safe_json_extract(raw)
    parsed["section"] = map_to_official(parsed["section"])
    return parsed


def sampling_pass(page_texts: List[str], rate: int = SAMPLE_RATE) -> Dict[int, dict]:
    total = len(page_texts)
    sampled = {}
    print(f"Sampling every {rate} pages (total pages {total})")
    for p in range(0, total, rate):
        start = max(0, p - (MAX_CONTEXT_PAGES // 2))
        end = min(total, start + MAX_CONTEXT_PAGES)
        start = max(0, end - MAX_CONTEXT_PAGES)
        context = "\n\n---PAGE_BREAK---\n\n".join(page_texts[start:end])
        if len(context.strip()) < 20:
            sampled[p] = {"section": "Unknown", "position": "unknown", "confidence": 0.0}
            print(f"  page {p}: local empty -> Unknown")
            continue
        res = ask_text_classification(context)
        sampled[p] = res
        print(f"  page {p}: -> {res['section']} ({res['confidence']})")
        time.sleep(0.08)
    return sampled

def build_rough_ranges(sampled: Dict[int, dict], total_pages: int) -> List[Tuple[str, int, int]]:
    """
    From sampled pages, create approximate contiguous ranges (inclusive indices)
    Returns list of (section_name, approx_start_page_index, approx_end_page_index)
    """
    items = sorted(sampled.items())
    ranges = []
    cur_section = None
    cur_start = None
    cur_end = None
    for p, info in items:
        sec = info["section"]
        if sec == "Unknown":
            if cur_section is not None:
                ranges.append((cur_section, cur_start, cur_end))
                cur_section = None
            continue
        if cur_section is None:
            cur_section = sec
            cur_start = p
            cur_end = p
        elif sec == cur_section:
            cur_end = p
        else:
            ranges.append((cur_section, cur_start, cur_end))
            cur_section = sec
            cur_start = p
            cur_end = p
    if cur_section is not None:
        ranges.append((cur_section, cur_start, cur_end))
    cleaned = []
    for sec, s, e in ranges:
        s = max(0, s)
        e = min(total_pages - 1, e)
        cleaned.append((sec, s, e))
    print("Rough ranges from sampling:", cleaned)
    return cleaned


def classify_single_page(page_texts: List[str], page_idx: int) -> dict:
    total = len(page_texts)
    start = max(0, page_idx - 1)
    end = min(total, page_idx + 2)
    context = "\n\n---PAGE_BREAK---\n\n".join(page_texts[start:end])
    return ask_text_classification(context)

def binary_refine(page_texts: List[str], low: int, high: int, target_section: str) -> int:
    
    lo = low
    hi = high
    last_positive = None
    while lo <= hi:
        mid = (lo + hi) // 2
        info = classify_single_page(page_texts, mid)
        sec = info["section"]
        print(f"   checking page {mid} -> {sec} ({info['confidence']})")
        if sec == target_section:
            last_positive = mid
            hi = mid - 1  
        else:
            lo = mid + 1
    return last_positive if last_positive is not None else lo

def finalize_ranges(refined: List[Dict], total_pages: int) -> List[Dict]:
    if not refined:
        return []
    refined_sorted = sorted(refined, key=lambda x: x["start"])
    cleaned = []
    last_end = 0
    for r in refined_sorted:
        s = max(r["start"], last_end)
        e = r["end"]
        if s > e:
            continue
        cleaned.append({"name": r["name"], "startPage": s + 1, "endPage": e + 1})
        last_end = e + 1
    merged = []
    for r in cleaned:
        if merged and merged[-1]["name"] == r["name"] and merged[-1]["endPage"] + 1 >= r["startPage"]:
            merged[-1]["endPage"] = max(merged[-1]["endPage"], r["endPage"])
        else:
            merged.append(r)
    for m in merged:
        m["startPage"] = max(1, min(m["startPage"], total_pages))
        m["endPage"] = max(1, min(m["endPage"], total_pages))
    return merged

def build_sections_from_text(pdf_path: str) -> List[Dict]:
    page_texts = extract_texts(pdf_path)
    total_pages = len(page_texts)
    print("Total pages:", total_pages)

    sampled = sampling_pass(page_texts, rate=SAMPLE_RATE)
    rough = build_rough_ranges(sampled, total_pages)

    refined = []
    for sec, s_idx, e_idx in rough:
    
        refine_low = max(0, s_idx - BIN_SEARCH_WINDOW)
        refine_high = min(total_pages - 1, e_idx + BIN_SEARCH_WINDOW)
        print(f"Refining section {sec} approx {s_idx}-{e_idx} -> search window {refine_low}-{refine_high}")

        start_found = binary_refine(page_texts, refine_low, refine_high, sec)
        if start_found >= total_pages:
            continue
        cur = start_found
        last_sec = start_found - 1
        scan_limit = min(total_pages - 1, start_found + BIN_SEARCH_WINDOW * 3)
        while cur <= scan_limit:
            info = classify_single_page(page_texts, cur)
            if info["section"] == sec:
                last_sec = cur
                cur += 1
            else:
                break
        if last_sec < start_found:
            last_sec = min(total_pages - 1, start_found + 2)
        refined.append({"name": sec, "start": start_found, "end": last_sec})
        print(f"  Refined {sec} -> pages {start_found} to {last_sec}")

    final = finalize_ranges(refined, total_pages)
    return final

def save_result_json(sections: List[Dict], out_path: str = "result.json"):
    data = {"sections": sections}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Saved result to", out_path)


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python solution.py path/to/minute_book.pdf")
        return
    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print("PDF not found:", pdf_path)
        return
    sections = build_sections_from_text(pdf_path)
    print("\nFinal sections (1-based):")
    for s in sections:
        print(f" - {s['name']}: {s['startPage']}–{s['endPage']}")
    save_result_json(sections)

if __name__ == "__main__":
    main()
