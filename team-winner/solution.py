#!/usr/bin/env python3
"""
Simple PDF text extraction test script.
Extracts text from a PDF and displays it for testing purposes.
"""

import sys
import base64
import io
import time
import json
import argparse
import concurrent.futures
from pathlib import Path
import requests
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
API_URL = "https://ai-models.autocomply.ca"
API_KEY = "sk-ac-7f8e9d2c4b1a6e5f3d8c7b9a2e4f6d1c"
MODEL = "gemini-2.5-flash"

# Relative paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_FILE = SCRIPT_DIR / "outputs" / "merged_test_results.txt"
JSON_OUTPUT_FILE = SCRIPT_DIR / "result.json"

BATCH_SIZE = 12
GRID_COLS = 4

def process_page_image(doc: fitz.Document, page_num: int) -> Image.Image:
    """
    1. Render page
    2. Zoom 10% (crop edges)
    3. Crop to top 1/4
    4. Resize to 50%
    5. Draw red page number
    """
    page = doc.load_page(page_num)
    
    # 1. Render (standard resolution)
    pix = page.get_pixmap()
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    
    # 2. Zoom in by 10% (crop 5% from all sides to remove margins)
    w, h = img.size
    zoom_w = int(w * 0.05)
    zoom_h = int(h * 0.05)
    img = img.crop((zoom_w, zoom_h, w - zoom_w, h - zoom_h))
    
    # 3. Crop to top 1/4, skipping top 1/15 margin
    w, h = img.size
    margin_top = int(h / 15)
    crop_height = int(h / 4)
    img = img.crop((0, margin_top, w, margin_top + crop_height))
    
    # 4. Resize to 50%
    new_w = int(w * 0.5)
    new_h = int(img.size[1] * 0.5)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 5. Draw red page number
    draw = ImageDraw.Draw(img)
    
    # Try to load a large font, fallback to default
    try:
        # Windows standard font
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()
        
    text = str(page_num + 1)
    
    # Draw text with a white outline for visibility
    x, y = 10, 10
    outline_color = "white"
    text_color = "red"
    
    # Draw outline
    for adj in range(-2, 3):
        for adj2 in range(-2, 3):
            draw.text((x+adj, y+adj2), text, font=font, fill=outline_color)
            
    # Draw main text
    draw.text((x, y), text, font=font, fill=text_color)
    
    return img

def create_grid_image(images: list[Image.Image], cols: int) -> Image.Image:
    """Merges a list of images into a grid."""
    if not images:
        return None
        
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    
    grid_w = w * cols
    grid_h = h * rows
    
    grid_img = Image.new('RGB', (grid_w, grid_h), color='white')
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid_img.paste(img, (col * w, row * h))
        
    return grid_img

def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_api(base64_image: str, prompt: str) -> str:
    """Send the image to the API with retry logic."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "pdfPage": base64_image,
        "prompt": prompt,
        "model": MODEL
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"  Sending request to {API_URL}/process-pdf (Attempt {attempt+1}/{max_retries})...")
            response = requests.post(
                f"{API_URL}/process-pdf",
                json=payload,
                headers=headers,
                timeout=120 
            )
            
            if response.status_code == 200:
                result = response.json().get("result", "No result field in response")
                return result
            elif response.status_code in [504, 502, 500]:
                print(f"  [Warning] Server Error {response.status_code}. Retrying in 5 seconds...")
                time.sleep(5)
                continue
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            print(f"  [Error] Exception: {e}. Retrying...")
            time.sleep(5)
            
    return "Failed to get response after multiple retries."

def call_text_api(query: str) -> str:
    """Calls the /ask endpoint for text-only queries."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"query": query, "model": MODEL}
    
    # Allow 1 retry if 504 happens
    for attempt in range(2):
        try:
            print(f"  Sending text request to {API_URL}/ask...")
            response = requests.post(f"{API_URL}/ask", json=payload, headers=headers, timeout=60)
            
            if response.status_code == 200:
                return response.json().get("result", "")
            elif response.status_code == 504:
                if attempt == 0:
                    print(f"  [Warning] Server Error 504. Retrying...")
                    time.sleep(2)
                    continue
                return f"Error: {response.status_code} - {response.text}"
            else:
                return f"Error: {response.status_code} - {response.text}"
            
        except Exception as e:
            return f"Exception calling API: {e}"
    return "Failed to get response."

def process_batch(pdf_path: Path, start_idx: int, end_idx: int, batch_num: int) -> tuple[int, str]:
    """Process a single batch of pages in a separate thread."""
    print(f"Starting Batch {batch_num} (Pages {start_idx+1}-{end_idx})...")
    
    # Open a new document instance for this thread to ensure thread safety
    doc = fitz.open(pdf_path)
    
    batch_images = []
    batch_page_nums = []
    
    try:
        for p_idx in range(start_idx, end_idx):
            img = process_page_image(doc, p_idx)
            batch_images.append(img)
            batch_page_nums.append(p_idx + 1)
    finally:
        doc.close()
        
    # Create grid
    grid_img = create_grid_image(batch_images, GRID_COLS)
    
    # Save debug image
    debug_path = OUTPUT_FILE.parent / f"debug_grid_{batch_num}.png"
    grid_img.save(debug_path)
    
    # API Call
    b64_img = image_to_base64(grid_img)
    
    prompt = f"""
    This image contains {len(batch_images)} document snippets arranged in a grid.
    Each snippet has a LARGE RED NUMBER overlaid on it.
    
    Your task:
    1. Identify the red number on each snippet.
    2. Extract the text from that snippet.
    3. Return the result as a list.
    
    Format:
    Page [Number]: [Extracted Text]
    
    If a snippet appears empty or has no text, write:
    Page [Number]: [NO TEXT]
    
    Example:
    Page 1: MINUTES OF A MEETING OF THE BOARD OF DIRECTORS...
    Page 2: [NO TEXT]
    
    Process pages: {min(batch_page_nums)} to {max(batch_page_nums)}.
    ENSURE YOU OUTPUT AN ENTRY FOR EVERY PAGE NUMBER FROM {min(batch_page_nums)} TO {max(batch_page_nums)}.
    """
    
    result = call_api(b64_img, prompt)
    
    formatted_output = f"\n--- Batch {batch_num} (Pages {min(batch_page_nums)}-{max(batch_page_nums)}) ---\n"
    formatted_output += result + "\n"
    formatted_output += "-" * 50 + "\n"
    
    return batch_num, formatted_output

def main():
    parser = argparse.ArgumentParser(description="Extract text and structure from a Minute Book PDF.")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file to process.")
    args = parser.parse_args()

    pdf_path = args.pdf_path

    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    print(f"Processing PDF: {pdf_path.name}")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Total pages: {total_pages}")
    
    # Clear output file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Processing Report for {pdf_path.name}\n")
        f.write("="*50 + "\n\n")

    # Close the initial doc as we open new ones in threads
    doc.close()

    # Prepare batches
    batches = []
    current_page_idx = 0
    batch_count = 0
    while current_page_idx < total_pages:
        batch_count += 1
        end_idx = min(current_page_idx + BATCH_SIZE, total_pages)
        batches.append((current_page_idx, end_idx, batch_count))
        current_page_idx = end_idx

    results = {}
    
    print(f"Starting parallel processing of {len(batches)} batches with 5 workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_batch = {
            executor.submit(process_batch, pdf_path, start, end, b_num): b_num 
            for start, end, b_num in batches
        }
        
        for future in concurrent.futures.as_completed(future_to_batch):
            b_num = future_to_batch[future]
            try:
                batch_num, text_result = future.result()
                results[batch_num] = text_result
                print(f"Batch {batch_num} completed.")
            except Exception as exc:
                print(f"Batch {b_num} generated an exception: {exc}")
                results[b_num] = f"\n--- Batch {b_num} FAILED ---\nError: {exc}\n" + "-" * 50 + "\n"

    # Write results in order
    print(f"Writing results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i in range(1, batch_count + 1):
            if i in results:
                f.write(results[i])
            else:
                f.write(f"\n--- Batch {i} MISSING ---\n" + "-" * 50 + "\n")

    print(f"\nDone. Results in {OUTPUT_FILE}")

    # --- CHECK FOR EMPTY BATCH 1 ---
    try:
        content = OUTPUT_FILE.read_text(encoding="utf-8")
        empty_batch_marker = "--- Batch 1 (Pages 1-12) ---\n\n--------------------------------------------------"
        # Note: BATCH_SIZE is 12 now, so pages are 1-12. Adjusting check logic dynamically.
        
        # Construct the marker based on actual batch size used
        batch1_end = min(BATCH_SIZE, total_pages)
        marker = f"--- Batch 1 (Pages 1-{batch1_end}) ---\n\n--------------------------------------------------"
        
        if marker in content:
            print("\n[Warning] Batch 1 appears empty. Retrying Batch 1...")
            
            # Re-open doc
            doc = fitz.open(pdf_path)
            
            # Re-process Batch 1
            batch_images = []
            batch_page_nums = []
            for p_idx in range(0, batch1_end):
                img = process_page_image(doc, p_idx)
                batch_images.append(img)
                batch_page_nums.append(p_idx + 1)
            
            grid_img = create_grid_image(batch_images, GRID_COLS)
            b64_img = image_to_base64(grid_img)
            
            prompt = f"""
            This image contains {len(batch_images)} document snippets arranged in a grid.
            Each snippet has a LARGE RED NUMBER overlaid on it.
            
            Your task:
            1. Identify the red number on each snippet.
            2. Extract the text from that snippet.
            3. Return the result as a list.
            
            Format:
            Page [Number]: [Extracted Text]
            
            If a snippet appears empty or has no text, write:
            Page [Number]: [NO TEXT]
            
            Process pages: {min(batch_page_nums)} to {max(batch_page_nums)}.
            ENSURE YOU OUTPUT AN ENTRY FOR EVERY PAGE NUMBER FROM {min(batch_page_nums)} TO {max(batch_page_nums)}.
            """
            
            print("Calling API for Batch 1 Retry...")
            result = call_api(b64_img, prompt)
            
            # Replace in file
            new_content = content.replace(marker, f"--- Batch 1 (Pages 1-{batch1_end}) ---\n{result}\n--------------------------------------------------")
            OUTPUT_FILE.write_text(new_content, encoding="utf-8")
            print("Batch 1 updated in output file.")
            doc.close()
            
    except Exception as e:
        print(f"Error checking/retrying Batch 1: {e}")

    # --- FINAL STEP: Structure Extraction ---
    print("\n" + "="*50)
    print("FINAL STEP: Generating JSON Structure")
    print("="*50)
    
    try:
        extracted_text = OUTPUT_FILE.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading output file: {e}")
        return

    def parse_json_response(response_text):
        try:
            clean_json = response_text
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
            return json.loads(clean_json)
        except Exception:
            return None

    def get_structure(text_segment, current_sections=None, part_idx=1, total_parts=1):
        if total_parts == 1:
            # Strategy 1: Original Prompt (Single Pass)
            prompt = """
    You are a legal document AI assistant analyzing a corporate minute book.
    Below is the text extracted from the document pages, with page numbers indicated.

    Identify the start and end pages for each of the following sections:
    - Articles & Amendments
    - By Laws
    - Unanimous Shareholder Agreement
    - Minutes & Resolutions
    - Directors Register
    - Officers Register
    - Shareholder Register
    - Securities Register
    - Share Certificates
    - Ultimate Beneficial Owner Register

    Return the result STRICTLY as a JSON object with a "sections" key containing a list of objects.
    Each object must have:
    - "name": The section name (from the list above)
    - "startPage": The first page number of the section
    - "endPage": The last page number of the section

    Example:
    {
      "sections": [
        { "name": "Articles & Amendments", "startPage": 1, "endPage": 20 },
        { "name": "By Laws", "startPage": 21, "endPage": 45 }
      ]
    }

    If a section is not found, do not include it.
    If pages are missing or unknown, skip them.

    --- EXTRACTED TEXT ---
    """ + text_segment
            return call_text_api(prompt)

        else:
            # Strategy 2 & 3: Context-aware Prompt (Multi-part)
            context_prompt = ""
            if current_sections:
                context_prompt = f"""
    --- PREVIOUS ANALYSIS CONTEXT ---
    We are analyzing a large document in parts.
    So far, we have identified these sections:
    {json.dumps(current_sections, indent=2)}
    
    Your task is to CONTINUE building this list based on the new text provided below.
    - If the text continues a section from the previous list, update its "endPage".
    - If a new section starts, add it to the list.
    - Return the COMPLETE updated list of sections.
    """
            
            prompt = f"""
    You are a legal document AI assistant analyzing a corporate minute book (Part {part_idx} of {total_parts}).
    Below is the text extracted from the document pages, with page numbers indicated.

    Identify the start and end pages for each of the following sections:
    - Articles & Amendments
    - By Laws
    - Unanimous Shareholder Agreement
    - Minutes & Resolutions
    - Directors Register
    - Officers Register
    - Shareholder Register
    - Securities Register
    - Share Certificates
    - Ultimate Beneficial Owner Register

    {context_prompt}

    Return the result STRICTLY as a JSON object with a "sections" key containing a list of objects.
    Each object must have:
    - "name": The section name (from the list above)
    - "startPage": The first page number of the section
    - "endPage": The last page number of the section

    Example:
    {{
      "sections": [
        {{ "name": "Articles & Amendments", "startPage": 1, "endPage": 20 }},
        {{ "name": "By Laws", "startPage": 21, "endPage": 45 }}
      ]
    }}

    If a section is not found, do not include it.
    If pages are missing or unknown, skip them.

    --- EXTRACTED TEXT (Part {part_idx}/{total_parts}) ---
    """ + text_segment
            
            return call_text_api(prompt)

    # Strategy: Try 1 chunk, then 4, then 8
    strategies = [1, 4, 8]
    final_json_obj = None

    for num_chunks in strategies:
        print(f"\nAttempting structure extraction with {num_chunks} chunk(s)...")
        
        lines = extracted_text.splitlines(keepends=True)
        total_lines = len(lines)
        chunk_size = (total_lines + num_chunks - 1) // num_chunks
        
        current_sections = []
        success = True
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_lines)
            chunk_text = "".join(lines[start:end])
            
            if not chunk_text.strip():
                continue

            print(f"  Processing chunk {i+1}/{num_chunks}...")
            response = get_structure(chunk_text, current_sections, i+1, num_chunks)
            
            parsed = parse_json_response(response)
            if parsed and "sections" in parsed:
                current_sections = parsed["sections"]
            else:
                print(f"  Failed to parse JSON for chunk {i+1}. Response start: {str(response)[:100]}...")
                success = False
                break
        
        if success:
            final_json_obj = {"sections": current_sections}
            print("  Success!")
            break
        else:
            print("  Strategy failed. Trying next strategy...")

    if final_json_obj:
        print("\n--- Final JSON Result ---")
        print(json.dumps(final_json_obj, indent=2))
        
        try:
            with open(JSON_OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(final_json_obj, f, indent=2)
            print(f"JSON saved to {JSON_OUTPUT_FILE}")
        except Exception as e:
            print(f"Error saving JSON: {e}")
    else:
        print("All strategies failed to generate valid JSON.")

if __name__ == "__main__":
    main()
