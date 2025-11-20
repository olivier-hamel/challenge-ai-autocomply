# PDF Text Extraction & Structure Analysis

This tool extracts text from Minute Book PDFs and identifies the structure of the document (sections like Articles, By-Laws, etc.) using AI models.

## How to Execute

### 1. Install Dependencies

Ensure you have Python 3 installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Run the Script

Run the script by providing the path to the PDF file you want to process:

```bash
python solution.py path/to/your/document.pdf
```

The script will generate:

* `result.json`: Structured JSON identifying document sections and page ranges.

**Important:** The output file is intended strictly for internal program use. Its format and structure are optimized for machine processing, not for end-user readability.

## How it Works

* **Image Preprocessing**: The script renders PDF pages as images, crops margins, resizes them for efficiency, and overlays a large red page number on each page for clear identification by the AI.
* **Batching & Grid Creation**: To optimize API usage, it groups multiple page images (e.g., 12 pages) into a single "grid" image.
* **AI-Powered Extraction**: These grid images are sent in parallel to a Vision Language Model (VLM) API, allowing multiple extraction requests to run simultaneously. The VLM reads the text from each snippet and associates it with the overlaid page number.
* **Structural Analysis**: Finally, the script aggregates all extracted text and sends it to a text-based LLM to identify logical sections (e.g., "Articles & Amendments", "By Laws") and their corresponding page ranges, saving the final structure as JSON.

**Note:** API responses may occasionally return *504 Gateway Timeout* errors due to heavy loads, but automatic retry mechanisms are implemented to handle these cases gracefully.

---

## From the Creator

As someone who likes LLMs a lot and is familiar with AI agent systems and tools like CrewAI and LangGraph, my reasoning for this project was to minimize the number of requests and simply let the AI do the work of checking and classifying everything.

It's funny — I initially wanted to showcase complex AI-agent structures, but for this problem, I recognized that those tools aren’t the most efficient approach.

Have fun reviewing my project! Also, yes, the solution is engineered by me for the algorithm, but I used AI to help build some components faster (nce this is just a prototype) and generate comments to assist both myself and you in reviewing the code.
