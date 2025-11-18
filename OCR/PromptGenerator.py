from typing import Mapping


class PromptGenerator:

    def __init__(self, categories: Mapping[str, str]):
        self.categories = "\n".join(
            [f"{key} - {value}" for key, value in categories.items()]
        )

    def get_main_prompt(self) -> str:
        return (
            "You are analyzing pages from a corporate Minute Book. Below are text samples extracted from each page. "
            "Your task is to identify which of the 10 possible Minute Book sections each page most likely belongs to.\n\n"
            "The possible categories are (use EXACTLY these number mappings):\n"
            f"{self.categories}\n\n"
            "**CRITICAL Instructions:**\n"
            "- Each section appears AT MOST ONCE in the document and forms a CONTINUOUS block of pages.\n"
            "- Not all sections may be present in the document.\n"
            "- Sections appear in sequential order (1→2→3→...→10), never out of order.\n"
            "- Once a section ends, it NEVER appears again later in the document.\n"
            "- Look for clear section markers like titles, headers, table formats, and document types.\n"
            "- Minutes & Resolutions (4) typically contains meeting records with dates and signatures.\n"
            "- Registers (5-8,10) are typically tabular with columns for names, addresses, dates.\n"
            "- Share Certificates (9) are formal certificates with certificate numbers.\n"
            "- When pages have similar content to the current section, maintain that classification.\n"
            "- Confidence score must be 0-100 (use lower scores like 60-70 for ambiguous pages).\n"
            "- Do NOT explain your reasoning.\n\n"
            "Output format (CSV):\n"
            "Page Number, Category Number, Confidence Score\n\n"
            "Example:\n"
            "1, 1, 95\n"
            "2, 1, 90\n\n"
            "Now provide your answer in this exact CSV format for ALL pages provided below:"
        )

    def get_discontinuity_prompt(self, context_before: str, context_after: str) -> str:
        return (
            "You are analyzing a discontinuity in a corporate Minute Book classification.\n\n"
            f"{context_before}\n"
            f"{context_after}\n\n"
            "Re-classify the pages below, ensuring they fit logically between the surrounding sections.\n"
            "Remember: each section appears AT MOST ONCE and sections are continuous blocks.\n\n"
            "The possible categories are:\n"
            f"{self.categories}\n\n"
            "Output format (CSV):\n"
            "Page Number, Category Number, Confidence Score\n\n"
            "Do NOT explain your reasoning.\n\n"
        )
