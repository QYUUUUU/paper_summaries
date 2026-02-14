import os
import fitz  # PyMuPDF
import re
from typing import Dict, List, Tuple

class AssetMiner:
    """
    Object 1: Handles raw data ingestion.
    Extracts text, cleans math for LaTeX, and mines images/tables.
    """
    def __init__(self, papers_dir: str, assets_dir: str):
        self.papers_dir = papers_dir
        # Create a true assets subdirectory to keep the output clean
        self.assets_dir = os.path.join(assets_dir, "assets")
        os.makedirs(self.assets_dir, exist_ok=True)

    def process_pdf(self, filename: str) -> Dict:
        """
        Main entry point for a single PDF.
        Returns a dictionary with metadata, text, and list of saved image paths.
        """
        filepath = os.path.join(self.papers_dir, filename)
        doc = fitz.open(filepath)
        
        # 1. Extract Text (Layout-Aware)
        # Scientific papers are often 2-column. 
        # Using PyMuPDF's built-in 'sort=True' flag handles column detection much better 
        # than naive xy-sorting, which interleaves columns.
        full_text = ""
        for page in doc:
            # sort=True attempts to sequence text blocks in reading order (top-left, col1, col2, etc.)
            full_text += page.get_text("text", sort=True) + "\n"
        
        cleaned_text = self._clean_text_for_latex(full_text)
        
        # 2. Extract Images
        image_paths = self._extract_images(doc, filename)
        
        # 3. Basic Metadata Heuristic (Title is usually first line, Abstract is first paragraph)
        # In a real scenario, use specific metadata parsing or the LLM to refine this.
        title = filename.replace('.pdf', '')
        
        return {
            "id": filename,
            "title": title,
            "text": cleaned_text,
            "images": image_paths,
            "page_count": len(doc)
        }

    def _extract_images(self, doc: fitz.Document, filename: str) -> List[str]:
        """
        Saves significant images (ignoring tiny icons) to the assets folder.
        """
        saved_images = []
        # Sanitize filename for filesystem safety
        safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
        
        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                except Exception as e:
                    continue # Skip broken images
                
                image_bytes = base_image["image"]
                
                # Filter small icons/logos (arbitrary threshold 10kb)
                if len(image_bytes) < 10240: 
                    continue
                
                image_ext = base_image["ext"]
                image_name = f"{safe_filename}_p{page_index}_i{img_index}.{image_ext}"
                image_path = os.path.join(self.assets_dir, image_name)
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                # Store relative path for HTML (now matches actual location)
                saved_images.append(os.path.join("assets", image_name))
                
        return saved_images

    def _clean_text_for_latex(self, text: str) -> str:
        """
        Heuristic to wrap potential math equations in $$ tags.
        """
        # Example: looking for isolated greek letters or common math symbols
        # This is a simple regex; LLM rewriting is usually better for this.
        text = re.sub(r'(\b[a-z]\s*=\s*[0-9]+\b)', r'$$\1$$', text)
        return text