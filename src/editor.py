from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
import difflib
import os
from llama_cpp import Llama

class ScienceEditor:
    """
    Object 3: The LLM Wrapper. 
    Focuses on Technical Depth and Graph-Aware Writing.
    Supports both Transformers (standard) and Llama.cpp (GGUF).
    """
    def __init__(self, model_name: str, use_gpu: bool = True):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model = None # Lazy load
        self.tokenizer = None
        
        # Auto-detect local GGUF if extension is missing
        if not os.path.exists(self.model_name) and os.path.exists(self.model_name + ".gguf"):
            self.model_name += ".gguf"

        self.backend = "llama" if "gguf" in self.model_name.lower() else "transformers"
        print(f"ScienceEditor initialized. Model '{self.model_name}' will be loaded on first use (GPU={self.use_gpu}).")

    def _load_model(self):
        """Internal method to load the model only when needed."""
        print(f"Loading Model: {self.model_name}...")
        
        if self.backend == "llama":
            # Determine layer offloading based on GPU flag
            # Use a large number instead of -1 if -1 is problematic, but -1 is standard.
            n_gpu = -1 if self.use_gpu else 0
            
            # Check if it's a local file or a HF Repo
            if os.path.exists(self.model_name):
                print(f"Loading local GGUF: {self.model_name}")
                self.model = Llama(
                    model_path=self.model_name,
                    n_ctx=32768, 
                    n_gpu_layers=-1, # Ensure all layers go to GPU
                    main_gpu=0,      # Explicitly select the A100 (device 0)
                    n_batch=2048, 
                    use_mmap=False,  # <--- CRITICAL: Disable mmap to force VRAM loading
                    verbose=True
                )
            else:
                # It's a HF Repo - Download the Q4_K_M variant by default
                print(f"Path '{self.model_name}' not found locally. Treating as HF Repo...")
                self.model = Llama.from_pretrained(
                    repo_id=self.model_name,
                    filename="Qwen3-4B-Instruct-2507-Q5_K_M.gguf",
                    n_ctx=32768, 
                    n_gpu_layers=-1,
                    main_gpu=0,
                    n_batch=2048, 
                    use_mmap=False,  # <--- CRITICAL: Disable mmap
                    verbose=True
                )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            device_map = "auto" if self.use_gpu else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map=device_map
            )
    
    def _generate(self, prompt_text: str, max_new_tokens: int = 4096) -> str:
        """Internal helper for generation using the Transformers pipeline."""
        if self.model is None:
            self._load_model()

        if self.backend == "llama":
            # GGUF Generation
            response = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful compliant research assistant. Answer strictly based on the provided text. Do not converse."},
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens=max_new_tokens,
                temperature=0.7
            )
            return response['choices'][0]['message']['content']
        else:
            # Transformers Generation
            messages = [
                {"role": "system", "content": "You are a helpful compliant research assistant. Answer strictly based on the provided text. Do not converse."},
                {"role": "user", "content": prompt_text}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def extract_rich_concepts(self, text: str, paper_id: str) -> list:
        """
        Extracts complex relationships.
        We want to know if a paper OPTIMIZES, CRITIQUES, or APPLIES a concept.
        """
        # CRITICAL UPDATE: High context limit to capture full nuance
        prompt = f"""You are an NLP Knowledge Graph Architect.
Extract entities (Models, Metrics, Datasets, Architectures) and their specific relationship to the paper.
Relations must be specific: "OPTIMIZES", "INTRODUCES", "CRITIQUES", "USES", "TRAINS_ON", "EVALUATES_ON".
Return a JSON list.

Text: {text[:12000]}
Format: [{{"target": "ConceptName", "relation": "RELATION_TYPE", "strength": 0.0-1.0}}]
JSON:
"""
        return self._clean_json(self._generate(prompt, max_new_tokens=300))

    def extract_tdms_schema(self, text: str, paper_id: str) -> list:
        """
        Extracts entities strictly adhering to the TDMS (Task, Dataset, Metric, Method) schema.
        Also attempts to align with Computer Science Ontology (CSO) concepts.
        """
        # CRITICAL UPDATE: High context limit to read full Methods/Results
        prompt = f"""You are a Scientific Knowledge Graph Builder.
Extract entities strictly matching these types:
- TASK (e.g., "Extractive Summarization", "Question Answering")
- DATASET (e.g., "SQuAD", "GSM8K")
- METRIC (e.g., "ROUGE-L", "F1 Score")
- METHOD (e.g., "Transformer", "LoRA", "PPO")

For each entity, determine the relation to the paper (e.g., "EVALUATED_ON", "PROPOSES", "USES_BASELINE").
Return a JSON list.

Text: {text[:15000]}
Format: [{{"entity": "Name", "type": "TYPE", "relation": "RELATION", "cso_topics": ["Topic1", "Topic2"]}}]
JSON:
"""
        return self._clean_json(self._generate(prompt, max_new_tokens=500))

    def classify_citation_intent(self, context_text: str, cited_paper_title: str) -> str:
        """
        Determines the semantic intent of a citation based on the context.
        Categories: BACKGROUND, METHOD, COMPARISON, CONTRADICTION, EXTENSION.
        """
        prompt = f"""Classify the citation intent for "{cited_paper_title}" based on the text.
Categories:
- BACKGROUND (Theoretical foundation, definitions)
- METHOD (Uses tool/dataset/method from cited work)
- COMPARISON (Compares performance against cited work)
- CONTRADICTION (Refutes/critiques cited work)
- EXTENSION (Builds directly upon cited work)

Context: "...{context_text}..."
Intent:
"""
        intent = self._generate(prompt, max_new_tokens=20).strip().upper()
        valid_intents = ["BACKGROUND", "METHOD", "COMPARISON", "CONTRADICTION", "EXTENSION"]
        
        # Simple fuzzy matching or fallback
        for v in valid_intents:
            if v in intent:
                return v
        return "BACKGROUND" # Default

    def align_entity_canonical(self, entity_name: str, existing_entities: list) -> str:
        """
        Uses LLM to map a raw entity to a canonical list if a close match exists.
        Simulates Entity Resolution.
        """
        if not existing_entities:
            return entity_name
            
        # Optimization: Only show LLM the top 20 lexically similar candidates to avoid context overflow
        matches = difflib.get_close_matches(entity_name, existing_entities, n=20, cutoff=0.4)
        if not matches:
             return entity_name

        candidates = ", ".join(matches) 
        prompt = f"""Canonicalize the entity name. If it matches an existing entity (synonym/acronym), return the existing one. Otherwise return the new one.
Existing: {candidates}

New Entity: {entity_name}
Canonical Name:
"""
        return self._generate(prompt, max_new_tokens=20).strip()

    def summarize_community(self, paper_titles: list) -> str:
        """Generates a high-level theme for a cluster of papers."""
        titles_str = "\n".join([f"- {t}" for t in paper_titles])
        prompt = f"""You are an expert NLP classifier. 
Analyze the following list of research paper titles and determine the single most specific research sub-field they belong to.
Do not say you don't know. Infer the field from the keywords in the titles.
Return ONLY the name of the sub-field (e.g., "Parameter-Efficient Fine-Tuning").

Papers:
{titles_str}

Specific Sub-Field:"""
        return self._generate(prompt, max_new_tokens=50).strip()

    def generate_contrastive_summary(self, paper_ids: list, context_str: str) -> dict:
        """
        Synthesizes a community report focusing on contrastive analysis.
        """
        prompt = f"""You are a Lead Researcher performing a Meta-Analysis.
OBJECTIVE: Synthesize the findings of the provided papers into a coherent technical narrative.
INSTRUCTIONS:
1. Identify the common problem being addressed.
2. Group papers by their approach (Methodology Families).
3. EXPLICITLY COMPARE results. Who is SOTA? What are the trade-offs (Speed vs Accuracy)?
4. Highlight contradictions or disagreements. Did Paper B refute Paper A?
5. Use the specific entity names provided in the context.
6. Do NOT just summarize list-style. Write a comparative essay.

FORMAT: Markdown with sections.

CONTEXT:
{context_str[:12000]} # Truncate to fit context

ANALYSIS:
# Community Synthesis
"""
        content = self._generate(prompt, max_new_tokens=4096)
        
        return {
            "title": "Community Analysis",
            "body": content
        }

    def write_technical_article(self, task: dict, context_text: str) -> dict:
        """
        Writes a deep technical analysis based on the topology type.
        """
        task_type = task['type']
        
        # 1. Define the Persona and Structure based on Graph Topology
        if task_type == "COMPARATIVE_ANALYSIS":
            persona = "You are a Benchmarking Engineer."
            instruction = "Compare the methodologies strictly. Which handles compute better? Which is more data efficient? Use math/logic, not fluff."
        elif task_type == "ARCHITECTURAL_REVIEW":
            persona = "You are a Systems Architect."
            instruction = "Deconstruct the model architecture. Explain the specific mechanism (e.g., Attention modification) that makes it novel."
        elif task_type == "EVOLUTION_TIMELINE":
            persona = "You are an AI Historian."
            instruction = "Trace the lineage. Paper A introduced X, but failed at Y. Paper B fixed Y using Z."
        else:
            persona = "You are a Research Scientist."
            instruction = "Synthesize the findings. What is the consensus? What are the outliers?"

        prompt = f"""{persona}
OBJECTIVE: Write a technical article about: {task['subject']}.
CONTEXT: {instruction}

RULES:
1. NO INTROS ("In recent years..."). Start immediately with the technical problem.
2. CITE HEAVILY: Refer to specific papers provided in the source.
3. BE CRITICAL: Mention limitations and trade-offs.
4. FORMAT: Markdown. Use code blocks for algorithms/math.

SOURCE MATERIAL:
{context_text}

ARTICLE:
# {task['subject']}
"""
        content = self._generate(prompt, max_new_tokens=4096)
        
        return {
            "title": task['subject'],
            "body": content
        }

    def _clean_json(self, text: str) -> list:
        try:
            text = text.strip()
            if not text.startswith("["): text = "[" + text
            if not text.endswith("]"): text = text.split("]")[0] + "]"
            return json.loads(text)
        except:
            return []