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
        self.use_gpu = use_gpu and torch.cuda.is_available()  # Only enable if GPU available
        self.model = None  # Lazy load
        self.tokenizer = None

        # Auto-detect local GGUF if extension is missing
        if not os.path.exists(self.model_name) and os.path.exists(self.model_name + ".gguf"):
            self.model_name += ".gguf"

        self.backend = "llama" if "gguf" in self.model_name.lower() else "transformers"
        print(f"ScienceEditor initialized. Model '{self.model_name}' will be loaded on first use (GPU={self.use_gpu}).")

    def _detect_gpu_layers(self, total_layers: int = 32) -> int:
        """
        Detect a safe number of layers to load on GPU based on VRAM.
        Defaults to 50% of layers if VRAM is limited.
        """
        if not self.use_gpu:
            return 0
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = mem_info.free / 1024**3
            # Heuristic: assume each layer ~1GB
            n_gpu_layers = min(total_layers, max(1, int(free_gb * 0.8)))
            return n_gpu_layers
        except:
            # Fallback: assume all layers to GPU
            return -1

    def _load_model(self):
        """Internal method to load the model only when needed."""
        print(f"Loading Model: {self.model_name}...")

        if self.backend == "llama":
            # Determine GPU layers
            n_gpu_layers = self._detect_gpu_layers()
            main_gpu = 0 if self.use_gpu else -1

            # Check if it's a local file or a HF Repo
            if os.path.exists(self.model_name):
                print(f"Loading local GGUF: {self.model_name}")
                self.model = Llama(
                    model_path=self.model_name,
                    n_ctx=32768,
                    n_gpu_layers=n_gpu_layers,
                    main_gpu=main_gpu,
                    n_batch=2048,
                    use_mmap=not self.use_gpu,  # mmap only if CPU fallback
                    verbose=True
                )
            else:
                # It's a HF Repo - Download the Q4_K_M variant by default
                print(f"Path '{self.model_name}' not found locally. Treating as HF Repo...")
                self.model = Llama.from_pretrained(
                    repo_id=self.model_name,
                    filename="Qwen3-4B-Instruct-2507-Q5_K_M.gguf",
                    n_ctx=32768,
                    n_gpu_layers=n_gpu_layers,
                    main_gpu=main_gpu,
                    n_batch=2048,
                    use_mmap=not self.use_gpu,
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

    # --- Rest of your methods remain exactly the same ---
    def extract_rich_concepts(self, text: str, paper_id: str) -> list:
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
        for v in valid_intents:
            if v in intent:
                return v
        return "BACKGROUND"

    def align_entity_canonical(self, entity_name: str, existing_entities: list) -> str:
        if not existing_entities:
            return entity_name
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
{context_str[:12000]}

ANALYSIS:
# Community Synthesis
"""
        content = self._generate(prompt, max_new_tokens=4096)
        return {"title": "Community Analysis", "body": content}

    def write_technical_article(self, task: dict, context_text: str) -> dict:
        task_type = task['type']
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
        return {"title": task['subject'], "body": content}

    def _clean_json(self, text: str) -> list:
        try:
            text = text.strip()
            if not text.startswith("["): text = "[" + text
            if not text.endswith("]"): text = text.split("]")[0] + "]"
            return json.loads(text)
        except:
            return []
