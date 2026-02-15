from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
import difflib
import os

class ScienceEditor:
    """
    Object 3: The LLM Wrapper. 
    Focuses on Technical Depth and Graph-Aware Writing.
    Now exclusively uses Transformers for maximum A100 performance.
    """
    def __init__(self, model_name: str, use_gpu: bool = True):
        self.model_name = model_name
        # Check for CUDA
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model = None  # Lazy load
        self.tokenizer = None
        
        print(f"ScienceEditor initialized. Model '{self.model_name}' will be loaded on first use (GPU={self.use_gpu}).")

    def _load_model(self):
        """Internal method to load the model using Transformers."""
        if self.model is not None:
            return

        print(f"Loading Model via Transformers: {self.model_name}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Smart device mapping for A100
            device_map = "auto" if self.use_gpu else "cpu"
            
            # Using bfloat16 is best for A100
            torch_dtype = torch.bfloat16 if self.use_gpu else "auto"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR loading model: {e}")
            raise e

    def _generate(self, prompt_text: str, max_new_tokens: int = 4096) -> str:
        """Internal helper for generation using the Transformers pipeline."""
        if self.model is None:
            self._load_model()

        # Formatting strictly for Instruct models
        messages = [
            {"role": "system", "content": "You are a helpful compliant research assistant. Answer strictly based on the provided text. Do not converse."},
            {"role": "user", "content": prompt_text}
        ]
        
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        model_inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True 
        )
        
        # Decode only the new tokens
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def extract_rich_concepts(self, text: str, paper_id: str) -> list:
        # ...existing code...
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
        # ...existing code...
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
        # ...existing code...
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
        # ...existing code...
        valid_intents = ["BACKGROUND", "METHOD", "COMPARISON", "CONTRADICTION", "EXTENSION"]
        for v in valid_intents:
            if v in intent:
                return v
        return "BACKGROUND"

    def align_entity_canonical(self, entity_name: str, existing_entities: list) -> str:
        # ...existing code...
        if not existing_entities: return entity_name
        matches = difflib.get_close_matches(entity_name, existing_entities, n=20, cutoff=0.4)
        if not matches: return entity_name
        candidates = ", ".join(matches)
        prompt = f"""Canonicalize the entity name. If it matches an existing entity (synonym/acronym), return the existing one. Otherwise return the new one.
Existing: {candidates}
New Entity: {entity_name}
Canonical Name:
"""
        return self._generate(prompt, max_new_tokens=20).strip()

    def summarize_community(self, paper_titles: list) -> str:
        # ...existing code...
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
        # ...existing code...
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
        # ...existing code...
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
        # ...existing code...
        try:
            text = text.strip()
            if not text.startswith("["): text = "[" + text
            if not text.endswith("]"): text = text.split("]")[0] + "]"
            return json.loads(text)
        except:
            return []