import os
import re
import argparse
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

# Import our upgraded modules
from src.miner import AssetMiner # Assuming miner still handles raw PDF text extraction
from src.graph import KnowledgeNetwork
from src.editor import ScienceEditor
from src.publisher import HtmlRenderer

# --- CONFIG ---
PAPERS_DIR = "./papers" 
OUTPUT_DIR = "./output"
    # CRITICAL UPDATE: Per "Top-Notch" requirements, 1.5B is insufficient for complex graph reasoning.
# Switched to Qwen 2.5 32B (Q5_K_M) - High performance model for A100 40GB.
MODEL_PATH = "Qwen/Qwen2.5-32B-Instruct" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def smart_chunker(text: str):
    """
    Upgraded: Uses semantic mapping to identify paper sections 
    regardless of exact naming conventions.
    """
    # Define semantic maps for research paper structures
    section_map = {
        'intro': ['introduction', 'background', 'motivation', 'related work'],
        'methods': ['methodology', 'architecture', 'proposed approach', 'implementation'],
        'results': ['experiments', 'evaluation', 'results', 'discussion', 'ablation']
    }
    
    lines = text.split('\n')
    sections = {'intro': [], 'methods': [], 'results': [], 'other': []}
    current_sec = 'intro'

    for line in lines:
        clean_line = line.strip().lower()
        # Heuristic: Short lines that match section keywords are headers
        if len(clean_line) < 50: 
            found_header = False
            for sec_key, keywords in section_map.items():
                if any(k in clean_line for k in keywords):
                    current_sec = sec_key
                    found_header = True
                    break
        
        sections[current_sec].append(line)

    return {k: "\n".join(v) for k, v in sections.items()}


def main():
    # 0. CLI Arguments
    parser = argparse.ArgumentParser(description="ACL Science Reporter")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration for the LLM")
    parser.add_argument("--reset", action="store_true", help="Reset processing and ignore checkpoints")
    args = parser.parse_args()

    # 1. Init
    print("Loading AI Engines...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    network = KnowledgeNetwork()
    editor = ScienceEditor(MODEL_PATH, use_gpu=args.gpu)
    publisher = HtmlRenderer(OUTPUT_DIR)
    
    miner = AssetMiner(PAPERS_DIR, OUTPUT_DIR)
    paper_store = {}
    
    # CHECKPOINT LOADING
    # If a checkpoint exists and reset is False, load it
    checkpoint_path = "checkpoint.pkl"
    processed_pids = set()
    
    if os.path.exists(checkpoint_path) and not args.reset:
        try:
            print(f"Found checkpoint: {checkpoint_path}. Attempting to resume...")
            with open(checkpoint_path, "rb") as f:
                saved_state = pickle.load(f)
                paper_store = saved_state.get('paper_store', {})
                # Restore network graph if available
                if 'network_graph' in saved_state:
                    network.graph = saved_state['network_graph']
                processed_pids = set(paper_store.keys())
            print(f"Successfully resumed. {len(processed_pids)} papers already processed.")
        except Exception as e:
            print(f"Error loading checkpoint ({e}). Starting fresh.")
            processed_pids = set()

    # 2. Advanced Ingestion
    print("\n--- Phase 1: Layout-Aware Ingestion ---")
    paper_files = [f for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')]
    
    # Sort files to ensure deterministic order for resuming
    paper_files.sort()
    
    for filename in tqdm(paper_files):
        try:
            data = miner.process_pdf(filename) 
            pid = data['id']
            
            if pid in processed_pids:
                continue

            full_text = data['text']
            
            # B. Smart Chunking
            chunks = smart_chunker(full_text)
            
            # C. Embed (We embed the Abstract + Conclusion for a better "Paper Fingerprint")
            # Critique Fix: text[:1000] was too shallow. Using a composite fingerprint.
            # If 'results' is empty (common in parsing errors), fall back to more intro.
            fingerprint = chunks['intro'][:2000] + "\n" + chunks['results'][-1000:] 
            vector = embedder.encode(fingerprint)
            network.add_paper(pid, vector, metadata={'title': data['title']})
            
            # D. TDMS Extraction (Run on specific sections for better recall)
            # We assume a global set of seen entities for basic resolution
            seen_entities = set(network.graph.nodes())
            
            # Extract METHODS from the Method section
            method_ents = editor.extract_tdms_schema(chunks['methods'], pid)
            for e in method_ents:
                if not isinstance(e, dict) or 'entity' not in e:
                    continue
                # Entity Resolution Step (Section 2.3)
                # Try to map new entity to existing seen entities
                canonical_name = editor.align_entity_canonical(e['entity'], list(seen_entities))
                
                # Pass extracted CSO topics to the graph to enable hierarchical reasoning
                cso_topics = e.get('cso_topics', [])
                # Critique Fix: Increased evidence snippet from 200 to 500 chars for better context in the UI
                network.add_tdms_entity(pid, canonical_name, e.get('type', 'UNKNOWN'), e.get('relation', 'RELATED'), chunks['methods'][:500], cso_topics=cso_topics)
                seen_entities.add(canonical_name)
                
            # Extract METRICS/DATASETS from Results section
            result_ents = editor.extract_tdms_schema(chunks['results'], pid)
            for e in result_ents:
                if not isinstance(e, dict) or 'entity' not in e:
                    continue
                canonical_name = editor.align_entity_canonical(e['entity'], list(seen_entities))
                cso_topics = e.get('cso_topics', [])
                # Critique Fix: Increased evidence snippet
                network.add_tdms_entity(pid, canonical_name, e.get('type', 'UNKNOWN'), e.get('relation', 'RELATED'), chunks['results'][:500], cso_topics=cso_topics)
                seen_entities.add(canonical_name)

            paper_store[pid] = {**data, 'chunks': chunks}
            processed_pids.add(pid)
            
            # CHECKPOINT: Save every 5 papers 
            if len(paper_store) % 5 == 0:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump({
                        "paper_store": paper_store,
                        "network_graph": network.graph
                    }, f)
                    
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
            continue

    # Final save after Ingestion completion
    with open(checkpoint_path, "wb") as f:
        pickle.dump({
            "paper_store": paper_store,
            "network_graph": network.graph
        }, f)

    # 3. Citation Logic & Graph Calculation
    print("\n--- Phase 2: Graph Reasoning ---")
    
    # Implemented Citation Linking (Section 2.2)
    # Scans full text for mentions of other paper titles to create semantic edges
    print("Linking papers via Citation Intent Analysis...")
    paper_titles = {pid: data['title'] for pid, data in paper_store.items()}
    
    # Pre-calculate titles for faster lookup
    title_lookup = [(pid, title, title.lower().replace("_", " ").strip()) 
                   for pid, title in paper_titles.items() 
                   if len(title) > 10]

    for source_pid, source_data in tqdm(paper_store.items(), desc="Analyzing Citations"):
        source_text = source_data['text'].lower()
        
        for target_pid, original_title, t_clean in title_lookup:
            if source_pid == target_pid: continue
            
            if t_clean in source_text:
                idx = source_text.find(t_clean)
                start = max(0, idx - 250)
                end = min(len(source_text), idx + 250 + len(t_clean))
                context_snippet = source_data['text'][start:end]
                
                if re.search(r'\d{4}\.\s*$', context_snippet.strip()) or "References" in source_data['text'][max(0, idx-100):idx]:
                     network.add_citation_link(source_pid, target_pid, intent="BACKGROUND")
                     continue

                intent = editor.classify_citation_intent(context_snippet, original_title)
                
                network.add_citation_link(source_pid, target_pid, intent=intent)
    
    print("Computing PageRank & Communities...")
    network.compute_importance()
    communities = network.detect_communities()
    
    print(f"Detected {len(communities)} Distinct Research Clusters.")

    # 4. Synthesis
    print("\n--- Phase 3: Generative Synthesis ---")
    
    toc = []
    
    for cid, nodes in communities.items():
        cluster_papers = [n for n in nodes if network.graph.nodes[n].get('type') == 'PAPER']
        if not cluster_papers: continue
        
        cluster_titles = [paper_store[p]['title'] for p in cluster_papers]
        theme_title = editor.summarize_community(cluster_titles)
        print(f"Synthesizing Community {cid}: {theme_title}...")
        
        viz_md = publisher.generate_tdms_viz(cluster_papers, network)
        
        context_str = ""
        for pid in cluster_papers:
            neighbors = network.graph.neighbors(pid)
            feats = [f"{network.graph.nodes[n]['label']} ({network.graph.nodes[n]['type']})" 
                     for n in neighbors if network.graph.nodes[n]['type'] != 'PAPER']
            
            competitors = network.find_competitors(pid)
            comp_list = []
            for c in competitors:
                c_title = paper_store[c]['title']
                path_desc = network.analyze_conflict_path(pid, c)
                if path_desc:
                    comp_list.append(f"{c_title} (Conflict Path: {path_desc})")
                else:
                    comp_list.append(c_title)
            
            comp_str = ", ".join(comp_list) if comp_list else "None detected"

            lineage = network.trace_lineage(pid)
            foundational_str = ", ".join([paper_store[p]['title'] for p in lineage['foundational']])
            derivative_str = ", ".join([paper_store[p]['title'] for p in lineage['derivative']])
            
            seed_entities = [n for n in neighbors if network.graph.nodes[n]['type'] in ['METHOD', 'TASK']]
            associative_matches = network.run_ppr_search(seed_entities)
            related_str = ", ".join([paper_store[p]['title'] for p in associative_matches if p != pid][:3])

            similar_papers = network.find_similar_papers(pid, threshold=0.4)
            redundancy_str = ", ".join([paper_store[p]['title'] for p in similar_papers[:2]]) if similar_papers else "None"

            p_title = paper_store[pid]['title']
            p_results = paper_store[pid]['chunks']['results'][:1500]
            
            context_str += f"\n### Paper: {p_title}\n"
            context_str += f"Key Entities: {', '.join(feats)}\n"
            context_str += f"Potential Competitors: {comp_str}\n"
            context_str += f"Lineage: Builds on [{foundational_str}]; Influenced [{derivative_str}]\n"
            context_str += f"Structurally Related (Non-obvious): {related_str}\n" 
            context_str += f"Redundancy Risk (Same Story): {redundancy_str}\n"
            context_str += f"Experimental Results: {p_results}\n"

        report = editor.generate_contrastive_summary(cluster_papers, context_str)
        
        cluster_images = []
        for pid in cluster_papers:
            if pid in paper_store and 'images' in paper_store[pid]:
                cluster_images.extend(paper_store[pid]['images'])
        
        fname = f"community_{cid}.html"
        publisher.render_article(
            fname, 
            f"{theme_title} (Cluster {cid})", 
            report['body'], 
            cluster_images, 
            cluster_graph_md=viz_md
        )
        toc.append({'title': f"{theme_title}", 'link': fname})

    publisher.generate_index(toc)
    print("Processing Complete.")

if __name__ == "__main__":
    main()