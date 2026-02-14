import os
import markdown  # Recommended: pip install markdown

class HtmlRenderer:
    """
    Upgraded Object 4: Magazine-style HTML with Graph Visualization
    and LaTeX support via MathJax.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._write_css()

    def render_article(self, filename: str, title: str, body_md: str, images: list, cluster_graph_md: str = ""):
        """
        Renders a full technical analysis with interactive graph visualizations.
        """
        # Proper Markdown conversion to handle LaTeX and structure
        body_html = markdown.markdown(body_md, extensions=['extra', 'smarty'])
        
        gallery_html = ""
        if images:
            gallery_html = "<div class='gallery'>"
            for img in images[:3]:
                gallery_html += f"<figure><img src='{img}' class='article-img'><figcaption>Fig: Extracted Insight</figcaption></figure>"
            gallery_html += "</div>"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <link rel="stylesheet" href="style.css">
            <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
            <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
            </script>
        </head>
        <body>
            <div class="container">
                <nav><a href="index.html">← Back to Research Index</a></nav>
                <header>
                    <h1>{title}</h1>
                    <div class="visual-abstract">
                        <h3>Structural Relationship Map</h3>
                        <div class="mermaid">
                            {cluster_graph_md}
                        </div>
                    </div>
                </header>
                {gallery_html}
                <article class="content">
                    {body_html}
                </article>
            </div>
        </body>
        </html>
        """
        with open(os.path.join(self.output_dir, filename), "w", encoding="utf-8") as f:
            f.write(html_content)

    def generate_tdms_viz(self, cluster_papers: list, network):
        """
        Generates a Mermaid graph showing the TDMS connections.
        Reasoning: Highlights common benchmarks (Datasets) and 
        divergent strategies (Methods).
        """
        mermaid_code = "graph TD;\n"
        seen_nodes = set()
        
        for pid in cluster_papers:
            p_label = network.graph.nodes[pid].get('label', pid)
            mermaid_code += f'  P_{pid}["📄 {p_label}"]\n'
            
            # Get neighbors (Tasks, Methods, Datasets)
            for neighbor in network.graph.neighbors(pid):
                n_type = network.graph.nodes[neighbor].get('type')
                n_label = network.graph.nodes[neighbor].get('label', neighbor)
                
                # Assign icons based on TDMS type
                if n_type == "METHOD":
                    icon = "⚙️"
                elif n_type == "DATASET":
                    icon = "📊"
                elif n_type == "METRIC":
                    icon = "📏"
                elif n_type == "TOPIC":
                    icon = "📚"
                else: 
                    icon = "🎯" # Default/Task

                node_id = f"{n_type}_{neighbor.replace(' ', '_')}"
                
                if node_id not in seen_nodes:
                    mermaid_code += f'  {node_id}["{icon} {n_label}"]\n'
                    seen_nodes.add(node_id)
                
                mermaid_code += f"  P_{pid} --> {node_id};\n"
                
        return mermaid_code

    def generate_index(self, toc: list):
        """
        Generates a main index.html linking to all community reports.
        """
        toc_html = "<ul>"
        for item in toc:
            toc_html += f"<li><a href='{item['link']}'>{item['title']}</a></li>"
        toc_html += "</ul>"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Research Index</title>
            <link rel="stylesheet" href="style.css">
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>Scientific Synthesis Report</h1>
                    <p>Generated on {os.popen('date').read()}</p>
                </header>
                <article class="content">
                    <h2>Table of Contents</h2>
                    {toc_html}
                </article>
            </div>
        </body>
        </html>
        """
        with open(os.path.join(self.output_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(html_content)

    def _write_css(self):
        css = """
        body { font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: auto; padding: 2em; background: #f4f4f9; color: #333; }
        .container { background: white; padding: 3em; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { border-bottom: 2px solid #007acc; padding-bottom: 0.2em; color: #007acc; }
        h2 { margin-top: 2em; color: #444; }
        .visual-abstract { background: #eef; padding: 1em; border-radius: 8px; margin: 2em 0; border: 1px solid #ccd; }
        .gallery { display: flex; gap: 1em; overflow-x: auto; padding: 1em 0; }
        .article-img { max-height: 300px; border: 1px solid #ddd; border-radius: 4px; }
        pre { background: #2d2d2d; color: #ccc; padding: 1em; overflow-x: auto; border-radius: 4px; }
        a { color: #007acc; text-decoration: none; }
        a:hover { text-decoration: underline; }
        """
        with open(os.path.join(self.output_dir, "style.css"), "w", encoding="utf-8") as f:
            f.write(css)