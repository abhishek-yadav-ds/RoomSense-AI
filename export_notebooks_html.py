"""
Utility to export notebooks with inline base64 images to standalone HTML files
so the user can open them in any web browser (Chrome, Edge, Firefox) with 1 click.
"""
import json
import os

def nb_to_html(nb_path, html_path):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    
    title = os.path.basename(nb_path)
    html = [
        '<!DOCTYPE html>',
        '<html><head><meta charset="utf-8">',
        f'<title>{title} — RoomSense AI</title>',
        '<style>',
        '  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 1100px; margin: 30px auto; background: #0f172a; color: #f8fafc; padding: 20px; line-height: 1.6; }',
        '  h1, h2, h3 { color: #38bdf8; border-bottom: 1px solid #334155; padding-bottom: 8px; }',
        '  .cell { background: #1e293b; margin: 20px 0; padding: 18px; border-radius: 10px; border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }',
        '  .markdown { background: #0b1329; border-left: 4px solid #38bdf8; padding: 14px 20px; }',
        '  pre { background: #0d1117; padding: 14px; border-radius: 8px; overflow-x: auto; color: #7dd3fc; border: 1px solid #30363d; font-family: "Consolas", monospace; font-size: 13px; }',
        '  .output-stream { background: #161b22; color: #a5f3fc; border-left: 3px solid #10b981; }',
        '  img { max-width: 100%; border-radius: 8px; margin: 15px auto; display: block; box-shadow: 0 4px 15px rgba(0,0,0,0.6); border: 1px solid #475569; background: #ffffff; padding: 4px; }',
        '  .badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: bold; background: #6366f1; color: white; margin-bottom: 10px; }',
        '</style></head><body>',
        f'<h1>📊 {title}</h1>',
        '<p style="color: #94a3b8;">Pre-rendered standalone view. All code cells and inline visualization outputs are embedded directly below.</p>'
    ]
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            md_text = ''.join(cell['source']).replace('\n', '<br>')
            html.append(f'<div class="cell markdown"><div>{md_text}</div></div>')
        elif cell['cell_type'] == 'code':
            code_src = ''.join(cell['source'])
            html.append(f'<div class="cell"><span class="badge">In [{i}]</span><pre><code>{code_src}</code></pre>')
            for out in cell.get('outputs', []):
                if 'data' in out and 'image/png' in out['data']:
                    b64 = out['data']['image/png'].replace('\n', '')
                    html.append(f'<img src="data:image/png;base64,{b64}" alt="Inline Visualization" />')
                elif 'text' in out:
                    out_text = ''.join(out['text'])
                    html.append(f'<pre class="output-stream"><code>{out_text}</code></pre>')
            html.append('</div>')
            
    html.append('</body></html>')
    
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"Generated: {html_path}")

if __name__ == '__main__':
    nb_to_html('notebooks/01_EDA_Preprocessing.ipynb', 'outputs/reports/01_EDA_Preprocessing_View.html')
    nb_to_html('notebooks/02_Feature_Engineering_Selection.ipynb', 'outputs/reports/02_Feature_Engineering_Selection_View.html')
    nb_to_html('notebooks/03_ML_DL_Training.ipynb', 'outputs/reports/03_ML_DL_Training_View.html')
    print("All HTML notebook views generated successfully!")

