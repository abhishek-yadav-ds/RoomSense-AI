"""
Master Notebook Runner & Inline Visual Output Embedder
======================================================
Executes notebook cells in-process, captures:
1. Standard text stdout
2. Rich Matplotlib / Seaborn figures via plt.show() as base64 image/png display_data
3. DataFrame display representations
Embeds them directly inside the .ipynb JSON format so graphs appear
immediately beneath each cell when opened in Jupyter Notebook, VS Code, or GitHub.
"""

import nbformat as nbf
import os
import io
import sys
import base64
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def run_and_embed_notebook(nb_path):
    print(f"\n==================================================")
    print(f"Executing & Embedding Visuals for: {nb_path}")
    print(f"==================================================")
    
    nb = nbf.read(nb_path, as_version=4)
    
    original_cwd = os.getcwd()
    nb_dir = os.path.abspath(os.path.dirname(nb_path))
    os.chdir(nb_dir)
    
    exec_globals = {
        '__file__': os.path.abspath(nb_path),
        '__name__': '__main__'
    }
    
    execution_counter = 1
    
    try:
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                code = cell.source
                cell.outputs = []
                current_outputs = []
                
                # Filter out %matplotlib inline for in-process exec
                clean_lines = []
                for line in code.splitlines():
                    if line.strip().startswith('%'):
                        continue
                    clean_lines.append(line)
                clean_code = '\n'.join(clean_lines)
                
                # Monkey-patched plt.show() to capture figures
                def custom_show(*args, **kwargs):
                    if plt.get_fignums():
                        fig = plt.gcf()
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                        buf.seek(0)
                        b64_data = base64.b64encode(buf.read()).decode('utf-8')
                        current_outputs.append(nbf.v4.new_output(
                            output_type='display_data',
                            data={
                                'image/png': b64_data,
                                'text/plain': f"<Figure size {fig.get_size_inches()} with {len(fig.axes)} Axes>"
                            },
                            metadata={}
                        ))
                        plt.close(fig)
                        
                # Custom display function
                def custom_display(*args):
                    for a in args:
                        if isinstance(a, pd.DataFrame):
                            current_outputs.append(nbf.v4.new_output(
                                output_type='execute_result',
                                execution_count=execution_counter,
                                data={
                                    'text/html': a._repr_html_(),
                                    'text/plain': str(a)
                                },
                                metadata={}
                            ))
                        else:
                            print(a)
                            
                exec_globals['display'] = custom_display
                plt.show = custom_show
                
                stdout_buf = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = stdout_buf
                err = None
                
                try:
                    exec(clean_code, exec_globals)
                    
                    # Capture any remaining open figure
                    if plt.get_fignums():
                        custom_show()
                        
                except Exception as e:
                    err = traceback.format_exc()
                finally:
                    sys.stdout = old_stdout
                    
                txt = stdout_buf.getvalue()
                if txt:
                    # Insert stdout before or after display data
                    cell.outputs.append(nbf.v4.new_output(
                        output_type='stream',
                        name='stdout',
                        text=txt
                    ))
                    
                cell.outputs.extend(current_outputs)
                
                if err:
                    print(f"Error in cell {idx} of {nb_path}:\n{err}")
                    cell.outputs.append(nbf.v4.new_output(
                        output_type='error',
                        ename='ExecutionError',
                        evalue=err.splitlines()[-1],
                        traceback=err.splitlines()
                    ))
                    break
                    
                cell.execution_count = execution_counter
                execution_counter += 1
                
        # Save back the notebook with embedded visual outputs
        with open(os.path.basename(nb_path), 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
            
        print(f"SUCCESS: {nb_path} executed with {execution_counter - 1} cells and embedded visuals.")
        
    finally:
        os.chdir(original_cwd)

if __name__ == '__main__':
    # 1. Build all 3 notebooks from generators
    import build_and_run_notebook_01
    import build_notebook_02
    import build_notebook_03
    
    build_and_run_notebook_01.build_notebook_01()
    with open('notebooks/01_EDA_Preprocessing.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(build_and_run_notebook_01.build_notebook_01(), f)
        
    build_notebook_02.build_notebook_02()
    with open('notebooks/02_Feature_Engineering_Selection.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(build_notebook_02.build_notebook_02(), f)
        
    build_notebook_03.build_notebook_03()
    with open('notebooks/03_ML_DL_Training.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(build_notebook_03.build_notebook_03(), f)
        
    print("Base notebook files reconstructed.")
    
    # 2. Execute and embed visuals into all 3
    for path in [
        'notebooks/01_EDA_Preprocessing.ipynb',
        'notebooks/02_Feature_Engineering_Selection.ipynb',
        'notebooks/03_ML_DL_Training.ipynb'
    ]:
        run_and_embed_notebook(path)

