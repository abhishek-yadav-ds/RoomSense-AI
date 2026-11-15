"""
Accurate Notebook Renderer:
Executes cells with full Matplotlib figure capture on plt.show(),
embedding base64-encoded PNGs into the cell display_data outputs.
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

def execute_and_embed(nb_path):
    print(f"\nProcessing {nb_path}...")
    nb = nbf.read(nb_path, as_version=4)
    
    # Track outputs
    captured_images = 0
    
    # Execution namespace
    root_dir = os.path.abspath('.')
    exec_scope = {
        '__file__': os.path.abspath(nb_path),
        '__name__': '__main__'
    }
    
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            code = cell.source
            cell.outputs = []
            new_outputs = []
            
            # Strip IPython magics for in-process execution
            lines = []
            for l in code.splitlines():
                if l.strip().startswith('%'):
                    continue
                lines.append(l)
            exec_code = '\n'.join(lines)
            
            # Monkey-patched plt.show
            def capture_show(*args, **kwargs):
                nonlocal captured_images
                if plt.get_fignums():
                    fig = plt.gcf()
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode('utf-8')
                    new_outputs.append(nbf.v4.new_output(
                        output_type='display_data',
                        data={
                            'image/png': b64,
                            'text/plain': f"<Figure size {fig.get_size_inches()} with {len(fig.axes)} Axes>"
                        },
                        metadata={}
                    ))
                    captured_images += 1
                    plt.close(fig)
                    
            plt.show = capture_show
            
            # Custom display
            def custom_display(*args):
                for a in args:
                    if isinstance(a, pd.DataFrame):
                        new_outputs.append(nbf.v4.new_output(
                            output_type='execute_result',
                            execution_count=idx,
                            data={
                                'text/html': a._repr_html_(),
                                'text/plain': str(a)
                            },
                            metadata={}
                        ))
                    else:
                        print(a)
            exec_scope['display'] = custom_display
            
            stdout_stream = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout_stream
            err_str = None
            
            try:
                exec(exec_code, exec_scope)
                if plt.get_fignums():
                    capture_show()
            except Exception as e:
                err_str = traceback.format_exc()
            finally:
                sys.stdout = old_stdout
                
            stdout_text = stdout_stream.getvalue()
            if stdout_text:
                cell.outputs.append(nbf.v4.new_output(
                    output_type='stream',
                    name='stdout',
                    text=stdout_text
                ))
                
            cell.outputs.extend(new_outputs)
            
            if err_str:
                print(f"Error in cell {idx} of {nb_path}:\n{err_str}")
                cell.outputs.append(nbf.v4.new_output(
                    output_type='error',
                    ename='CellError',
                    evalue=err_str.splitlines()[-1],
                    traceback=err_str.splitlines()
                ))
                break
                
            cell.execution_count = idx + 1
            
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Done {nb_path}: {captured_images} figures embedded as display_data.")

if __name__ == '__main__':
    for path in [
        'notebooks/01_EDA_Preprocessing.ipynb',
        'notebooks/02_Feature_Engineering_Selection.ipynb',
        'notebooks/03_ML_DL_Training.ipynb'
    ]:
        execute_and_embed(path)

