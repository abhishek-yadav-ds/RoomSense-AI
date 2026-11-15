"""
Direct In-Process Notebook Executor
Executes notebook cells sequentially in an isolated namespace,
capturing stdout and return values and writing them into the .ipynb JSON format.
Bypasses ZMQ socket communication issues in sandboxed environments.
"""

import nbformat as nbf
import io
import sys
import os
import traceback

def execute_notebook_direct(nb_path):
    print(f"Executing: {nb_path}...")
    nb = nbf.read(nb_path, as_version=4)
    
    # Execution namespace
    exec_globals = {
        '__file__': os.path.abspath(nb_path),
        '__name__': '__main__'
    }
    
    # Save original working dir
    original_cwd = os.getcwd()
    nb_dir = os.path.abspath(os.path.dirname(nb_path))
    os.chdir(nb_dir)
    
    execution_count = 1
    
    try:
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                cell_code = cell.source
                stdout_capture = io.StringIO()
                
                # Mock display function
                def display(*args):
                    for a in args:
                        print(a)
                exec_globals['display'] = display
                
                old_stdout = sys.stdout
                sys.stdout = stdout_capture
                error_occurred = False
                err_msg = ""
                
                try:
                    # Execute code cell
                    exec(cell_code, exec_globals)
                except Exception as e:
                    error_occurred = True
                    err_msg = traceback.format_exc()
                finally:
                    sys.stdout = old_stdout
                
                output_text = stdout_capture.getvalue()
                cell.outputs = []
                
                if output_text:
                    cell.outputs.append(nbf.v4.new_output(
                        output_type='stream',
                        name='stdout',
                        text=output_text
                    ))
                    
                if error_occurred:
                    print(f"Error in cell {idx} of {nb_path}:\n{err_msg}")
                    cell.outputs.append(nbf.v4.new_output(
                        output_type='error',
                        ename='ExecutionError',
                        evalue=str(err_msg.splitlines()[-1]),
                        traceback=err_msg.splitlines()
                    ))
                    break
                    
                cell.execution_count = execution_count
                execution_count += 1
                
        # Write back executed notebook
        with open(os.path.basename(nb_path), 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
            
        print(f"Finished executing {nb_path} ({execution_count - 1} code cells).")
        
    finally:
        os.chdir(original_cwd)

if __name__ == '__main__':
    for nb_file in [
        'notebooks/01_EDA_Preprocessing.ipynb',
        'notebooks/02_Feature_Engineering_Selection.ipynb',
        'notebooks/03_ML_DL_Training.ipynb'
    ]:
        execute_notebook_direct(nb_file)

