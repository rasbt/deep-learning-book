import os
import subprocess
import tempfile
import jupyter_nbformat


def run_ipynb(path):
    error_cells = []
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["python", "-m", "nbconvert", "--to", "notebook", "--execute", "--output",
                fout.name, path]
        print(" ".join(args))
        subprocess.check_call(args)
        fout.seek(1)
        nb = jupyter_nbformat.read(fout, jupyter_nbformat.current_nbformat)

    for cell in nb.cells:
        for output in cell["outputs"]:
            if output.output_type == "error":
                error_cells.append(output)

    return nb, errors

def test_appendix_g_tensorflow_basics():
    ipynb, errors = run_ipynb('code/appendix_g_tensorflow-basics.ipynb')
    assert errors == []
