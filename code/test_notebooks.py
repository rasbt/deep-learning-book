import os
import subprocess
import tempfile
import watermark
import nbformat


def run_ipynb(path):
    error_cells = []
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["python", "-m", "nbconvert", "--to",
                "notebook", "--execute", "--output",
                fout.name, path]
        print(" ".join(args))
        subprocess.check_call(args)
        fout.seek(0)

        # 'byte' vs str issues when running the following in Py 3.5
        nb = nbformat.read(fout, nbformat.current_nbformat)

    for cell in nb.cells:
        for output in cell["outputs"]:
            if output.output_type == "error":
                error_cells.append(output)

    return nb, errors


def test_appendix_g_tensorflow_basics():
    this_dir = os.path.dirname(os.path.abspath(__file__))

    ipynb, errors = run_ipynb(os.path.join(this_dir,
                              'appendix_g_tensorflow-basics.ipynb'))
    assert errors == []
