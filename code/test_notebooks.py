import os
import subprocess
import tempfile
import nbformat


def run_ipynb(path):
    dirname, basename = os.path.split(path)
    os.chdir(dirname)
    error_cells = []
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["nbconvert", "--to", "notebook", "--execute", "--output",
                fout.name, path]
        subprocess.check_call(args)
        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    for cell in nb.cells:
        for output in cell["outputs"]:
            if output.output_type == "error":
                error_cells.append(output)

    return nb, errors

    def test_appendix_g_tensorflow_basics():
        ipynb, errors = _notebook_run('appendix_g_tensorflow-basics.ipynb')
        assert errors == []
