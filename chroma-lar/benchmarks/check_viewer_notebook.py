"""Execute the viewer notebook and exercise its widget callbacks headlessly.

Requires nbclient and a Python Jupyter kernel with the CUDA dependencies.
This checks the kernel/widget model; it does not measure a browser or network.
"""

import argparse
import hashlib
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", default="trichroma-gpu")
    parser.add_argument("--all-detectors", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / "notebooks/detector_viewer.ipynb"
    notebook = nbformat.read(source, as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell("""assert len(controls.children) == 3
assert bytes(controls.children[1].value).startswith(b"\\x89PNG")
assert frame.rays == 2_500_000
assert frame.image.shape == (625, 1000, 3)
assert 45_000 <= example.metadata["sensor_instances"] <= 55_000
initial_geometry_metadata = dict(example.metadata)
import json
print("VIEWER_INITIAL_GEOMETRY="+json.dumps(initial_geometry_metadata))
controls.children[0].children[3].value = 125_003
assert viewer.rays == 125_003
assert "125,003 camera rays" in controls.children[2].value
controls.children[0].children[3].value = 2_500_000
controls.children[0].children[0].value = 36.
"""))
    notebook.cells.append(nbformat.v4.new_code_cell("""import asyncio
await asyncio.sleep(.5)
assert "2,500,000 camera rays" in controls.children[2].value
viewer.close()
print("Widget, preview callback and full-frame callback passed")"""))
    if args.all_detectors:
        notebook.cells.append(
            nbformat.v4.new_code_cell("""for name in ("reflect3wires", "pixelTPC", "pixelPads"):
    selection.value = name
    load_button.click()
    assert example.metadata["name"] == name
    assert controls is viewer._widget
    assert "2,500,000 camera rays" in controls.children[2].value
    assert bytes(controls.children[1].value).startswith(b"\\x89PNG")
viewer.close()
print("All detector-selection callbacks rendered successfully")""")
        )
    NotebookClient(
        notebook, timeout=180, kernel_name=args.kernel, resources={"metadata": {"path": str(root)}}
    ).execute()
    initial_geometry = None
    for cell in notebook.cells:
        for output in cell.get("outputs", ()):
            for line in output.get("text", "").splitlines():
                if line.startswith("VIEWER_INITIAL_GEOMETRY="):
                    initial_geometry = json.loads(line.split("=", 1)[1])
    assert initial_geometry is not None
    report = dict(
        cells=len(notebook.cells),
        executed_cells=sum(c.cell_type == "code" for c in notebook.cells),
        widget_created=True,
        initial_png_verified=True,
        preview_callback=True,
        scheduled_full_frame=True,
        browser_painting_tested=False,
        detector_selection_callbacks=(
            ["reflect3wires", "pixelTPC", "pixelPads"] if args.all_detectors else []
        ),
        ray_count_control=True,
        detector=initial_geometry,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
