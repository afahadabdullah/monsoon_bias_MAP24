#!/usr/bin/env python3
"""Export selected saved notebook PNGs without executing research code.

Uses only the Python standard library. Cell and output indices are zero-based.
Run from any directory: python /path/to/scripts/export_readme_figures.py
"""

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION = (
    ("GEOS_ML_precip2.ipynb", 43, "cnn_skill.png",
     "Two-month-lead test-set RMSE, correlation, and ML-minus-GEOS differences."),
    ("GEOS_ML_precip2.ipynb", 46, "cnn_mean_lead2.png",
     "Two-month-lead test-set mean precipitation and mean forecast-minus-observation bias."),
    ("GEOS_ML_precip2.ipynb", 40, "cnn_case.png",
     "Saved two-month-lead example: GEOS, CNN-corrected forecast, observations, and biases."),
    ("GEOS_ML_precip3_unet.ipynb", 27, "training_loss.png",
     "Saved training and validation loss for the notebook's deep-supervision experiment."),
)


def main():
    output_dir = ROOT / "figures"
    output_dir.mkdir(exist_ok=True)
    manifest = []
    for filename, cell_index, output_name, caption in SELECTION:
        notebook_path = ROOT / "notebooks" / filename
        notebook = json.loads(notebook_path.read_text())
        outputs = notebook["cells"][cell_index].get("outputs", [])
        output_index, png = next(
            (i, output["data"]["image/png"])
            for i, output in enumerate(outputs)
            if "image/png" in output.get("data", {})
        )
        png = "".join(png) if isinstance(png, list) else png
        raw = base64.b64decode("".join(png.split()), validate=True)
        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Invalid PNG in {filename}, cell {cell_index}")
        (output_dir / output_name).write_bytes(raw)
        manifest.append({
            "figure": f"figures/{output_name}",
            "source_notebook": str(notebook_path.relative_to(ROOT)),
            "cell_index": cell_index,
            "output_index": output_index,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "caption": caption,
            "provenance": "Saved output supplied in precip_SA.tar; not recomputed.",
        })
        print(f"Exported figures/{output_name}")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
