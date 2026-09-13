#!/usr/bin/env python3
"""Export web assets from the paper without modifying the paper sources."""

import argparse
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
from PIL import Image


def rgb_on_white(image):
    # Paper PNGs use transparency between panels; flatten it onto page white.
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    return Image.alpha_composite(background, rgba).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, default=Path(__file__).resolve().parents[2] / "Spatial-Interactor-arXiv")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    paper = args.paper_dir
    figures = {
        "diagnostics": "fig1_state_transition_diagnostics_landscape.pdf",
        "paradigm": "fig2_spatial_learning_paradigms_landscape.pdf",
        "construction": "fig3_lsi_construction_landscape.pdf",
        "curriculum": "source3_print_visibility_300dpi.pdf",
        "curriculum-ablation": "experiment_curriculum_progression.pdf",
        "training-dynamics": "main_training_dynamics_manual_mockup.pdf",
        "temporal-evidence": "experiment_frame_budget_vsi_route_preview.pdf",
        "frame-order": "vsti_shuffle_iopd_current_fig1_exact.pdf",
        "local-case": "fig7_satreal_attention_paper_layout_direct_preview.pdf",
        "trajectory-case": "source5_compact_300dpi.pdf",
    }
    for name, source in figures.items():
        content = subprocess.check_output([
            "pdftoppm", "-singlefile", "-scale-to", "2600", "-png",
            str(paper / "Figures" / source),
        ])
        image = rgb_on_white(Image.open(BytesIO(content)))
        image.save(assets / f"{name}.webp", quality=94, method=6)
        print(f"{name}: {image.width} x {image.height}", flush=True)
    with Image.open(paper / "Figures" / "fig5_opd_300dpi.png") as image:
        rgb_on_white(image).save(assets / "opd.webp", quality=96, method=6)
    for case in ["E02", "E03", "E09", "E12", "E13", "E22", "E23", "E28", "E29", "E32", "E33", "E34"]:
        with Image.open(paper / "appendix_artifacts" / "task_showcase_media" / f"{case}.png") as image:
            image.thumbnail((2000, 1400), Image.Resampling.LANCZOS)
            rgb_on_white(image).save(assets / f"{case}.webp", quality=94, method=6)
    with Image.open(paper / "others" / "ZJU.png") as image:
        image.thumbnail((192, 192), Image.Resampling.LANCZOS)
        image.save(assets / "zju.png")
    shutil.copy2(paper / "fig7_manual_preview" / "paper.pdf", assets / "paper.pdf")
    print("Paper and all figure assets exported.")


if __name__ == "__main__":
    main()
