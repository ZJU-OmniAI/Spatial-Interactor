#!/usr/bin/env python3
"""Export web assets from the paper without modifying the paper sources."""

import argparse
from collections import deque
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
from PIL import Image


def remove_border_white(image, threshold=248):
    """Make only edge-connected white canvas pixels transparent.

    White fills enclosed by colored borders remain intact, while the page
    canvas around a figure can blend with the homepage section background.
    """
    rgba = image.convert("RGBA")
    width, height = rgba.size
    pixels = rgba.load()
    queue = deque()
    visited = bytearray(width * height)

    def is_canvas_white(x, y):
        red, green, blue, alpha = pixels[x, y]
        return (
            alpha > 0
            and red >= threshold
            and green >= threshold
            and blue >= threshold
        )

    def enqueue(x, y):
        index = y * width + x
        if not visited[index] and is_canvas_white(x, y):
            visited[index] = 1
            queue.append((x, y))

    for x in range(width):
        enqueue(x, 0)
        enqueue(x, height - 1)
    for y in range(1, height - 1):
        enqueue(0, y)
        enqueue(width - 1, y)

    while queue:
        x, y = queue.popleft()
        red, green, blue, _ = pixels[x, y]
        pixels[x, y] = (red, green, blue, 0)
        if x:
            enqueue(x - 1, y)
        if x + 1 < width:
            enqueue(x + 1, y)
        if y:
            enqueue(x, y - 1)
        if y + 1 < height:
            enqueue(x, y + 1)

    return rgba


def prepare_web_image(image):
    # Preserve source alpha and remove only the outer white page canvas.
    return remove_border_white(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, default=Path(__file__).resolve().parents[2] / "Spatial-Interactor-arXiv")
    parser.add_argument("--figures", nargs="+", help="Export only these figure names and refresh the paper PDF.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    paper = args.paper_dir
    figures = {
        "diagnostics": "fig1_state_transition_diagnostics_landscape.pdf",
        "curriculum": "source3_print_visibility_300dpi.pdf",
        "curriculum-ablation": "experiment_curriculum_progression.pdf",
        "training-dynamics": "main_training_dynamics_manual_mockup.pdf",
        "temporal-evidence": "experiment_frame_budget_vsi_route_preview.pdf",
        "frame-order": "vsti_shuffle_iopd_current_fig1_exact.pdf",
        "local-case": "fig7_satreal_attention_paper_layout_direct_preview.pdf",
        "trajectory-case": "source5_compact_300dpi.pdf",
    }
    figures = {name: paper / "Figures" / source for name, source in figures.items()}
    updated = paper / "fig7_manual_preview" / "updated_figures"
    figures.update({
        "overview": updated / "fig1_overview.png",
        "paradigm": updated / "fig3_spatial_paradigms.png",
        "construction": updated / "fig4_dataset_construction.png",
    })
    if args.figures:
        unknown = set(args.figures) - figures.keys()
        if unknown:
            parser.error(f"Unknown figures: {', '.join(sorted(unknown))}")
        figures = {name: figures[name] for name in args.figures}
    for name, source in figures.items():
        if source.suffix.lower() == ".pdf":
            content = subprocess.check_output([
                "pdftocairo", "-singlefile", "-scale-to", "2600", "-png",
                "-transp", str(source), "-",
            ])
            image = prepare_web_image(Image.open(BytesIO(content)))
        else:
            with Image.open(source) as original:
                original.thumbnail((2600, 2600), Image.Resampling.LANCZOS)
                # The illustrated overview has an intentional paper texture.
                image = original.convert("RGBA") if name == "overview" else prepare_web_image(original)
        image.save(assets / f"{name}.webp", quality=96, method=6)
        print(f"{name}: {image.width} x {image.height}", flush=True)
    if args.figures:
        shutil.copy2(paper / "fig7_manual_preview" / "paper.pdf", assets / "paper.pdf")
        print("Selected figures and current paper PDF exported.")
        return
    with Image.open(paper / "Figures" / "fig5_opd_300dpi.png") as image:
        prepare_web_image(image).save(assets / "opd.webp", quality=96, method=6)
    for case in [f"E{index:02d}" for index in range(1, 36)]:
        with Image.open(paper / "appendix_artifacts" / "task_showcase_media" / f"{case}.png") as image:
            image.thumbnail((2000, 1400), Image.Resampling.LANCZOS)
            prepare_web_image(image).save(assets / f"{case}.webp", quality=96, method=6)
    with Image.open(paper / "others" / "ZJU.png") as image:
        image.thumbnail((192, 192), Image.Resampling.LANCZOS)
        image.save(assets / "zju.png")
    shutil.copy2(paper / "fig7_manual_preview" / "paper.pdf", assets / "paper.pdf")
    print("Paper and all figure assets exported.")


if __name__ == "__main__":
    main()
