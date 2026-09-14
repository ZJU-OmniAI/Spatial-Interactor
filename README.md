# Spatial-Interactor Project Website

Open `index.html` in a browser. No installation, build step, server, or network
connection is needed to view the site.

## Contents

- `index.html`: paper-ordered project narrative, figures, results, and citation.
- `data.js`: five result tables transcribed from the paper.
- `examples.js`: all 35 selected appendix examples across L1, L2, and L3.
- `styles.css` / `app.js`: responsive layout, animated example browser, sortable
  tables, cursor-responsive figure previews, CSV export, reading progress, and
  citation copying.
- `media.js`: zoomable, keyboard-accessible figure gallery.
- `assets/`: local figures, example images, fonts, icons, and the paper PDF.

All 11 main-paper figures are included. The Analysis section contains curriculum
ablation, OPD training dynamics, frame-budget/trace comparisons, and frame-order
sensitivity. WalkerBench and ESI-Bench include every subtask from Tables 4-5.
The three paper result tables are shown together in one continuous results
section, with all reported rows expanded by default.

Content follows the current paper. Multi-step object operations are included in
L1, not L3. Scores are reported paper results, not an independent evaluation.
The table export contains the currently selected rows, in their displayed order.
The Code, Dataset, and Models controls in the masthead point to the public
GitHub and Hugging Face releases. They are intentionally not repeated in the
body of the page.

## Publish

Upload `index.html`, `styles.css`, `app.js`, `media.js`, `data.js`, `examples.js`, `.nojekyll`, and `assets/`
to a static host or the root of a GitHub Pages publishing branch. All paths are
relative, so the site also works under a project subdirectory.

The deployed project URL is
`https://zju-omniai.github.io/Spatial-Interactor/`. The PDF is a separate
download and is not fetched during normal page viewing.

Only public author/affiliation/contact information is included. Do not upload
the source PPT, credential files, research logs, or unrelated working folders.

## Refresh Paper Assets

With Pillow and Poppler (`pdftoppm`) installed:

```bash
python3 scripts/build_assets.py --paper-dir ../Spatial-Interactor-arXiv
```

This updates website assets only. It reads the PDF/figures in the paper folder
without modifying them. To rebuild the appendix example index, run:

```bash
python3 scripts/build_examples.py --index ../AuthorKit27/appendix_artifacts/selected_case_index.json
```

Update `data.js` separately when result tables change.
All raster images preserve their aspect ratio; the figure viewer opens the
larger exported image.

Page transitions respect the browser's reduced-motion preference.

For an HTTP preview:

```bash
python3 scripts/serve.py --port 8080
```

Use `--bind 0.0.0.0` to preview on another device on the same network.

## Checks

With Playwright and Chromium installed:

```bash
node scripts/check_site.cjs
node scripts/check_media.cjs
```

The check opens the local HTML directly, tests desktop/mobile layouts and
interactions, verifies local assets, and writes screenshots to a temporary
directory. `PLAYWRIGHT_PATH` can point to an existing Playwright installation;
`OUTPUT_DIR` can override the screenshot directory. Set `SITE_URL` to check an
HTTP preview instead of the local file.

The visual structure is inspired by the
[ProVisE project page](https://zju-omniai.github.io/ProVisE/).
Third-party asset licenses are documented in `THIRD_PARTY_NOTICES.md`.
