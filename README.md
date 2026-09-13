# Spatial-Interactor Project Website

Open `index.html` in a browser. No installation, build step, server, or network
connection is needed to view the site. Videos load only after pressing play.

## Contents

- `index.html`: project introduction, method, figures, citation, and resource links.
- `data.js`: five result tables and 12 QA examples, transcribed from the paper.
- `styles.css` / `app.js`: responsive layout, animated example browser, sortable
  tables with row/column hover tracking, cursor-responsive figure previews, CSV
  export, reading progress, and citation copying.
- `media.js`: two video demos and a zoomable, keyboard-accessible figure gallery.
- `assets/`: local figures, example images, fonts, icons, and the paper PDF.

All 11 main-paper figures are included. The Analysis section contains curriculum
ablation, OPD training dynamics, frame-budget/trace comparisons, and frame-order
sensitivity. WalkerBench and ESI-Bench include every subtask from Tables 4-5.
The three paper result tables are shown together in one continuous results
section, with all reported rows expanded by default.

Content follows the current paper. Multi-step object operations are included in
L1, not L3. Scores are reported paper results, not an independent evaluation.
The table export contains the currently selected rows, in their displayed order.
The two trajectory videos are illustrative walkthroughs, not recorded model
responses.
The Code, Dataset, and Models controls point to the public GitHub and Hugging
Face releases. The resource section also links each of the four checkpoints
individually.

## Publish

Upload `index.html`, `styles.css`, `app.js`, `media.js`, `data.js`, `.nojekyll`, and `assets/`
to a static host or the root of a GitHub Pages publishing branch. All paths are
relative, so the site also works under a project subdirectory.

The project URL proposed in the paper is
`https://zju-omniai.github.io/Spatial-Interactor/`. This local directory has not
been deployed. Configure the absolute `og:image` URL after the domain is final.
The PDF is a separate download and is not fetched during normal page viewing.

Only public author/affiliation/contact information is included. Do not upload
the source PPT, credential files, research logs, or unrelated working folders.

## Refresh Paper Assets

With Pillow and Poppler (`pdftoppm`) installed:

```bash
python3 scripts/build_assets.py --paper-dir ../Spatial-Interactor-arXiv
```

This updates website assets only. It reads the PDF/figures in the paper folder
without modifying them. Update `data.js` separately when results or QA change.
All raster images preserve their aspect ratio; the figure viewer opens the
larger exported image.

To refresh videos and their posters, with FFmpeg installed:

```bash
python3 scripts/build_videos.py
```

This reads the sibling visualizer folder. Override `--visualizer-dir` when
needed. Video streams are copied without re-encoding;
only MP4 metadata placement is optimized for playback. No videos autoplay, and
playback stops when switching clips, leaving the section, or hiding the page.
Page transitions respect the browser's reduced-motion preference.

For an HTTP preview with video seeking:

```bash
python3 scripts/serve.py --port 8080
```

Use `--bind 0.0.0.0` to preview on another device on the same network. The small
preview server supports byte-range requests; the standard Python file server
does not. Public static hosting should also support HTTP byte ranges.

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
