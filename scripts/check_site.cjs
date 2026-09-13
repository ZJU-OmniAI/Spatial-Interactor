/* Local-file smoke test. Playwright is a development dependency, not a runtime. */
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const { pathToFileURL, fileURLToPath } = require("node:url");
const { chromium } = require(process.env.PLAYWRIGHT_PATH || "playwright");

const root = path.resolve(__dirname, "..");
const output =
  process.env.OUTPUT_DIR ||
  path.join(os.tmpdir(), "spatial-interactor-site-check");
const errors = [];

async function assertImage(page, selector) {
  await page.locator(selector).evaluate(async (image) => {
    image.loading = "eager";
    await image.decode();
  });
  assert(
    await page.locator(selector).evaluate((image) => image.naturalWidth > 0),
  );
}

async function checkLocalLinks(page) {
  const urls = await page.evaluate(() =>
    [...document.querySelectorAll("[src],link[href],a[href]")]
      .map((node) => node.src || node.href)
      .filter((url) => url?.startsWith("file:")),
  );
  for (const url of new Set(urls)) {
    await fs.access(fileURLToPath(new URL(url)));
  }
  const missing = await page.evaluate(() =>
    [...document.querySelectorAll('a[href^="#"]')]
      .filter(
        (link) => link.hash && !document.getElementById(link.hash.slice(1)),
      )
      .map((link) => link.hash),
  );
  assert.deepEqual(missing, []);
}

async function loadAllImages(page) {
  await page.evaluate(async () => {
    await Promise.all(
      [...document.images]
        .filter((image) => image.hasAttribute("src"))
        .map(async (image) => {
          image.loading = "eager";
          await image.decode();
        }),
    );
    await document.fonts.ready;
  });
}

async function main() {
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox"],
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
      reducedMotion: "reduce",
    });
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("requestfailed", (request) =>
      errors.push(`${request.url()}: ${request.failure()?.errorText}`),
    );
    await page.goto(
      process.env.SITE_URL || pathToFileURL(path.join(root, "index.html")).href,
    );
    await checkLocalLinks(page);
    await loadAllImages(page);
    assert.equal(await page.locator("#results-main-table tbody tr").count(), 18);
    assert.equal(
      await page.locator("#results-generalization-table tbody tr").count(),
      12,
    );
    assert.equal(await page.locator("#results-ablation-table tbody tr").count(), 5);
    const figureIds = await page
      .locator("[data-paper-figure]")
      .evaluateAll((nodes) =>
        nodes
          .map((node) => Number(node.dataset.paperFigure))
          .sort((a, b) => a - b),
      );
    assert.deepEqual(
      figureIds,
      Array.from({ length: 11 }, (_, index) => index + 1),
    );
    for (const id of figureIds) {
      assert(await page.locator(`[data-paper-figure="${id}"]`).isVisible());
      await assertImage(page, `[data-paper-figure="${id}"] > img`);
    }
    const tableIds = await page
      .locator("[data-paper-table]")
      .evaluateAll((nodes) =>
        [...new Set(nodes.map((node) => Number(node.dataset.paperTable)))].sort(
          (a, b) => a - b,
        ),
      );
    assert.deepEqual(tableIds, [1, 2, 3, 4, 5]);
    assert.equal(await page.locator("#walker-table tbody tr").count(), 2);
    assert.equal(await page.locator("#esi-table tbody tr").count(), 2);
    assert.equal(await page.locator("#walker-table thead th").count(), 8);
    assert.equal(await page.locator("#esi-table thead th").count(), 11);
    for (const [key, expected] of [
      [
        "walker",
        [
          ["5.0", "0.0", "25.0", "5.0", "0.0", "7.0", "17.71"],
          ["20.0", "0.0", "40.0", "10.0", "0.0", "14.0", "6.52"],
        ],
      ],
      [
        "esi",
        [
          [
            "33.3",
            "26.7",
            "56.7",
            "23.3",
            "50.0",
            "20.0",
            "26.7",
            "26.7",
            "32.9",
            "12.34",
          ],
          [
            "40.0",
            "36.7",
            "63.3",
            "33.3",
            "56.7",
            "33.3",
            "33.3",
            "33.3",
            "41.2",
            "10.55",
          ],
        ],
      ],
    ]) {
      const actual = await page
        .locator(`#${key}-table tbody tr`)
        .evaluateAll((rows) =>
          rows.map((row) =>
            [...row.querySelectorAll("td")].map((cell) => cell.textContent),
          ),
        );
      assert.deepEqual(actual, expected);
      const downloaded = page.waitForEvent("download");
      await page.locator(`[data-download-interaction="${key}"]`).click();
      const file = await downloaded;
      const csvPath = path.join(output, file.suggestedFilename());
      await file.saveAs(csvPath);
      const lines = (await fs.readFile(csvPath, "utf8")).trim().split(/\r?\n/);
      assert.equal(lines.length, 3);
      assert(lines.at(-1).endsWith(`"${expected[1].at(-1)}"`));
    }
    console.log(
      "PASS: all 11 main-paper figures and 5 tables, including interaction subtasks and CSVs",
    );

    for (const level of ["l1", "l2", "l3"]) {
      await page.locator(`#tab-${level}`).click();
      const ids = await page
        .locator("#example-select option")
        .evaluateAll((options) => options.map((option) => option.value));
      assert.equal(ids.length, 4);
      for (const id of ids) {
        await page.selectOption("#example-select", id);
        await assertImage(page, "#example-image");
        assert.equal(await page.locator("#example-id").textContent(), id);
        assert.equal(
          await page.locator("#answer-details").evaluate((node) => node.open),
          false,
        );
        await page.locator("#answer-details summary").click();
        await page.waitForFunction(
          () =>
            document.querySelectorAll("#example-options li.is-answer").length >
            0,
        );
        assert.equal(
          await page.locator("#example-options li.is-answer").count(),
          id === "E33" ? 3 : 1,
        );
        assert(
          (await page.locator("#example-answer").textContent()).length > 5,
        );
      }
      await page.locator("#next-example").click();
      assert.equal(await page.locator("#example-select").inputValue(), ids[0]);
      await page.locator("#previous-example").click();
      assert.equal(
        await page.locator("#example-select").inputValue(),
        ids.at(-1),
      );
    }
    console.log(
      "PASS: all 12 examples, answers, sources, images, and wrapping pager",
    );

    await page.locator("#tab-l1").focus();
    await page.keyboard.press("ArrowRight");
    assert.equal(
      await page.locator("#tab-l2").getAttribute("aria-selected"),
      "true",
    );
    await page.keyboard.press("Home");
    assert.equal(
      await page.locator("#tab-l1").getAttribute("aria-selected"),
      "true",
    );
    await page.selectOption("#example-select", "E02");

    const mainTable = page.locator("#results-main-table");
    await mainTable.locator('[data-sort="7"]').click();
    assert.equal(
      await mainTable.locator("tbody tr:first-child th").textContent(),
      "Spatial-Interactor-8B",
    );
    await mainTable.locator('[data-sort="7"]').click();
    assert.equal(
      await mainTable.locator("tbody tr:first-child th").textContent(),
      "Qwen2.5-VL-3B",
    );

    const generalizationTable = page.locator("#results-generalization-table");
    assert.equal(await generalizationTable.locator("tbody tr").count(), 12);
    const ablationTable = page.locator("#results-ablation-table");
    assert.equal(await ablationTable.locator("tbody tr").count(), 5);
    const downloadEvent = page.waitForEvent("download");
    await page.locator('[data-download-results="ablation"]').click();
    const download = await downloadEvent;
    const csvPath = path.join(output, download.suggestedFilename());
    await download.saveAs(csvPath);
    const csv = await fs.readFile(csvPath, "utf8");
    assert.equal(csv.trim().split(/\r?\n/).length, 6);
    assert(csv.includes('"Full SFT + OPD"'));
    assert(csv.includes('"60.1"'));
    console.log("PASS: all three result tables, sorting, and CSV download");

    await page.locator(".teaser [data-zoom]").click();
    assert(await page.locator("#figure-dialog").isVisible());
    await assertImage(page, "#figure-dialog-image");
    await page.keyboard.press("Escape");
    assert.equal(await page.locator("#figure-dialog").isVisible(), false);
    assert.equal(
      await page.evaluate(() =>
        document.activeElement.matches(".teaser [data-zoom]"),
      ),
      true,
    );
    await page.locator("#copy-citation").click();
    await page.waitForFunction(
      () =>
        document.querySelector("#copy-status").textContent === "BibTeX copied.",
    );
    console.log("PASS: figure dialog, keyboard controls, and citation copy");

    const layouts = [
      [1440, 1000, "desktop"],
      [1920, 1080, "wide"],
      [768, 1024, "tablet"],
      [390, 844, "mobile"],
      [320, 740, "small-mobile"],
    ];
    for (const [width, height, name] of layouts) {
      await page.setViewportSize({ width, height });
      await page.evaluate(() => {
        document.activeElement?.blur();
        window.scrollTo(0, 0);
      });
      await loadAllImages(page);
      assert(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth + 1,
        ),
        `${name}: horizontal overflow`,
      );
      const overlaps = await page.evaluate(() => {
        const results = [];
        for (const selector of [
          ".nav-inner",
          ".resources",
          ".example-toolbar",
          ".curriculum-tabs",
          ".results-tables",
        ]) {
          const container = document.querySelector(selector);
          const children = [...container.children].filter(
            (node) => node.getClientRects().length,
          );
          children.forEach((child, index) => {
            const a = child.getBoundingClientRect();
            children.slice(index + 1).forEach((other) => {
              const b = other.getBoundingClientRect();
              if (
                Math.min(a.right, b.right) - Math.max(a.left, b.left) > 2 &&
                Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 2
              )
                results.push(selector);
            });
          });
        }
        return results;
      });
      assert.deepEqual(overlaps, [], `${name}: controls overlap`);
      await page.screenshot({ path: path.join(output, `${name}.png`) });
      if (name === "desktop" || name === "mobile") {
        await page.screenshot({
          path: path.join(output, `${name}-full.png`),
          fullPage: true,
        });
      }
      if (width <= 760) {
        await page.locator("#menu-toggle").click();
        assert.equal(
          await page.locator("#menu-toggle").getAttribute("aria-expanded"),
          "true",
        );
        await page.locator('#site-nav a[href="#dataset"]').click();
        assert.equal(
          await page.locator("#menu-toggle").getAttribute("aria-expanded"),
          "false",
        );
        await page.locator("#tab-l3").click();
        await page.selectOption("#example-select", "E33");
        await page.locator("#answer-details summary").click();
        assert(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth + 1,
          ),
        );
        await page.locator("#tab-l1").click();
        await page.selectOption("#example-select", "E02");
      }
      console.log(
        `PASS: ${name} ${width}x${height}, no page overflow or control overlap`,
      );
    }
    assert.deepEqual(errors, [], "Browser console / asset failures");
    await fs.writeFile(
      path.join(output, "checks.json"),
      JSON.stringify(
        {
          examples: 12,
          tables: 5,
          figures: 11,
          viewports: layouts.length,
          errors,
          localFile: !process.env.SITE_URL,
        },
        null,
        2,
      ),
    );
    console.log(`Screenshots and check report: ${output}`);
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
