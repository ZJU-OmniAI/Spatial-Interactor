const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const os = require("node:os");
const { pathToFileURL } = require("node:url");
const { chromium } = require(process.env.PLAYWRIGHT_PATH || "playwright");

const root = path.resolve(__dirname, "..");
const url =
  process.env.SITE_URL || pathToFileURL(path.join(root, "index.html")).href;
const output =
  process.env.OUTPUT_DIR || path.join(os.tmpdir(), "spatial-site-hover-check");

async function main() {
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox"],
  });
  const errors = [];
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
      reducedMotion: "no-preference",
    });
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(url);
    const imageLink = page.locator('[data-paper-figure="1"]');
    await imageLink.scrollIntoViewIfNeeded();
    await imageLink.locator(":scope > img").evaluate((image) => image.decode());
    await page.waitForTimeout(400);
    const before = await imageLink.boundingBox();
    await imageLink.hover({ position: { x: 50, y: 50 } });
    await page.waitForTimeout(300);
    const scaled = await imageLink.locator(":scope > img").boundingBox();
    const origin = await imageLink.evaluate((node) =>
      node.style.getPropertyValue("--focus-x"),
    );
    assert(scaled.width > before.width * 1.01);
    await imageLink.hover({ position: { x: before.width - 50, y: 50 } });
    await page.waitForTimeout(50);
    assert.notEqual(
      await imageLink.evaluate((node) =>
        node.style.getPropertyValue("--focus-x"),
      ),
      origin,
    );
    assert.deepEqual(
      await imageLink.boundingBox(),
      before,
      "Image wrapper stays stable",
    );
    await page.screenshot({ path: path.join(output, "image-hover.png") });
    await page.mouse.move(0, 0);
    await page.waitForTimeout(300);
    assert.equal(
      await imageLink
        .locator(":scope > img")
        .evaluate((node) => getComputedStyle(node).transform),
      "none",
    );
    assert.equal(
      await imageLink.evaluate((node) =>
        node.style.getPropertyValue("--focus-x"),
      ),
      "",
    );

    await page.emulateMedia({ reducedMotion: "reduce" });
    await imageLink.hover();
    assert.equal(
      await imageLink
        .locator(":scope > img")
        .evaluate((node) => getComputedStyle(node).transform),
      "none",
    );
    await imageLink.click();
    assert(await page.locator("#figure-dialog").isVisible());
    await page.keyboard.press("Escape");
    console.log(
      "PASS: cursor-following image hover, stable layout, reset, reduced motion, image viewer",
    );

    async function checkTable(id) {
      const table = page.locator(`#${id}`);
      const values = await table.innerText();
      const row = table.locator("tbody tr.ours").first();
      const cell = row.locator("td").first();
      await cell.scrollIntoViewIfNeeded();
      const bounds = await cell.boundingBox();
      await cell.hover();
      assert.equal(await table.locator(".is-hover-cell").count(), 1);
      assert.equal(await table.locator("tr.is-hover-row").count(), 1);
      assert.equal(
        await table.locator("tbody .is-hover-column").count(),
        await table.locator("tbody tr").count(),
      );
      assert((await table.locator("thead .is-hover-column").count()) > 0);
      assert.notEqual(
        await cell.evaluate((node) => getComputedStyle(node).backgroundColor),
        await row
          .locator("td")
          .nth(1)
          .evaluate((node) => getComputedStyle(node).backgroundColor),
      );
      assert.deepEqual(
        await cell.boundingBox(),
        bounds,
        "Cells never move or resize",
      );
      assert.equal(await table.innerText(), values, "Values are unchanged");
      await table.screenshot({ path: path.join(output, `${id}-hover.png`) });
      await page.evaluate(() => document.activeElement?.blur());
      await page.mouse.move(0, 0);
      assert.equal(
        await table
          .locator(".is-hover-column,.is-hover-cell,.is-hover-row")
          .count(),
        0,
      );
    }
    await checkTable("results-table");
    const table = page.locator("#results-table");
    await table.locator('thead th[colspan="3"]').hover();
    assert.equal(
      await table.locator("tbody .is-hover-column").count(),
      3 * (await table.locator("tbody tr").count()),
    );
    for (const mode of ["generalization", "ablation", "main"]) {
      await page.locator(`[data-table="${mode}"]`).click();
      await checkTable("results-table");
    }
    await page.selectOption("#model-filter", "ours");
    await checkTable("results-table");
    await page.locator('[data-sort="0"]').click();
    assert.equal(await table.locator(".is-hover-cell").count(), 1);
    await checkTable("walker-table");
    await checkTable("esi-table");
    console.log(
      "PASS: row/column/cell feedback across five tables, grouped headers, filters, sorting, keyboard focus",
    );

    const mobile = await browser.newPage({
      viewport: { width: 390, height: 844 },
      hasTouch: true,
      isMobile: true,
      reducedMotion: "no-preference",
    });
    mobile.on("pageerror", (error) => errors.push(error.message));
    await mobile.goto(url);
    await mobile
      .locator("#results-table tbody tr")
      .first()
      .locator("td")
      .first()
      .tap();
    assert.equal(
      await mobile
        .locator(".is-hover-column,.is-hover-cell,.is-hover-row")
        .count(),
      0,
    );
    assert(
      await mobile.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    await mobile.screenshot({ path: path.join(output, "mobile-table.png") });
    assert.deepEqual(errors, []);
    console.log(`PASS: touch has no sticky hover. Screenshots: ${output}`);
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
