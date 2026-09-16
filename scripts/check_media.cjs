/* Figure gallery, motion, and responsive checks. No runtime dependencies. */
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require(process.env.PLAYWRIGHT_PATH || "playwright");

const root = path.resolve(__dirname, "..");
const url =
  process.env.SITE_URL || pathToFileURL(path.join(root, "index.html")).href;
const output =
  process.env.OUTPUT_DIR || path.join(os.tmpdir(), "spatial-site-media-check");

async function checkControls(page, selectors) {
  const issues = await page.evaluate((groups) => {
    const issues = [];
    for (const group of groups) {
      const node = document.querySelector(group);
      if (!node) {
        issues.push(`${group}: missing`);
        continue;
      }
      const children = [...node.children].filter(
        (child) => child.getClientRects().length,
      );
      children.forEach((child, index) => {
        const a = child.getBoundingClientRect();
        if (child.scrollWidth > child.clientWidth + 2)
          issues.push(`${group}: clipped text`);
        children.slice(index + 1).forEach((other) => {
          const b = other.getBoundingClientRect();
          if (
            Math.min(a.right, b.right) - Math.max(a.left, b.left) > 2 &&
            Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 2
          )
            issues.push(`${group}: overlap`);
        });
      });
    }
    return issues;
  }, selectors);
  assert.deepEqual(issues, []);
}

async function waitForFigure(page) {
  await page.waitForFunction(
    () =>
      document.getElementById("figure-viewport").getAttribute("aria-busy") ===
      "false",
  );
}

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
      reducedMotion: "reduce",
    });
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(url);

    assert.equal(await page.locator("video").count(), 1);
    assert(await page.locator("video").evaluate((video) => video.autoplay && video.muted && video.loop && video.controls));
    assert.equal(await page.locator("[data-paper-figure]").count(), 11);

    for (const [width, height] of [
      [1440, 1000],
      [1920, 1080],
      [768, 1024],
      [390, 844],
      [320, 740],
    ]) {
      await page.setViewportSize({ width, height });
      assert(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth + 1,
        ),
      );
      await page.locator('[data-paper-figure="2"]').click();
      await waitForFigure(page);
      assert.equal(await page.locator("#figure-position").textContent(), "2 / 11");

      const fit = await page.locator("#figure-dialog-image").boundingBox();
      await page.locator("#zoom-in").click();
      const zoomed = await page.locator("#figure-dialog-image").boundingBox();
      assert(zoomed.width > fit.width * 1.4);
      assert.equal(await page.locator("#zoom-level").textContent(), "150%");
      await page.locator("#fit-figure").click();

      await page.locator("#next-figure").click();
      await waitForFigure(page);
      assert.equal(await page.locator("#figure-position").textContent(), "3 / 11");
      await checkControls(page, [".dialog-toolbar", ".figure-controls"]);
      await page
        .locator("#figure-dialog")
        .screenshot({ path: path.join(output, `gallery-${width}.png`) });

      await page.keyboard.press("ArrowLeft");
      await waitForFigure(page);
      assert.equal(await page.locator("#figure-position").textContent(), "2 / 11");
      await page.keyboard.press("Escape");
      assert.equal(await page.locator("#figure-dialog").isVisible(), false);
      assert.equal(
        await page.evaluate(() => document.activeElement.dataset.paperFigure),
        "2",
      );
      console.log(`PASS: ${width}px zoom gallery`);
    }

    await page.locator("#example-zoom").click();
    await waitForFigure(page);
    assert.equal(await page.locator("#figure-navigation").isVisible(), false);
    await page.keyboard.press("Escape");

    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator("#results").scrollIntoViewIfNeeded();
    assert(await page.locator("#back-to-top").isVisible());
    assert.notEqual(
      await page.locator("#reading-progress").evaluate((node) => node.style.transform),
      "scaleX(0)",
    );
    await page.locator("#back-to-top").click();
    await page.waitForFunction(() => window.scrollY === 0);
    await page.waitForFunction(
      () => document.getElementById("back-to-top").hidden,
    );

    assert.equal(await page.evaluate(() => document.getAnimations().length), 0);
    await page.emulateMedia({ reducedMotion: "no-preference" });
    await page.locator("#tab-l2").click();
    assert(
      await page
        .locator(".example-body")
        .evaluate((node) => node.getAnimations().length > 0),
    );
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.waitForFunction(() => document.getAnimations().length === 0);
    await checkControls(page, [".example-toolbar"]);
    await page.locator("#next-example").click({ clickCount: 3 });
    await page.locator("#example-image").evaluate((image) => image.decode());
    assert.equal(await page.locator("#example-id").textContent(), "E14");

    assert.deepEqual(errors, []);
    await fs.writeFile(
      path.join(output, "media-checks.json"),
      JSON.stringify({ figures: 11, viewports: 5, errors }, null, 2),
    );
    console.log("PASS: gallery, navigation, transitions, and reduced motion");
    console.log(`Screenshots and report: ${output}`);
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
