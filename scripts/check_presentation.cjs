const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require(process.env.PLAYWRIGHT_PATH || "playwright");

async function main() {
  const browser = await chromium.launch({ headless: true, args: ["--no-sandbox"] });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(process.env.SITE_URL || pathToFileURL(path.resolve(__dirname, "../index.html")).href);
    await page.mouse.move(0, 0);
    await page.waitForFunction(() => Number(document.querySelector("#story-page-number").textContent) > 1);

    // Measure two consecutive changes instead of depending on navigation time.
    const interval = await page.evaluate(() => new Promise((resolve) => {
      const times = [];
      const observer = new MutationObserver(() => {
        times.push(performance.now());
        if (times.length === 2) {
          observer.disconnect();
          resolve(times[1] - times[0]);
        }
      });
      observer.observe(document.querySelector("#story-page-number"), { childList: true });
    }));
    assert(interval >= 900 && interval < 1300, `Slide interval: ${interval} ms`);
    console.log(`PASS: automatic page change every ${Math.round(interval)} ms`);

    await page.locator("#story-slide").hover();
    const before = await page.locator("#story-page-number").textContent();
    await page.waitForTimeout(1250);
    assert.equal(await page.locator("#story-page-number").textContent(), before);
    await page.locator("#story-toggle").click();
    await page.mouse.move(0, 0);
    await page.waitForTimeout(1250);
    assert.equal(await page.locator("#story-page-number").textContent(), before);
    assert.equal(await page.locator("#story-toggle").getAttribute("aria-label"), "Play presentation");

    await page.locator("#story-next").click();
    assert.notEqual(await page.locator("#story-page-number").textContent(), before);
    await page.locator('[data-story-index="3"]').click();
    assert.equal(await page.locator("#story-page-title").textContent(), "Observation, action, next observation");
    const caption = await page.locator(".story-deck-stage figcaption").boundingBox();
    const slide = await page.locator("#story-slide").boundingBox();
    assert(caption.y >= slide.y + slide.height - 1, "Caption overlaps slide");
    const videoWidth = (await page.locator("video").boundingBox()).width;
    assert(Math.abs(videoWidth - slide.width) < 3, "Video and slide widths differ");

    await page.locator("#story-toggle").click();
    await page.mouse.move(0, 0);
    await page.waitForTimeout(1250);
    assert.notEqual(await page.locator("#story-page-number").textContent(), "04");
    await page.emulateMedia({ reducedMotion: "reduce" });
    const reduced = await page.locator("#story-page-number").textContent();
    await page.waitForTimeout(1250);
    assert.equal(await page.locator("#story-page-number").textContent(), reduced);
    assert.equal(await page.locator(".story-thumb").count(), 20);
    assert.deepEqual(errors, []);
    console.log("PASS: hover pause, explicit pause/resume, manual navigation, reduced motion, and media sizing");
  } finally {
    await browser.close();
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
