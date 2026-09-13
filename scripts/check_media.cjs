/* Playback, motion, gallery, and responsive checks. No runtime dependencies. */
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

async function waitForPlayback(page) {
  await page.waitForFunction(
    () => {
      const video = document.getElementById("demo-video");
      return !video.paused && video.currentTime > 0.25 && video.videoWidth > 0;
    },
    undefined,
    { timeout: 20000 },
  );
}

async function videoFrame(page, time) {
  return page.locator("#demo-video").evaluate(async (video, target) => {
    video.pause();
    await new Promise((resolve) => {
      video.addEventListener("seeked", resolve, { once: true });
      video.currentTime = target;
    });
    const canvas = document.createElement("canvas");
    canvas.width = 96;
    canvas.height = 60;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, 96, 60);
    return Array.from(context.getImageData(0, 0, 96, 60).data);
  }, time);
}

async function checkControls(page, selectors) {
  const issues = await page.evaluate((groups) => {
    const issues = [];
    for (const group of groups) {
      const node = document.querySelector(group);
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

async function main() {
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox"],
  });
  const errors = [];
  const requests = [];
  const playback = [];
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
      reducedMotion: "reduce",
    });
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("request", (request) => {
      if (/\.mp4(?:\?|$)/.test(request.url())) requests.push(request.url());
    });
    await page.goto(url);
    await page.locator("#demo-stage").scrollIntoViewIfNeeded();
    assert.equal(requests.length, 0, "No videos requested before play");
    assert.equal(await page.locator("#demo-video").getAttribute("src"), null);
    assert.equal(await page.locator(".demo-tabs [role=tab]").count(), 2);
    for (let index = 0; index < 2; index += 1) {
      await page.locator(`[data-demo="${index}"]`).click();
      await page.locator("#demo-stage").scrollIntoViewIfNeeded();
      const before = requests.length;
      assert.equal(await page.locator("#demo-video").getAttribute("src"), null);
      await page.locator("#demo-play").click();
      await waitForPlayback(page);
      assert.equal(
        requests.length,
        before + 1,
        "Only the selected video loads",
      );
      const metadata = await page.locator("#demo-video").evaluate((video) => ({
        width: video.videoWidth,
        height: video.videoHeight,
        duration: video.duration,
        controls: video.controls,
      }));
      assert(metadata.width >= 1280 && metadata.controls);
      assert(metadata.duration > 25 && metadata.duration < 52);
      if (url.startsWith("http")) {
        const a = await videoFrame(page, 2);
        const b = await videoFrame(page, 14);
        let changed = 0;
        for (let i = 0; i < a.length; i += 4) {
          if (
            Math.abs(a[i] - b[i]) +
              Math.abs(a[i + 1] - b[i + 1]) +
              Math.abs(a[i + 2] - b[i + 2]) >
            15
          )
            changed += 1;
        }
        assert(changed > 100, `Video ${index}: nonblank moving frames`);
        metadata.changedPixels = changed;
      }
      await page
        .locator("#demo-stage")
        .screenshot({ path: path.join(output, `video-${index}.png`) });
      playback.push(metadata);
    }
    await page.locator('[data-demo="0"]').click();
    await page.locator("#demo-play").click();
    await waitForPlayback(page);
    await page.locator("#results").scrollIntoViewIfNeeded();
    await page.waitForFunction(
      () => document.getElementById("demo-video").paused,
    );
    assert(await page.locator("#back-to-top").isVisible());
    assert.notEqual(
      await page
        .locator("#reading-progress")
        .evaluate((node) => node.style.transform),
      "scaleX(0)",
    );
    await page.locator("#back-to-top").click();
    await page.waitForFunction(() => window.scrollY === 0);
    await page.waitForFunction(
      () => document.getElementById("back-to-top").hidden,
    );
    await page.locator("#demo-path").focus();
    await page.keyboard.press("End");
    assert.equal(
      await page.locator("#demo-path").getAttribute("aria-selected"),
      "true",
    );
    await page.keyboard.press("Home");
    assert.equal(
      await page.locator("#demo-trajectory").getAttribute("aria-selected"),
      "true",
    );
    console.log(
      "PASS: two videos, lazy loading, motion pixels, pause on exit, tabs, reading progress",
    );

    for (const [width, height] of [
      [1440, 1000],
      [1920, 1080],
      [768, 1024],
      [390, 844],
      [320, 740],
    ]) {
      await page.setViewportSize({ width, height });
      await page.locator("#demos").scrollIntoViewIfNeeded();
      await checkControls(page, [".demo-tabs", ".demo-caption", ".demo-meta"]);
      assert(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth + 1,
        ),
      );
      await page
        .locator("#demos")
        .screenshot({ path: path.join(output, `demos-${width}.png`) });
      await page.locator('[data-paper-figure="2"]').click();
      await page.waitForFunction(
        () =>
          document
            .getElementById("figure-viewport")
            .getAttribute("aria-busy") === "false",
      );
      assert.equal(
        await page.locator("#figure-position").textContent(),
        "2 / 11",
      );
      const fit = await page.locator("#figure-dialog-image").boundingBox();
      await page.locator("#zoom-in").click();
      const zoomed = await page.locator("#figure-dialog-image").boundingBox();
      assert(zoomed.width > fit.width * 1.4);
      assert.equal(await page.locator("#zoom-level").textContent(), "150%");
      await page.locator("#fit-figure").click();
      await page.locator("#next-figure").click();
      await page.waitForFunction(
        () =>
          document
            .getElementById("figure-viewport")
            .getAttribute("aria-busy") === "false",
      );
      assert.equal(
        await page.locator("#figure-position").textContent(),
        "3 / 11",
      );
      await checkControls(page, [".dialog-toolbar", ".figure-controls"]);
      await page
        .locator("#figure-dialog")
        .screenshot({ path: path.join(output, `gallery-${width}.png`) });
      await page.keyboard.press("ArrowLeft");
      assert.equal(
        await page.locator("#figure-position").textContent(),
        "2 / 11",
      );
      await page.keyboard.press("Escape");
      assert.equal(await page.locator("#figure-dialog").isVisible(), false);
      assert.equal(
        await page.evaluate(() => document.activeElement.dataset.paperFigure),
        "2",
      );
      console.log(`PASS: ${width}px demos and zoom gallery`);
    }
    await page.locator("#demo-play").click();
    await waitForPlayback(page);
    await page
      .locator("#demo-stage")
      .screenshot({ path: path.join(output, "mobile-playback.png") });
    await page.locator("#example-zoom").click();
    assert.equal(await page.locator("#figure-navigation").isVisible(), false);
    await page.keyboard.press("Escape");
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
    assert.equal(await page.locator("#example-id").textContent(), "E23");
    console.log(
      "PASS: mobile playback, standalone figure, transitions, reduced motion, rapid case switching",
    );

    if (url.startsWith("http")) {
      await page.locator('[data-demo="1"]').click();
      await page.route("**/videos/path-shape.mp4", (route) => route.abort());
      await page.locator("#demo-play").click();
      await page.waitForFunction(() =>
        document
          .getElementById("demo-status")
          .textContent.startsWith("Video unavailable"),
      );
      assert(await page.locator("#demo-play").isVisible());
      await page.unroute("**/videos/path-shape.mp4");
      await page.locator("#demo-play").click();
      await waitForPlayback(page);
      console.log("PASS: failed-video retry");
    }
    assert.deepEqual(errors, []);
    await fs.writeFile(
      path.join(output, "media-checks.json"),
      JSON.stringify({ playback, viewports: 5, errors }, null, 2),
    );
    console.log(`Screenshots and report: ${output}`);
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
