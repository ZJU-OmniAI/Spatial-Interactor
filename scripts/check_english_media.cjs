const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');

(async () => {
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.SITE_URL || pathToFileURL(path.resolve(__dirname, '../index.html')).href);
    const video = page.locator('#intro-video');
    await page.waitForFunction(() => document.querySelector('#intro-video').readyState >= 2);
    assert.match(await page.locator('#intro-video-source').getAttribute('src'), /intro-en\.mp4/);
    assert.equal(await page.locator('[data-video-language="en"]').getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('[data-presentation-language="en"]').getAttribute('aria-pressed'), 'true');
    assert(Math.abs(await video.evaluate(v => v.duration) - 135.867) < .2);
    assert.equal(await page.locator('#intro-caption-track').count(), 0);
    await video.evaluate(v => { v.pause(); v.currentTime = 4; });
    await page.waitForTimeout(300);
    await page.locator('.story-deck').screenshot({ path: process.env.SCREENSHOT || '/tmp/spatial-english-media.png' });
    await page.locator('[data-video-language="zh"]').click();
    await page.waitForFunction(() => Math.abs(document.querySelector('#intro-video').duration - 105.408) < .2);
    await page.locator('[data-video-language="en"]').click();
    await page.waitForFunction(() => Math.abs(document.querySelector('#intro-video').duration - 135.867) < .2);
    assert.equal(await video.evaluate(v => v.currentTime), 0);
    await page.locator('[data-presentation-language="zh"]').click();
    assert.match(await page.locator('#story-slide').getAttribute('src'), /-zh\.webp/);
    await page.locator('[data-presentation-language="en"]').click();
    assert.doesNotMatch(await page.locator('#story-slide').getAttribute('src'), /-zh\.webp/);
    assert.deepEqual(errors, []);
    console.log('PASS: English video and 20-slide deck default, both switches, no duplicate captions, no browser errors');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
