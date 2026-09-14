"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const demos = [
    {
      file: "trajectory-integration",
      title: "Trajectory integration",
      description:
        "Local transitions, traveled distance, and start-to-end displacement.",
    },
    {
      file: "path-shape",
      title: "Path shape",
      description:
        "Connect successive movements and turns into a complete route.",
    },
  ];
  const video = $("demo-video");
  const play = $("demo-play");
  const tabs = [...document.querySelectorAll(".demo-tabs [role=tab]")];
  let demoIndex = 0;
  let playGeneration = 0;

  function videoStatus(message) {
    $("demo-status").textContent = message;
    $("demo-status").hidden = !message;
  }

  function selectDemo(index) {
    demoIndex = index;
    playGeneration += 1;
    video.pause();
    if (video.hasAttribute("src")) {
      video.removeAttribute("src");
      video.load();
    }
    video.controls = false;
    play.hidden = false;
    play.disabled = false;
    const demo = demos[index];
    const base = `assets/videos/${demo.file}`;
    video.poster = `${base}.webp`;
    video.setAttribute("aria-label", `${demo.title} demonstration`);
    play.setAttribute("aria-label", `Play ${demo.title.toLowerCase()}`);
    $("demo-title").textContent = demo.title;
    $("demo-description").textContent = demo.description;
    $("demo-download").href = `${base}.mp4`;
    const fallback = video.querySelector("a");
    fallback.href = `${base}.mp4`;
    fallback.textContent = `${demo.title} video`;
    tabs.forEach((tab, i) => {
      tab.setAttribute("aria-selected", String(i === index));
      tab.tabIndex = i === index ? 0 : -1;
    });
    $("demo-panel").setAttribute("aria-labelledby", tabs[index].id);
    videoStatus("");
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => {
      if (index !== demoIndex) selectDemo(index);
    });
    tab.addEventListener("keydown", (event) => {
      const next = {
        ArrowRight: (index + 1) % tabs.length,
        ArrowLeft: (index + tabs.length - 1) % tabs.length,
        Home: 0,
        End: tabs.length - 1,
      }[event.key];
      if (next === undefined) return;
      event.preventDefault();
      if (next !== demoIndex) selectDemo(next);
      tabs[next].focus();
    });
  });

  function playbackError() {
    if (!video.hasAttribute("src")) return;
    play.hidden = false;
    play.disabled = false;
    play.setAttribute("aria-label", "Retry video playback");
    videoStatus("Video unavailable. Retry or download the video below.");
  }

  play.addEventListener("click", async () => {
    const generation = ++playGeneration;
    play.hidden = true;
    video.controls = true;
    videoStatus("Loading video...");
    if (!video.hasAttribute("src") || video.error) {
      video.src = `assets/videos/${demos[demoIndex].file}.mp4`;
    }
    try {
      await video.play();
      if (generation === playGeneration) video.focus({ preventScroll: true });
    } catch (error) {
      if (generation !== playGeneration) return;
      if (error.name === "AbortError") videoStatus("");
      else playbackError();
    }
  });
  video.addEventListener("playing", () => videoStatus(""));
  video.addEventListener("canplay", () => videoStatus(""));
  video.addEventListener("error", playbackError);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) video.pause();
  });
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) video.pause();
    }).observe($("demo-stage"));
  }

  const dialog = $("figure-dialog");
  const figure = $("figure-dialog-image");
  const viewport = $("figure-viewport");
  const gallery = [...document.querySelectorAll("[data-paper-figure]")].sort(
    (a, b) => Number(a.dataset.paperFigure) - Number(b.dataset.paperFigure),
  );
  const zoomSteps = [1, 1.5, 2, 3];
  let zoomIndex = 0;
  let galleryIndex = -1;
  let figureTrigger;
  let imageGeneration = 0;

  function fitImage(keepCenter = false) {
    if (!dialog.open || !figure.naturalWidth) return;
    const x =
      (viewport.scrollLeft + viewport.clientWidth / 2) /
      Math.max(1, viewport.scrollWidth);
    const y =
      (viewport.scrollTop + viewport.clientHeight / 2) /
      Math.max(1, viewport.scrollHeight);
    const ratio = figure.naturalWidth / figure.naturalHeight;
    const fitWidth = Math.min(
      figure.naturalWidth,
      viewport.clientWidth - 24,
      (viewport.clientHeight - 24) * ratio,
    );
    const width = Math.max(1, fitWidth) * zoomSteps[zoomIndex];
    figure.style.width = `${width}px`;
    figure.style.marginTop = `${Math.max(0, (viewport.clientHeight - width / ratio) / 2)}px`;
    $("zoom-level").textContent = `${zoomSteps[zoomIndex] * 100}%`;
    $("zoom-out").disabled = zoomIndex === 0;
    $("zoom-in").disabled = zoomIndex === zoomSteps.length - 1;
    if (keepCenter) {
      viewport.scrollLeft = x * viewport.scrollWidth - viewport.clientWidth / 2;
      viewport.scrollTop =
        y * viewport.scrollHeight - viewport.clientHeight / 2;
    } else {
      viewport.scrollLeft = 0;
      viewport.scrollTop = 0;
    }
  }

  async function showFigure(trigger) {
    const generation = ++imageGeneration;
    zoomIndex = 0;
    galleryIndex = gallery.indexOf(trigger);
    $("figure-navigation").hidden = galleryIndex < 0;
    $("figure-position").textContent =
      galleryIndex < 0 ? "" : `${galleryIndex + 1} / ${gallery.length}`;
    $("figure-dialog-caption").textContent = trigger.dataset.caption;
    figure.src = trigger.href;
    figure.alt = trigger.querySelector("img").alt;
    $("figure-original").href = trigger.href;
    viewport.setAttribute("aria-busy", "true");
    try {
      await figure.decode();
      if (generation === imageGeneration) fitImage();
    } catch {
      if (generation === imageGeneration)
        figure.alt = "Image unavailable. Open the original image below.";
    } finally {
      if (generation === imageGeneration)
        viewport.setAttribute("aria-busy", "false");
    }
  }

  document.addEventListener("click", (event) => {
    const trigger = event.target.closest("a[data-zoom]");
    if (
      !trigger ||
      event.ctrlKey ||
      event.metaKey ||
      event.shiftKey ||
      event.altKey
    )
      return;
    event.preventDefault();
    figureTrigger = trigger;
    dialog.showModal();
    document.body.classList.add("modal-open");
    showFigure(trigger);
    $("close-figure").focus();
  });
  function moveFigure(offset) {
    if (galleryIndex < 0) return;
    showFigure(
      gallery[(galleryIndex + offset + gallery.length) % gallery.length],
    );
  }
  function changeZoom(offset) {
    zoomIndex = Math.min(zoomSteps.length - 1, Math.max(0, zoomIndex + offset));
    fitImage(true);
  }
  $("previous-figure").addEventListener("click", () => moveFigure(-1));
  $("next-figure").addEventListener("click", () => moveFigure(1));
  $("zoom-in").addEventListener("click", () => changeZoom(1));
  $("zoom-out").addEventListener("click", () => changeZoom(-1));
  $("fit-figure").addEventListener("click", () => {
    zoomIndex = 0;
    fitImage();
  });
  dialog.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === "+" || event.key === "=") changeZoom(1);
    else if (event.key === "-") changeZoom(-1);
    else if (event.key === "ArrowLeft" && zoomIndex === 0) moveFigure(-1);
    else if (event.key === "ArrowRight" && zoomIndex === 0) moveFigure(1);
    else return;
    event.preventDefault();
  });
  $("close-figure").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (
      event.clientX < rect.left ||
      event.clientX > rect.right ||
      event.clientY < rect.top ||
      event.clientY > rect.bottom
    )
      dialog.close();
  });
  dialog.addEventListener("close", () => {
    imageGeneration += 1;
    document.body.classList.remove("modal-open");
    figureTrigger?.focus({ preventScroll: true });
  });
  window.addEventListener("resize", () => fitImage());
})();
