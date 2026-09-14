"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
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
