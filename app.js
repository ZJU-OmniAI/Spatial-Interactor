"use strict";

(() => {
  const data = window.SPATIAL_DATA;
  const $ = (id) => document.getElementById(id);
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  function animateIn(node, distance = 10) {
    if (reducedMotion.matches || !node.animate) return;
    node.getAnimations().forEach((animation) => animation.cancel());
    node.animate(
      [
        { opacity: 0.35, transform: `translateY(${distance}px)` },
        { opacity: 1, transform: "translateY(0)" },
      ],
      { duration: 320, easing: "cubic-bezier(0.2, 0.7, 0.2, 1)" },
    );
  }
  reducedMotion.addEventListener("change", (event) => {
    if (event.matches)
      document.getAnimations().forEach((animation) => animation.cancel());
  });
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };

  const storyPages = [
    "Opening",
    "Why spatial interaction?",
    "State-transition diagnostics",
    "Observation, action, next observation",
    "Spatial-Interactor overview",
    "Three-level curriculum",
    "L1 · Passive world state",
    "L2 · Active self state",
    "L3 · Long-horizon integration",
    "LSI-108K dataset",
    "Two-stage training",
    "On-Policy Distillation",
    "Experimental setup",
    "Main results",
    "Cross-benchmark transfer",
    "Ablation",
    "Training analysis",
    "Temporal evidence",
    "Closed-loop interaction and qualitative cases",
    "Takeaway",
  ];
  let storyIndex = 0;
  const storySlide = $("story-slide");
  const storyPageNumber = $("story-page-number");
  const storyPageTitle = $("story-page-title");
  const storyProgress = $("story-progress");
  const storyThumbs = $("story-thumbs");
  if (storySlide && storyPageNumber && storyPageTitle && storyProgress && storyThumbs) {
    const storyViewer = document.querySelector(".story-deck-viewer");
    const storyToggle = $("story-toggle");
    const storyInterval = 1000;
    let storyTimer;
    let storyPaused = reducedMotion.matches;
    let storyHovered = false;
    let storyFocused = false;
    function updateStoryToggle() {
      const label = storyPaused ? "Play presentation" : "Pause presentation";
      storyToggle.setAttribute("aria-label", label);
      storyToggle.title = label;
      storyToggle.querySelector("img").src = `assets/icons/${storyPaused ? "play" : "pause"}.svg`;
    }
    const storyAsset = (index) => {
      const cacheBust = [1, 3, 17].includes(index) ? "?v=20260916" : "";
      return `assets/presentation/slide-${String(index + 1).padStart(2, "0")}.webp${cacheBust}`;
    };
    storyPages.forEach((title, index) => {
      const button = element("button", undefined, "story-thumb");
      button.type = "button";
      button.dataset.storyIndex = String(index);
      button.title = `${index + 1}: ${title}`;
      button.setAttribute("aria-label", `Open presentation page ${index + 1}: ${title}`);
      const image = document.createElement("img");
      image.src = storyAsset(index);
      image.alt = "";
      image.loading = index < 4 ? "eager" : "lazy";
      button.append(image, element("span", String(index + 1).padStart(2, "0")));
      button.addEventListener("click", () => selectStory(index));
      storyThumbs.append(button);
    });
    function renderStory(nextIndex, animate = true) {
      storyIndex = (nextIndex + storyPages.length) % storyPages.length;
      const page = String(storyIndex + 1).padStart(2, "0");
      storySlide.src = storyAsset(storyIndex);
      storySlide.alt = `Presentation page ${storyIndex + 1} of ${storyPages.length}: ${storyPages[storyIndex]}`;
      storyPageNumber.textContent = page;
      storyPageTitle.textContent = storyPages[storyIndex];
      storyProgress.style.transform = `scaleX(${(storyIndex + 1) / storyPages.length})`;
      [...storyThumbs.children].forEach((thumb, index) => {
        const active = index === storyIndex;
        thumb.classList.toggle("is-active", active);
        thumb.setAttribute("aria-current", active ? "page" : "false");
      });
      if (animate && !reducedMotion.matches) animateIn(storySlide, 0);
    }
    function stopStoryAuto() {
      clearTimeout(storyTimer);
    }
    function startStoryAuto() {
      stopStoryAuto();
      if (storyPaused || storyHovered || storyFocused || document.hidden) return;
      storyTimer = setTimeout(() => {
        renderStory(storyIndex + 1);
        startStoryAuto();
      }, storyInterval);
    }
    function selectStory(nextIndex) {
      renderStory(nextIndex);
      startStoryAuto();
    }
    $("story-previous").addEventListener("click", () => selectStory(storyIndex - 1));
    $("story-next").addEventListener("click", () => selectStory(storyIndex + 1));
    storyToggle.addEventListener("click", () => {
      storyPaused = !storyPaused;
      updateStoryToggle();
      startStoryAuto();
    });
    document.addEventListener("keydown", (event) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement || event.target.isContentEditable) return;
      if (document.querySelector("dialog[open]") || document.activeElement.closest("video")) return;
      const bounds = storyViewer.getBoundingClientRect();
      const focused = document.activeElement;
      const isOtherControl = focused.matches("button, a, select, [role=tab], summary") && !storyViewer.contains(focused) && !storyThumbs.contains(focused);
      const deckIsActive = !isOtherControl && (storyViewer.contains(focused) || storyThumbs.contains(focused) || (bounds.top < innerHeight * 0.72 && bounds.bottom > innerHeight * 0.28));
      if (!deckIsActive) return;
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        selectStory(storyIndex + (event.key === "ArrowLeft" ? -1 : 1));
      }
    });
    [storyViewer, storyThumbs].forEach((region) => {
      region.addEventListener("mouseenter", () => {
        storyHovered = true;
        stopStoryAuto();
      });
      region.addEventListener("mouseleave", () => {
        storyHovered = false;
        startStoryAuto();
      });
      region.addEventListener("focusin", (event) => {
        storyFocused = event.target !== storyToggle;
        startStoryAuto();
      });
      region.addEventListener("focusout", () => setTimeout(() => {
        const focused = document.activeElement;
        storyFocused = focused !== storyToggle && (storyViewer.contains(focused) || storyThumbs.contains(focused));
        startStoryAuto();
      }));
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stopStoryAuto();
      else startStoryAuto();
    });
    reducedMotion.addEventListener("change", () => {
      storyPaused = reducedMotion.matches;
      updateStoryToggle();
      startStoryAuto();
    });
    renderStory(0, false);
    updateStoryToggle();
    startStoryAuto();
  }

  function setupTabs(selector, onSelect) {
    const tabs = [...document.querySelectorAll(`${selector} [role="tab"]`)];
    function select(tab) {
      tabs.forEach((candidate) => {
        const active = candidate === tab;
        candidate.setAttribute("aria-selected", String(active));
        candidate.tabIndex = active ? 0 : -1;
      });
      $(tab.getAttribute("aria-controls")).setAttribute(
        "aria-labelledby",
        tab.id,
      );
      onSelect(tab);
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => select(tab));
      tab.addEventListener("keydown", (event) => {
        const targets = {
          ArrowRight: (index + 1) % tabs.length,
          ArrowLeft: (index + tabs.length - 1) % tabs.length,
          Home: 0,
          End: tabs.length - 1,
        };
        if (!(event.key in targets)) return;
        event.preventDefault();
        const next = tabs[targets[event.key]];
        select(next);
        next.focus();
      });
    });
    select(tabs[0]);
  }

  let activeLevel = "l1";
  const casePositions = { l1: 0, l2: 0, l3: 0 };

  function optionList(options, answer, compact = false) {
    const list = element(
      "ul",
      undefined,
      `qa-options${compact ? " compact" : ""}`,
    );
    options.forEach((option, index) => {
      const letter = String.fromCharCode(65 + index);
      const item = element("li");
      item.dataset.correct = String(letter === answer);
      item.append(
        element("span", letter, "option-letter"),
        element("span", option),
      );
      list.append(item);
    });
    return list;
  }

  function renderCase() {
    const ids = data.levels[activeLevel].cases;
    const position = casePositions[activeLevel];
    const id = ids[position];
    const example = data.cases[id];
    $("example-select").value = id;
    $("example-position").textContent =
      `${String(position + 1).padStart(2, "0")} / ${String(ids.length).padStart(2, "0")}`;
    $("example-domain").textContent = example.domain;
    $("example-source").textContent = example.source;
    $("example-id").textContent = id;
    $("example-question").textContent = example.question;
    const caption = `${example.title} (${example.source}). ${example.question}`;
    $("example-image").src = `assets/${id}.webp`;
    $("example-image").alt = caption;
    $("example-zoom").href = `assets/${id}.webp`;
    $("example-zoom").dataset.caption = caption;
    $("answer-details").open = false;
    const options = $("example-options");
    options.replaceChildren();
    if (example.fields) {
      example.fields.forEach((field) => {
        options.append(element("p", field.label, "field-title"));
        options.append(optionList(field.options, field.answer, true));
      });
      $("example-answer").textContent = example.fields
        .map(
          (field) =>
            `${field.label}: ${field.answer}. ${field.options[field.answer.charCodeAt(0) - 65]}`,
        )
        .join("; ");
    } else if (example.options.length) {
      options.append(optionList(example.options, example.answer));
      $("example-answer").textContent =
        `${example.answer}. ${example.options[example.answer.charCodeAt(0) - 65]}`;
    } else {
      $("example-answer").textContent = example.answerText;
      $("answer-details").open = true;
    }
    animateIn(document.querySelector(".example-body"), 6);
  }

  setupTabs(".curriculum-tabs", (tab) => {
    activeLevel = tab.dataset.level;
    const level = data.levels[activeLevel];
    $("curriculum-panel").dataset.level = activeLevel;
    $("level-title").textContent = level.title;
    $("level-description").textContent = level.description;
    $("level-count").textContent = level.count;
    $("task-types").replaceChildren(
      ...level.tasks.map((task) => element("li", task)),
    );
    $("example-select").replaceChildren(
      ...level.cases.map((id) => {
        const option = element("option", `${id} · ${data.cases[id].title}`);
        option.value = id;
        return option;
      }),
    );
    renderCase();
  });
  $("example-select").addEventListener("change", (event) => {
    casePositions[activeLevel] = data.levels[activeLevel].cases.indexOf(
      event.target.value,
    );
    renderCase();
  });
  function moveExample(offset) {
    const count = data.levels[activeLevel].cases.length;
    casePositions[activeLevel] =
      (casePositions[activeLevel] + count + offset) % count;
    renderCase();
  }
  $("previous-example").addEventListener("click", () => moveExample(-1));
  $("next-example").addEventListener("click", () => moveExample(1));
  $("answer-details").addEventListener("toggle", () => {
    document.querySelectorAll("#example-options li").forEach((item) => {
      item.classList.toggle(
        "is-answer",
        $("answer-details").open && item.dataset.correct === "true",
      );
    });
  });

  const mainColumns = [
    ["Rel. dist.", "VSI-Bench Relative Distance"],
    ["Route", "VSI-Bench Route Planning"],
    ["Avg.", "VSI-Bench Average"],
    ["MindCube", "MindCube-Tiny"],
    ["Displ.", "VSTI-Bench Camera Displacement"],
    ["Avg.", "VSTI-Bench Average"],
    ["SPBench", "SPBench-MV"],
    ["Overall", "Overall"],
  ];
  const generalColumns = [
    ["MMSI", "MMSI"],
    ["ViewSpatial", "ViewSpatial"],
    ["SAT-Real", "SAT-Real"],
    ["SAT-Syn", "SAT-Syn"],
    ["Overall", "Overall"],
  ];
  const resultsConfig = {
    main: {
      tableId: "results-main-table",
    },
    generalization: {
      tableId: "results-generalization-table",
    },
    ablation: {
      tableId: "results-ablation-table",
    },
  };
  const resultSort = Object.fromEntries(
    Object.keys(resultsConfig).map((key) => [key, { index: null, ascending: false }]),
  );

  function resultColumns(key) {
    return key === "generalization" ? generalColumns : mainColumns;
  }

  function sortHeader(key, index, rowSpan = 1) {
    const [shortName, fullName] = resultColumns(key)[index];
    const sort = resultSort[key];
    const th = element("th");
    th.scope = "col";
    if (rowSpan > 1) th.rowSpan = rowSpan;
    th.setAttribute(
      "aria-sort",
      sort.index === index
        ? sort.ascending
          ? "ascending"
          : "descending"
        : "none",
    );
    const button = element(
      "button",
      undefined,
      `sort-button${sort.index === index ? " active" : ""}${sort.ascending && sort.index === index ? " ascending" : ""}`,
    );
    button.dataset.sort = String(index);
    button.title = `Sort by ${fullName}`;
    button.setAttribute("aria-label", `Sort by ${fullName}`);
    const icon = element("img");
    icon.src = "assets/icons/arrow-down.svg";
    icon.alt = "";
    icon.width = 12;
    icon.height = 12;
    button.append(element("span", shortName), icon);
    th.append(button);
    return th;
  }

  function renderResultsTable(key) {
    const config = resultsConfig[key];
    const rows = data[key];
    const sort = resultSort[key];
    const columns = resultColumns(key);
    let visibleRows = [...rows];
    if (sort.index !== null) {
      visibleRows.sort(
        (a, b) =>
          (a.values[sort.index] - b.values[sort.index]) *
          (sort.ascending ? 1 : -1),
      );
    }

    const target = $(config.tableId);
    const head = target.querySelector("thead");
    head.replaceChildren();
    const top = element("tr");
    const model = element("th", key === "ablation" ? "Training" : "Model");
    model.scope = "col";
    top.append(model);
    if (key === "generalization") {
      columns.forEach((_, index) => top.append(sortHeader(key, index)));
      head.append(top);
    } else {
      model.rowSpan = 2;
      const vsi = element("th", "VSI-Bench");
      vsi.colSpan = 3;
      vsi.scope = "colgroup";
      const vsti = element("th", "VSTI-Bench");
      vsti.colSpan = 2;
      vsti.scope = "colgroup";
      top.append(
        vsi,
        sortHeader(key, 3, 2),
        vsti,
        sortHeader(key, 6, 2),
        sortHeader(key, 7, 2),
      );
      const sub = element("tr", undefined, "subhead");
      [0, 1, 2, 4, 5].forEach((index) => sub.append(sortHeader(key, index)));
      head.append(top, sub);
    }

    const maxima = columns.map((_, index) =>
      Math.max(...rows.map((row) => row.values[index])),
    );
    target.querySelector("tbody").replaceChildren(
      ...visibleRows.map((row) => {
        const tr = element("tr", undefined, row.group === "ours" ? "ours" : "");
        const label = element("th", row.name);
        label.scope = "row";
        tr.append(label);
        row.values.forEach((value, index) => {
          const td = element("td", value.toFixed(1));
          td.classList.toggle("best", value === maxima[index]);
          td.classList.toggle("overall", index === row.values.length - 1);
          tr.append(td);
        });
        return tr;
      }),
    );
  }

  Object.keys(resultsConfig).forEach((key) => {
    renderResultsTable(key);
    animateIn($(resultsConfig[key].tableId), 5);
  });

  document.querySelectorAll(".results-table").forEach((target) => {
    target.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-sort]");
      if (!button) return;
      const key = Object.entries(resultsConfig).find(
        ([, config]) => config.tableId === target.id,
      )?.[0];
      if (!key) return;
      const state = resultSort[key];
      const index = Number(button.dataset.sort);
      state.ascending = state.index === index ? !state.ascending : false;
      state.index = index;
      renderResultsTable(key);
      target
        .querySelector(`button[data-sort="${index}"]`)
        ?.focus({ preventScroll: true });
    });
  });
  function downloadCSV(filename, lines) {
    const csvCell = (value) => `"${String(value).replaceAll('"', '""')}"`;
    const blob = new Blob(
      [lines.map((line) => line.map(csvCell).join(",")).join("\r\n") + "\r\n"],
      { type: "text/csv;charset=utf-8" },
    );
    const link = element("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }
  document.querySelectorAll("[data-download-results]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.downloadResults;
      const rows = data[key];
      downloadCSV(`spatial-interactor-${key}.csv`, [
        [
          key === "ablation" ? "Training" : "Model",
          ...resultColumns(key).map((column) => column[1]),
        ],
        ...rows.map((row) => [
          row.name,
          ...row.values.map((value) => value.toFixed(1)),
        ]),
      ]);
    });
  });

  Object.entries(data.interaction).forEach(([key, benchmark]) => {
    const target = $(`${key}-table`);
    const header = element("tr");
    ["Model", ...benchmark.columns].forEach((name) => {
      const cell = element("th", name);
      cell.scope = "col";
      header.append(cell);
    });
    target.querySelector("thead").replaceChildren(header);
    const last = benchmark.columns.length - 1;
    const best = benchmark.columns.map((_, index) => {
      const values = benchmark.rows.map((row) => row.values[index]);
      return index === last ? Math.min(...values) : Math.max(...values);
    });
    target.querySelector("tbody").replaceChildren(
      ...benchmark.rows.map((row) => {
        const tr = element("tr", undefined, row.group === "ours" ? "ours" : "");
        const name = element("th", row.name);
        name.scope = "row";
        tr.append(name);
        row.values.forEach((value, index) => {
          const td = element("td", value.toFixed(index === last ? 2 : 1));
          td.classList.toggle("best", value > 0 && value === best[index]);
          td.classList.toggle("overall", index === last - 1);
          tr.append(td);
        });
        return tr;
      }),
    );
    document
      .querySelector(`[data-download-interaction="${key}"]`)
      .addEventListener("click", () => {
        downloadCSV(`spatial-interactor-${key}.csv`, [
          ["Model", ...benchmark.columns],
          ...benchmark.rows.map((row) => [
            row.name,
            ...row.values.map((value, index) =>
              value.toFixed(index === last ? 2 : 1),
            ),
          ]),
        ]);
      });
  });

  const finePointer = matchMedia("(hover: hover) and (pointer: fine)");
  document.querySelectorAll(".figure-link").forEach((link) => {
    let frame;
    let position;
    function reset() {
      cancelAnimationFrame(frame);
      frame = undefined;
      link.style.removeProperty("--focus-x");
      link.style.removeProperty("--focus-y");
    }
    link.addEventListener("pointermove", (event) => {
      if (
        !finePointer.matches ||
        reducedMotion.matches ||
        event.pointerType !== "mouse"
      )
        return;
      position = { x: event.clientX, y: event.clientY };
      if (frame !== undefined) return;
      frame = requestAnimationFrame(() => {
        const rect = link.getBoundingClientRect();
        const clamp = (value) => Math.max(0, Math.min(100, value));
        link.style.setProperty(
          "--focus-x",
          `${clamp(((position.x - rect.left) / rect.width) * 100)}%`,
        );
        link.style.setProperty(
          "--focus-y",
          `${clamp(((position.y - rect.top) / rect.height) * 100)}%`,
        );
        frame = undefined;
      });
    });
    link.addEventListener("pointerleave", reset);
    reducedMotion.addEventListener("change", reset);
    finePointer.addEventListener("change", reset);
  });

  document.querySelectorAll(".table-scroll table").forEach((target) => {
    let active;
    let marked = [];
    function clear() {
      marked.forEach((node) =>
        node.classList.remove(
          "is-hover-column",
          "is-hover-cell",
          "is-hover-row",
        ),
      );
      marked = [];
      active = undefined;
    }
    function highlight(cell) {
      if (!cell || cell.closest("table") !== target || cell === active) return;
      clear();
      active = cell;
      // Resolve actual column spans, including the two-row benchmark headers.
      const grid = [];
      const spans = new Map();
      [...target.rows].forEach((row, rowIndex) => {
        grid[rowIndex] ||= [];
        let column = 0;
        [...row.cells].forEach((entry) => {
          while (grid[rowIndex][column]) column += 1;
          spans.set(entry, { start: column, end: column + entry.colSpan - 1 });
          for (let r = rowIndex; r < rowIndex + entry.rowSpan; r += 1) {
            grid[r] ||= [];
            for (let c = column; c < column + entry.colSpan; c += 1)
              grid[r][c] = entry;
          }
          column += entry.colSpan;
        });
      });
      const selected = spans.get(cell);
      if (!selected) return;
      spans.forEach((span, entry) => {
        if (span.end < selected.start || span.start > selected.end) return;
        if (selected.start === 0 && entry.closest("tbody") && entry !== cell)
          return;
        entry.classList.add("is-hover-column");
        marked.push(entry);
      });
      if (cell.closest("tbody")) {
        cell.parentElement.classList.add("is-hover-row");
        marked.push(cell.parentElement);
      }
      cell.classList.add("is-hover-cell");
      marked.push(cell);
    }
    target.addEventListener("pointerover", (event) => {
      if (finePointer.matches && event.pointerType === "mouse")
        highlight(event.target.closest("th, td"));
    });
    target.addEventListener("pointerleave", () => {
      clear();
      if (target.contains(document.activeElement))
        highlight(document.activeElement.closest("th, td"));
    });
    target.addEventListener("focusin", (event) =>
      highlight(event.target.closest("th, td")),
    );
    target.addEventListener("focusout", (event) => {
      if (!target.contains(event.relatedTarget)) clear();
    });
    finePointer.addEventListener("change", clear);
  });

  const menu = $("menu-toggle");
  const nav = $("site-nav");
  function setMenu(open) {
    nav.classList.toggle("open", open);
    menu.setAttribute("aria-expanded", String(open));
    menu.setAttribute(
      "aria-label",
      open ? "Close navigation" : "Open navigation",
    );
    menu.title = open ? "Close navigation" : "Open navigation";
    menu.querySelector("img").src = `assets/icons/${open ? "x" : "menu"}.svg`;
  }
  menu.addEventListener("click", () =>
    setMenu(menu.getAttribute("aria-expanded") !== "true"),
  );
  nav.addEventListener("click", (event) => {
    if (event.target.closest("a")) setMenu(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && nav.classList.contains("open")) {
      setMenu(false);
      menu.focus();
    }
  });
  matchMedia("(min-width:761px)").addEventListener("change", (event) => {
    if (event.matches) setMenu(false);
  });
  const sectionLinks = [...nav.querySelectorAll('a[href^="#"]')];
  let scrollPending = false;
  function updateActiveLink() {
    const travel = document.documentElement.scrollHeight - window.innerHeight;
    const progress =
      travel > 0 ? Math.min(1, Math.max(0, window.scrollY / travel)) : 0;
    $("reading-progress").style.transform = `scaleX(${progress})`;
    $("back-to-top").hidden = window.scrollY < window.innerHeight;
    const active = sectionLinks
      .filter(
        (link) =>
          document.querySelector(link.hash).getBoundingClientRect().top <= 170,
      )
      .at(-1);
    sectionLinks.forEach((link) => {
      link.classList.toggle("active", link === active);
      if (link === active) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
    scrollPending = false;
  }
  window.addEventListener(
    "scroll",
    () => {
      if (!scrollPending) {
        requestAnimationFrame(updateActiveLink);
        scrollPending = true;
      }
    },
    { passive: true },
  );
  updateActiveLink();
  window.addEventListener("resize", updateActiveLink);
  $("back-to-top").addEventListener("click", () => {
    window.scrollTo({
      top: 0,
      behavior: reducedMotion.matches ? "instant" : "smooth",
    });
    document.querySelector(".brand").focus({ preventScroll: true });
  });
  if ("IntersectionObserver" in window) {
    // Content stays visible without JavaScript or animation support.
    const reveals = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          animateIn(entry.target, 14);
          reveals.unobserve(entry.target);
        });
      },
      { threshold: 0.08 },
    );
    document
      .querySelectorAll(
        ".story-deck-intro, .story-deck-viewer, .section-title, .paper-figure, .insights, .analysis-block",
      )
      .forEach((node) => reveals.observe(node));
  }

  let copyTimer;
  $("copy-citation").addEventListener("click", async () => {
    let copied = false;
    try {
      await navigator.clipboard.writeText($("bibtex").textContent);
      copied = true;
    } catch {
      const field = element("textarea");
      field.value = $("bibtex").textContent;
      field.className = "sr-only";
      document.body.append(field);
      field.select();
      copied = document.execCommand("copy");
      field.remove();
      $("copy-citation").focus({ preventScroll: true });
    }
    $("copy-status").textContent = copied
      ? "BibTeX copied."
      : "Unable to copy. The BibTeX is available above.";
    $("copy-icon").src = `assets/icons/${copied ? "check" : "copy"}.svg`;
    $("copy-citation").dataset.tooltip = copied ? "Copied" : "Copy BibTeX";
    clearTimeout(copyTimer);
    copyTimer = setTimeout(() => {
      $("copy-icon").src = "assets/icons/copy.svg";
      $("copy-citation").dataset.tooltip = "Copy BibTeX";
    }, 2500);
  });
})();
