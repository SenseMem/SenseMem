export function createMediaEvidence(assets) {
  const fragment = document.createDocumentFragment();
  const rendered = new Set();
  for (const asset of assets) {
    const kind = asset.media_type.startsWith("image/") ? "img"
      : asset.media_type.startsWith("video/") ? "video" : "audio";
    if (kind !== "img" && rendered.has(kind)) continue;
    rendered.add(kind);
    const segments = kind === "img" ? [asset] : assets.filter((item) =>
      kind === "video" ? item.media_type.startsWith("video/") : item.media_type.startsWith("audio/"));
    const figure = document.createElement("figure");
    figure.className = "original-media";
    const media = document.createElement(kind);
    const heading = document.createElement("figcaption");
    heading.className = "media-heading";
    const label = document.createElement("span");
    label.textContent = kind === "img" ? "原始画面" : kind === "video" ? "原始音视频" : "原始音频";
    heading.append(label);
    if (kind === "img") {
      media.alt = "此段记忆的原始画面";
      media.loading = "lazy";
    } else {
      media.controls = true;
      media.preload = "metadata";
      media.setAttribute("playsinline", "");
      media.setAttribute("aria-label", kind === "video" ? "回放此段原始音视频" : "回放此段原始音频");
      const duration = document.createElement("span");
      duration.className = "media-duration";
      duration.hidden = true;
      media.addEventListener("loadedmetadata", () => {
        const known = Number.isFinite(media.duration) && media.duration >= 0;
        duration.textContent = known ? `${media.duration.toFixed(1)} 秒` : "";
        duration.hidden = !known;
      });
      heading.append(duration);
      if (segments.length > 1) {
        const selector = document.createElement("select");
        selector.className = "media-selector";
        selector.setAttribute("aria-label", "选择原片片段");
        segments.forEach((segment, index) => {
          const option = document.createElement("option");
          option.value = String(index);
          option.textContent = `片段 ${index + 1} / ${segments.length}`;
          selector.append(option);
        });
        selector.addEventListener("change", () => {
          media.pause();
          duration.textContent = "";
          duration.hidden = true;
          media.src = segments[selector.selectedIndex].url;
          media.load();
        });
        heading.append(selector);
      }
    }
    media.src = asset.url;
    figure.append(media, heading);
    fragment.append(figure);
  }
  return fragment;
}
