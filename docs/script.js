"use strict";

const copyButton = document.getElementById("copy-citation");
let resetTimer;
copyButton.addEventListener("click", async () => {
  const code = document.getElementById("bibtex");
  const status = document.getElementById("copy-status");
  clearTimeout(resetTimer);
  try {
    await navigator.clipboard.writeText(code.textContent);
    copyButton.querySelector("img").src = "assets/icons/check.svg";
    status.textContent = "Citation copied.";
  } catch {
    const range = document.createRange();
    range.selectNodeContents(code);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = "Citation selected. Use your browser's Copy command.";
  }
  resetTimer = setTimeout(() => {
    copyButton.querySelector("img").src = "assets/icons/copy.svg";
    status.textContent = "";
  }, 4000);
});
