// Studio panel. M5 ships the panel shell (jobs and notes live in app.js);
// generated outputs (reports, study tools, mind maps, audio) arrive in M6+.
export async function render(state) {
  const cards = document.getElementById("studio-cards");
  if (cards && !cards.childElementCount) {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = "Generated outputs (reports, flashcards, quizzes, mind maps, audio) appear here.";
    cards.append(p);
  }
  return state;
}
