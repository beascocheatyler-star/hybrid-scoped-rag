
var pass = 0, fail = 0;
function check(name, fn) {
  var ok = false, detail = "";
  try { ok = fn(); } catch (e) { ok = false; detail = "EXCEPTION: " + e; }
  print((ok ? "  PASS  " : "  FAIL  ") + name);
  if (!ok && detail) print("          " + detail);
  if (ok) pass++; else fail++;
}

/** Fake scroll container with controllable geometry. */
function makeMessages(scrollTop, scrollHeight, clientHeight) {
  return {
    scrollTop: scrollTop,
    scrollHeight: scrollHeight,
    clientHeight: clientHeight,
    lastScrollTo: null,
    scrollTo: function (opts) { this.lastScrollTo = opts; this.scrollTop = opts.top; }
  };
}

print("\nisPinnedToBottom");

check("true when exactly at the bottom", function () {
  el.messages = makeMessages(900, 1400, 500);
  return isPinnedToBottom() === true;
});

check("true within the threshold (slightly scrolled up)", function () {
  // 1400 - 820 - 500 = 80, under the 120px threshold.
  el.messages = makeMessages(820, 1400, 500);
  return isPinnedToBottom() === true;
});

check("false when scrolled well up", function () {
  // 1400 - 100 - 500 = 800, far past the threshold.
  el.messages = makeMessages(100, 1400, 500);
  return isPinnedToBottom() === false;
});

check("true when content is shorter than the viewport", function () {
  el.messages = makeMessages(0, 300, 500);
  return isPinnedToBottom() === true;
});

print("\nscrollToBottom — the reported bug");

check("does NOT yank the view down while the user reads further up", function () {
  el.messages = makeMessages(100, 1400, 500);
  scrollToBottom(); // a new assistant message arrives
  return el.messages.scrollTop === 100 && el.messages.lastScrollTo === null;
});

check("follows along when the user is already at the bottom", function () {
  el.messages = makeMessages(900, 1400, 500);
  scrollToBottom();
  return el.messages.lastScrollTo !== null && el.messages.lastScrollTo.top === 1400;
});

check("force overrides the pinned check (user sent a message)", function () {
  el.messages = makeMessages(100, 1400, 500);
  scrollToBottom(true);
  return el.messages.lastScrollTo !== null && el.messages.lastScrollTo.top === 1400;
});

check("programmatic follow is instant, not animated", function () {
  el.messages = makeMessages(900, 1400, 500);
  scrollToBottom();
  return el.messages.lastScrollTo.behavior === "auto";
});

check("user-initiated jump is smooth", function () {
  el.messages = makeMessages(100, 1400, 500);
  scrollToBottom(true, true);
  return el.messages.lastScrollTo.behavior === "smooth";
});

print("\nscrollToTop");

check("scrolls to 0 smoothly", function () {
  el.messages = makeMessages(900, 1400, 500);
  scrollToTop();
  return el.messages.lastScrollTo.top === 0 &&
         el.messages.lastScrollTo.behavior === "smooth";
});

print("\nupdateScrollButtons visibility");

function buttonState(scrollTop, scrollHeight, clientHeight) {
  el.messages = makeMessages(scrollTop, scrollHeight, clientHeight);
  el.scrollTop = { hidden: null };
  el.scrollBottom = { hidden: null };
  updateScrollButtons();
  return { top: el.scrollTop.hidden, bottom: el.scrollBottom.hidden };
}

check("both hidden in a short conversation", function () {
  var s = buttonState(0, 300, 500);
  return s.top === true && s.bottom === true;
});

check("at bottom of a long chat: top shown, bottom hidden", function () {
  var s = buttonState(900, 1400, 500);
  return s.top === false && s.bottom === true;
});

check("at top of a long chat: top hidden, bottom shown", function () {
  var s = buttonState(0, 1400, 500);
  return s.top === true && s.bottom === false;
});

check("mid-scroll in a long chat: both shown", function () {
  var s = buttonState(600, 1400, 500);
  return s.top === false && s.bottom === false;
});

print("\n" + pass + " passed, " + fail + " failed");
if (fail > 0) { throw new Error(fail + " assertion(s) failed"); }
