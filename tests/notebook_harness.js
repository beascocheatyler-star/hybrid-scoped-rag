// XSS and behaviour checks for the notebook UI renderers (web/js/markdown.js).
// Bundled by scripts/build_nb_test.py and run under Node or JavaScriptCore.
var pass = 0, fail = 0;
function ok(name, cond, got) {
  print((cond ? "  PASS  " : "  FAIL  ") + name);
  if (!cond) print("          got: " + JSON.stringify(got).slice(0, 240));
  if (cond) pass++; else fail++;
}
function noRawTags(html) {
  // Only the renderer's own tags may appear.
  var allowed = /^(p|span|button|mark|strong|em|code|pre|h[1-6]|ul|ol|li|table|thead|tbody|tr|th|td|blockquote|hr|a|div)$/;
  // Inspect real tags only: escaped text (&lt;img onerror=...&gt;) is inert.
  var re = /<\/?([a-zA-Z][a-zA-Z0-9]*)([^>]*)>/g, m;
  while ((m = re.exec(html))) {
    if (!allowed.test(m[1].toLowerCase())) return false;
    if (/\son[a-z]+\s*=/i.test(m[2]) || /javascript:/i.test(m[2])) return false;
  }
  return true;
}
var CITES = [{marker: 1, source_title: "syllabus.md", quote: "Exam is Feb 18.", weak: false, page: 2},
             {marker: 2, source_title: "<img src=x onerror=alert(1)>", quote: "<script>alert(1)</script>", weak: true,
              section_path: "<b>Grading</b>", t_start_ms: 65000}];

print("\ncitation chips");
var h = citationChips("Fact [1]. Other [2][9].", CITES);
ok("chips for known markers", (h.match(/class="cite/g) || []).length === 2, h);
ok("unknown marker dropped", h.indexOf("[9]") < 0 && h.indexOf('data-marker="9"') < 0, h);
ok("weak chip styled", h.indexOf('cite cite-weak" data-marker="2"') >= 0, h);
ok("title escaped in aria-label", h.indexOf("<img") < 0 && h.indexOf("&lt;img") >= 0, h);
ok("markdown link not mistaken for marker", citationChips("[1](https://x.com)", CITES).indexOf("<button") < 0);
ok("comma list", (citationChips("x [1, 2]", CITES).match(/<button/g) || []).length === 2);

print("\nanswer renderer");
var a = renderAnswer("Exam is Feb 18 [1]. Moon is cheese [2].", CITES,
  [{text: "Exam is Feb 18.", markers: [1], supported: true},
   {text: "Moon is cheese.", markers: [2], supported: false}]);
ok("unsupported sentence wrapped", a.indexOf('<span class="unsupported"') >= 0 && a.indexOf("Moon is cheese") > a.indexOf("unsupported"), a);
ok("supported sentence not wrapped", a.indexOf("Exam is Feb 18.") < a.indexOf("unsupported"), a);
var evil = renderAnswer("x", [], [{text: "<script>alert(1)</script> <img src=x onerror=alert(1)>", markers: [], supported: true}]);
ok("sentence text escaped", noRawTags(evil) && evil.indexOf("&lt;script&gt;") >= 0, evil);
var forged = renderAnswer("y", CITES, [{text: "a", markers: ['1" onmouseover="alert(1)'], supported: true}]);
ok("forged marker cannot inject attributes", noRawTags(forged), forged);
var block = renderAnswer("- item one [1]\n- item two [2]", CITES,
  [{text: "- item two", markers: [2], supported: false}]);
ok("block answers render as list with chips", block.indexOf("<ul>") >= 0 && (block.match(/<button/g) || []).length === 2, block);
ok("block answers count unverified", block.indexOf("1 sentence could not be verified") >= 0, block);
ok("block answer escapes html", noRawTags(renderAnswer("- <svg onload=alert(1)>", [], [])));
ok("null text safe", renderAnswer(null, [], []) === "");

print("\ncitation card");
var card = renderCitationCard(CITES[1]);
ok("card escapes title, quote, section", noRawTags(card) && card.indexOf("&lt;script&gt;") >= 0 && card.indexOf("&lt;b&gt;Grading") >= 0, card);
ok("card shows weak warning", card.indexOf("Weak support") >= 0);
ok("card shows timestamp", card.indexOf("1:05") >= 0, card);
ok("card shows page", renderCitationCard(CITES[0]).indexOf("p. 2") >= 0);
ok("card null safe", renderCitationCard(null) === "");

print("\nnotes");
var n = renderNote({title: "<script>x</script>", body_md: "Saved [1] answer <img src=x onerror=alert(1)>", citations: CITES});
ok("note title and body escaped", noRawTags(n) && n.indexOf("&lt;script&gt;x") >= 0, n);
ok("note keeps clickable citation chips", n.indexOf('data-marker="1"') >= 0, n);
ok("note without citations drops markers", renderNote({title: "t", body_md: "a [1]"}).indexOf("<button") < 0);

print("\nsource viewer highlight");
ok("highlight wraps span", renderHighlighted("abc<def>ghi", 3, 8) === 'abc<mark id="hl" tabindex="-1">&lt;def&gt;</mark>ghi');
ok("highlight escapes outside segments", renderHighlighted("<a>x<b>", 3, 4).indexOf("&lt;a&gt;") === 0);
ok("out-of-range clamps", renderHighlighted("abc", -5, 99).indexOf("<mark") >= 0);
ok("no span -> plain escaped", renderHighlighted("<x>", null, null) === "&lt;x&gt;");
ok("inverted span -> plain", renderHighlighted("abc", 2, 1) === "abc");

print("\nformatting");
ok("formatMs minutes", formatMs(65000) === "1:05");
ok("formatMs hours", formatMs(3725000) === "1:02:05");

print("\n" + pass + " passed, " + fail + " failed");
if (fail) { throw new Error(fail + " failed"); }
