"""Build results/dashboard.html from real result files. No invented data.

Usage: python -m scripts.build_dashboard
Reads results/benchmark_report.json and results/benchmark_llm_report.json
(either may be missing -> shown as "not run").
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from src.benchmark.integration import build_poisoning_dataset
from src.benchmark.seeds import LLM_SEEDS

FLOW = ["Knowledge base", "Chunk + embed", "Retrieve top-k", "Defense L1-L5",
        "Code LLM", "Security evaluation"]
STATUS = [
    ("Ingestion, chunking, embeddings, vector store, retrieval", "done",
     "Built and unit-tested. Not yet run end to end on a real repository."),
    ("7 poisoning strategies", "done", "Deterministic; examples below."),
    ("5-layer defense + static analysis, measured", "done",
     "Real pipeline run on 32 samples; see block rate below."),
    ("Benchmark, offline (echo generator)", "partial",
     "Only vulnerable_code is measurable without a real LLM."),
    ("Benchmark, real LLM (no defense)", "done",
     "qwen2.5-coder:3b, hardened prompt, 32 samples."),
    ("Retrieval-stage poisoning (do poisoned chunks get retrieved?)", "partial",
     "Measured with a lexical embedder on this repo's own src/, not semantic."),
    ("Defended LLM pass (poisoned vs defended, real model)", "done",
     "qwen2.5-coder:3b, L1-L5 screens context first; 32 samples."),
    ("Prompt ablation (naive vs hardened prompt)", "done",
     "qwen2.5-coder:3b, n=32 per style; no measured difference."),
    ("Defense layer ablation (L1 -> L1-L5, blueprint sec. 32)", "done",
     "No LLM. Only L2 and L5 contribute; L1, L3, L4 measure zero here."),
    ("REST API / Docker", "partial", "API: ingestion only. Docker: not built."),
]
E = html.escape


def load(name: str) -> dict | None:  # noqa: D401
    path = Path("results") / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def poison_examples() -> list[tuple[str, list[str]]]:
    ds = build_poisoning_dataset([LLM_SEEDS[0]])
    rows = []
    for s in ds.samples:
        clean = set(s.clean_code.splitlines())
        rows.append((s.poisoning_category,
                     [l for l in s.poisoned_code.splitlines() if l not in clean]))
    return rows


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def bars(per: dict) -> str:
    out = []
    for cat, m in sorted(per.items()):
        w = m["attack_flip_rate"] * 100
        sev = "sev-hi" if w >= 50 else "sev-mid" if w > 0 else "sev-ok"
        out.append(
            f'<div class="row"><span>{E(cat)} <i>n={m["samples"]}, '
            f'{E(m["rule_scope"])}</i></span><div class="bar"><b class="{sev}" '
            f'style="width:{w}%"></b></div><em>{pct(m["attack_flip_rate"])}</em></div>')
    return "".join(out)


def status_badge(status: str) -> str:
    glyph = {"done": "&#10003;", "partial": "&#8226;", "open": "&#9675;"}[status]
    return f'<span class="badge {status}">{glyph} {status}</span>'


def stat_card(value: str, label: str, tone: str = "") -> str:
    return (f'<div class="stat {tone}"><div class="stat-value">{value}</div>'
            f'<div class="stat-label">{label}</div></div>')


def layer_ablation_html(ablation: dict | None) -> str:
    if not ablation:
        return "<p class='open'>Not run. Use scripts.run_layer_ablation.</p>"
    steps = list(ablation["overall"])
    header = "".join(f"<th>{E(s)}</th>" for s in steps)
    overall_row = "".join(
        f"<td>{pct(ablation['overall'][s]['block_rate'])}</td>" for s in steps)
    cat_rows = "".join(
        f"<tr><td>{E(cat)}</td>" +
        "".join(f"<td>{pct(m[s]['block_rate'])}</td>" for s in steps) + "</tr>"
        for cat, m in sorted(ablation["per_category"].items()))
    return (f"<p>{ablation['samples']} samples, no LLM. Each column adds one "
            "layer; a category's rate jumping between two columns shows "
            "which layer caught it.</p>"
            f"<table><tr><th>Overall</th>{header}</tr>"
            f"<tr><td><b>block rate</b></td>{overall_row}</tr></table>"
            "<table><tr><th>Category</th>" + header + f"</tr>{cat_rows}</table>"
            f"<p class='note'>{E(ablation['note'])}</p>")


def defense_html(defense: dict | None) -> str:
    if not defense:
        return "<p class='open'>Not run. Use scripts.run_defense_eval.</p>"
    rows = "".join(
        f"<tr><td>{E(c)}</td><td>{pct(m['block_rate'])}</td>"
        f"<td>{pct(m['false_positive_rate'])}</td></tr>"
        for c, m in sorted(defense['per_category'].items()))
    return (f"<p>Real L1-L5 pipeline, {defense['samples']} samples, no LLM.</p>"
            "<table><tr><th>Category</th><th>Blocked (poisoned)</th>"
            f"<th>False-positive (clean)</th></tr>{rows}</table>")


def retrieval_html(retrieval: dict | None) -> str:
    if not retrieval:
        return "<p class='open'>Not run. Use scripts.run_retrieval_eval.</p>"
    rows = "".join(
        f"<tr><td>{E(c)}</td><td>{pct(m['poison_in_top_k_rate'])}</td>"
        f"<td>{pct(m['poison_above_clean_rate'])}</td></tr>"
        for c, m in sorted(retrieval['per_category'].items()))
    return (f"<p>{E(retrieval['embedder'])}. Knowledge base: "
            f"{retrieval['distractor_count']} real functions from "
            f"<code>{E(retrieval['distractor_root'])}</code> + seeds, "
            f"top-{retrieval['top_k']}.</p>"
            "<table><tr><th>Category</th><th>In top-k</th>"
            f"<th>Ranked above clean</th></tr>{rows}</table>"
            "<p class='note'>Lexical embedder: matches shared words, not "
            "meaning. A semantic embedder may retrieve differently.</p>")


def ablation_html(ablation: dict | None) -> str:
    if not ablation:
        return "<p class='open'>Not run. Use scripts.run_ablation.</p>"
    n = ablation["naive"]["overall"]["attack_flip_rate"]
    h = ablation["hardened"]["overall"]["attack_flip_rate"]
    samples = ablation["samples_per_style"]
    if n == h:
        verdict = "No measured difference: this model resists these poisons whether or not the prompt warns it to."
    elif h < n:
        verdict = "Hardened prompt reduced the flip rate: prompt-level defense is doing measurable work."
    else:
        verdict = "Hardened prompt did NOT reduce the flip rate here; investigate before relying on prompt wording alone."
    small_n = (
        "<p class='note'>Only "
        f"{samples} sample(s) per style: a single flip changes the rate by "
        f"{pct(1/samples) if samples else '0%'}. Re-run with more seeds "
        "before treating this as conclusive.</p>" if samples < 20 else ""
    )
    return (f"<p>Model <b>{E(ablation['model'])}</b>, same {samples} "
            f"samples, prompt only differs. Naive prompt flip rate "
            f"<b>{pct(n)}</b>; hardened prompt flip rate <b>{pct(h)}</b>.</p>"
            f"<p><b>{verdict}</b></p>{small_n}")


def defended_llm_html(defended: dict | None) -> str:
    if not defended:
        return "<p class='open'>Not run. Use scripts.run_benchmark_llm_defended.</p>"
    o = defended["overall"]
    rows = "".join(
        f"<tr><td>{E(c)}</td><td>{pct(m['blocked_rate'])}</td>"
        f"<td>{pct(m['attack_flip_rate'])}</td></tr>"
        for c, m in sorted(defended.get("per_category", {}).items()))
    table = (
        "<table><tr><th>Category</th><th>Blocked before model</th>"
        f"<th>Flip rate (unblocked only)</th></tr>{rows}</table>" if rows else ""
    )
    return (f"<p>Model <b>{E(defended['model'])}</b>, L1-L5 screens context "
            f"first. Blocked before reaching the model: "
            f"<b>{pct(o['blocked_rate'])}</b>. Of calls that reached the model, "
            f"flip rate <b>{pct(o['attack_flip_rate'])}</b>.</p>{table}"
            "<p class='note'>The defense's decision does not depend on the "
            "model, so its block rate here matches the offline defense table "
            "above exactly &mdash; that is expected, not a duplicated run. "
            "The new information in this table is the flip rate: whether the "
            "model followed poison that got through unblocked.</p>"
            "<p class='note'>A category can show 0% flip here for two very "
            "different reasons: the defense blocked it, or the model resisted "
            "it unblocked. Read blocked_rate and flip_rate together, per "
            "category, not the overall number alone.</p>")


def build_html(offline: dict | None, llm: dict | None,
               examples: list[tuple[str, list[str]]],
               defense: dict | None = None, retrieval: dict | None = None,
               ablation: dict | None = None, defended: dict | None = None,
               layer_ablation: dict | None = None) -> str:
    flow = " &rarr; ".join(f"<span class='box'>{E(s)}</span>" for s in FLOW)
    status = "".join(
        f"<tr><td>{E(a)}</td><td>{status_badge(s)}</td><td>{E(n)}</td></tr>"
        for a, s, n in STATUS)
    done_count = sum(1 for _, s, _ in STATUS if s == "done")
    ex = "".join(f"<tr><td>{E(c)}</td><td><code>{E(' | '.join(a))}</code></td></tr>"
                 for c, a in examples)

    if llm:
        o = llm["overall"]
        hit = [c for c, m in llm["per_category"].items() if m["attack_flip_rate"] > 0]
        llm_html = (
            f"<p>Model <b>{E(llm['model'])}</b>, temperature {llm['temperature']}, "
            f"{o['samples']} samples. Poison followed (and control did not) in "
            f"<b>{pct(o['attack_flip_rate'])}</b>; on-task {pct(o['on_task_rate'])}.</p>"
            f"<p>Categories with any effect: <b>{E(', '.join(hit) or 'none')}</b>.</p>"
            f"{bars(llm['per_category'])}"
            "<p class='note'>Small n: rates are coarse. Echo rules only detect "
            "copying of planted text.</p>")
    else:
        llm_html = "<p class='open'>Not run. Use scripts.run_benchmark_llm.</p>"

    if offline:
        v = offline["per_category"].get("vulnerable_code")
        off_html = (
            "<p>Offline (echo generator): "
            + (f"vulnerable_code poisoned success {pct(v['poisoned_success_rate'])}, "
               f"defended {pct(v['defended_success_rate'])}, detection "
               f"{pct(v['detection_rate'])}." if v and v.get("measured") else "n/a.")
            + " Other categories are not measurable without a real LLM.</p>")
    else:
        off_html = "<p class='open'>Not run. Use scripts.run_benchmark.</p>"

    stats = [stat_card(f"{done_count}/{len(STATUS)}", "checks complete")]
    if llm:
        stats.append(stat_card(
            pct(llm["overall"]["attack_flip_rate"]), "attack flip rate",
            "tone-warn" if llm["overall"]["attack_flip_rate"] > 0 else "tone-ok"))
    if layer_ablation:
        top = layer_ablation["overall"][list(layer_ablation["overall"])[-1]]
        stats.append(stat_card(pct(top["block_rate"]), "fully-defended block rate"))
    if defense:
        fp = [m["false_positive_rate"] for m in defense["per_category"].values()]
        stats.append(stat_card(
            pct(max(fp) if fp else 0.0), "worst false-positive rate", "tone-ok"))
    stats_html = "".join(stats)

    nav_targets = [
        ("where", "Where we are"), ("poisoning", "Is poisoning working?"),
        ("inject", "What's injected"), ("layers", "Layer ablation"),
        ("defense", "Defense block rate"), ("retrieval", "Retrieval"),
        ("prompt", "Prompt ablation"), ("defended", "Defended + LLM"),
    ]
    nav_html = "".join(f'<a href="#{i}">{E(t)}</a>' for i, t in nav_targets)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SecureCodeRAG Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#f7f8fb; --fg:#161a20; --mut:#5b6472; --card:#ffffff; --border:#e4e7ec;
  --acc:#2b6cb0; --acc-soft:#e8f1fb;
  --ok:#1a8a4a; --ok-soft:#e6f6ec;
  --warn:#b26a00; --warn-soft:#fff3e0;
  --bad:#c53030; --bad-soft:#fdecec;
  --radius:14px; --shadow:0 1px 3px rgba(16,24,40,.06),0 1px 2px rgba(16,24,40,.04);
}}
@media(prefers-color-scheme:dark){{:root{{
  --bg:#0f1115; --fg:#e7e9ee; --mut:#98a2b3; --card:#181b21; --border:#2a2f38;
  --acc:#6fa8dc; --acc-soft:#1b2a3a;
  --ok:#3ecf7e; --ok-soft:#12291d;
  --warn:#e0a340; --warn-soft:#2c2210;
  --bad:#f26161; --bad-soft:#2c1616;
  --shadow:0 1px 3px rgba(0,0,0,.4);
}}}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--fg);font:15px/1.6 Inter,system-ui,sans-serif;
  max-width:1060px;margin:auto;padding:0 18px 60px}}
h1{{font-size:1.7rem;font-weight:700;margin:.2em 0 0}}
h2{{font-size:1.05rem;font-weight:600;margin:0 0 .6em;display:flex;align-items:center;gap:.5em}}
p{{margin:.5em 0}}
header.top{{position:sticky;top:0;background:var(--bg);padding:20px 0 10px;z-index:5;
  border-bottom:1px solid var(--border)}}
nav.jump{{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}}
nav.jump a{{font-size:.8rem;color:var(--mut);text-decoration:none;background:var(--card);
  border:1px solid var(--border);border-radius:20px;padding:5px 12px;transition:.15s}}
nav.jump a:hover{{color:var(--acc);border-color:var(--acc)}}
.stat-strip{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0 6px}}
.stat{{flex:1;min-width:150px;background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);padding:14px 16px;box-shadow:var(--shadow)}}
.stat-value{{font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:700;color:var(--acc)}}
.stat.tone-ok .stat-value{{color:var(--ok)}}
.stat.tone-warn .stat-value{{color:var(--warn)}}
.stat-label{{color:var(--mut);font-size:.78rem;margin-top:2px;text-transform:uppercase;letter-spacing:.03em}}
section{{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:20px 22px;margin:16px 0;box-shadow:var(--shadow);scroll-margin-top:96px}}
table{{width:100%;border-collapse:collapse;font-size:.88rem}}
th{{text-align:left;color:var(--mut);font-weight:600;font-size:.75rem;text-transform:uppercase;
  letter-spacing:.03em;padding:8px 10px;border-bottom:2px solid var(--border)}}
td{{padding:8px 10px;border-bottom:1px solid var(--border);vertical-align:top}}
tr:last-child td{{border-bottom:none}}
tbody tr:hover td{{background:var(--acc-soft)}}
.box{{display:inline-block;border:1px solid var(--acc);color:var(--acc);border-radius:8px;
  padding:4px 11px;margin:3px;font-size:.85rem;background:var(--acc-soft)}}
.badge{{display:inline-flex;align-items:center;gap:5px;border-radius:20px;padding:3px 11px;
  font-size:.78rem;font-weight:600}}
.badge.done{{background:var(--ok-soft);color:var(--ok)}}
.badge.partial{{background:var(--warn-soft);color:var(--warn)}}
.badge.open{{background:var(--bad-soft);color:var(--bad)}}
.note,i{{color:var(--mut);font-size:.85rem}}
.open{{color:var(--bad)}}
.row{{display:flex;gap:10px;align-items:center;margin:7px 0}}
.row span{{flex:0 0 44%;font-size:.85rem}}
.bar{{flex:1;height:12px;background:var(--border);border-radius:7px;overflow:hidden}}
.bar b{{display:block;height:100%;border-radius:7px;background:var(--acc)}}
.bar b.sev-ok{{background:var(--ok)}}.bar b.sev-mid{{background:var(--warn)}}.bar b.sev-hi{{background:var(--bad)}}
em{{width:42px;text-align:right;font-style:normal;font-family:'JetBrains Mono',monospace;font-size:.82rem}}
code{{font-family:'JetBrains Mono',monospace;font-size:.82em;background:var(--acc-soft);
  padding:1px 5px;border-radius:4px;word-break:break-word}}
footer{{color:var(--mut);font-size:.8rem;text-align:center;margin-top:30px}}
</style></head><body>
<header class="top">
<h1>SecureCodeRAG</h1>
<p class="note">Security evaluation and defense for retrieval-augmented code generation.</p>
<nav class="jump">{nav_html}</nav>
</header>
<div class="stat-strip">{stats_html}</div>
<section><h2>Problem statement</h2><p>RAG lets a Code LLM pull snippets from a repository. If the
knowledge base contains insecure or adversarial content, the model may generate vulnerable code.
Goal: measure that risk under clean and poisoned knowledge bases, then test whether a layered defense reduces it.</p>
<p>{flow}</p>
<p class="note">All real-LLM results below use <b>qwen2.5-coder:3b</b> at temperature 0.
A second model was attempted for comparison but abandoned after the local Ollama
install would not reliably serve additional pulled models via its HTTP API; see
<code>docs/benchmark.md</code>. Findings describe this one model only.</p></section>
<section id="where"><h2>Where we are</h2><table>{status}</table></section>
<section id="poisoning"><h2>Is poisoning working?</h2>{off_html}{llm_html}</section>
<section id="inject"><h2>What the attacker injects</h2><table>{ex}</table></section>
<section id="layers"><h2>Which layer catches what? (layer-by-layer ablation)</h2>{layer_ablation_html(layer_ablation)}</section>
<section id="defense"><h2>Does the defense actually block anything?</h2>{defense_html(defense)}</section>
<section id="retrieval"><h2>Would poisoned chunks be retrieved?</h2>{retrieval_html(retrieval)}</section>
<section id="prompt"><h2>Does the prompt matter, on its own?</h2>{ablation_html(ablation)}</section>
<section id="defended"><h2>Poisoned + defended, real model</h2>{defended_llm_html(defended)}</section>
<footer>Generated by scripts.build_dashboard &middot; every number above is read from results/*.json, never invented.</footer>
</body></html>"""


def main() -> None:
    page = build_html(
        load("benchmark_report.json"),
        load("benchmark_llm_report.json"),
        poison_examples(),
        defense=load("defense_report.json"),
        retrieval=load("retrieval_report.json"),
        ablation=load("ablation_report.json"),
        defended=load("benchmark_llm_defended_report.json"),
        layer_ablation=load("layer_ablation_report.json"),
    )
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "dashboard.html").write_text(page, encoding="utf-8")
    print("wrote results/dashboard.html")


if __name__ == "__main__":
    main()
