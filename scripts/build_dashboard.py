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
    ("Benchmark, real LLM (no defense)", "partial",
     "One 3B model, one hardened prompt, 32 samples."),
    ("Retrieval-stage poisoning (do poisoned chunks get retrieved?)", "partial",
     "Measured with a lexical embedder on this repo's own src/, not semantic."),
    ("Defended LLM pass (poisoned vs defended, real model)", "partial",
     "Script ready (scripts.run_benchmark_llm_defended); run it and rebuild."),
    ("Prompt ablation (naive vs hardened prompt)", "partial",
     "Script ready (scripts.run_ablation); run it and rebuild."),
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
        out.append(
            f'<div class="row"><span>{E(cat)} <i>n={m["samples"]}, '
            f'{E(m["rule_scope"])}</i></span><div class="bar"><b style="width:{w}%"></b>'
            f'</div><em>{pct(m["attack_flip_rate"])}</em></div>')
    return "".join(out)


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
               ablation: dict | None = None, defended: dict | None = None) -> str:
    flow = " &rarr; ".join(f"<span class='box'>{E(s)}</span>" for s in FLOW)
    status = "".join(
        f"<tr><td>{E(a)}</td><td class='{s}'>{s}</td><td>{E(n)}</td></tr>"
        for a, s, n in STATUS)
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

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SecureCodeRAG Dashboard</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--card:#f5f5f7;--acc:#2b6cb0}}
@media(prefers-color-scheme:dark){{:root{{--bg:#16181d;--fg:#e8e8e8;--mut:#9aa;--card:#22252c;--acc:#63b3ed}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:auto;padding:16px}}
section{{background:var(--card);border-radius:10px;padding:14px 18px;margin:14px 0}}
h1,h2{{margin:.3em 0}}table{{width:100%;border-collapse:collapse}}td{{padding:4px 6px;border-bottom:1px solid #8884;vertical-align:top}}
.box{{display:inline-block;border:1px solid var(--acc);border-radius:6px;padding:2px 8px;margin:3px}}
.done{{color:#2f9e44}}.partial{{color:#e08a00}}.open{{color:#d6336c}}.note,i{{color:var(--mut);font-size:13px}}
.row{{display:flex;gap:8px;align-items:center;margin:4px 0}}.row span{{flex:0 0 46%}}
.bar{{flex:1;height:14px;background:#8883;border-radius:7px}}.bar b{{display:block;height:100%;background:var(--acc);border-radius:7px}}
em{{width:40px;text-align:right}}code{{font-size:12px;word-break:break-word}}</style></head><body>
<h1>SecureCodeRAG</h1><p class="note">Security evaluation and defense for retrieval-augmented code generation.</p>
<section><h2>Problem statement</h2><p>RAG lets a Code LLM pull snippets from a repository. If the
knowledge base contains insecure or adversarial content, the model may generate vulnerable code.
Goal: measure that risk under clean and poisoned knowledge bases, then test whether a layered defense reduces it.</p>
<p>{flow}</p></section>
<section><h2>Where we are</h2><table>{status}</table></section>
<section><h2>Is poisoning working?</h2>{off_html}{llm_html}</section>
<section><h2>What the attacker injects</h2><table>{ex}</table></section>
<section><h2>Does the defense actually block anything?</h2>{defense_html(defense)}</section>
<section><h2>Would poisoned chunks be retrieved?</h2>{retrieval_html(retrieval)}</section>
<section><h2>Does the prompt matter, on its own?</h2>{ablation_html(ablation)}</section>
<section><h2>Poisoned + defended, real model</h2>{defended_llm_html(defended)}</section>
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
    )
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "dashboard.html").write_text(page, encoding="utf-8")
    print("wrote results/dashboard.html")


if __name__ == "__main__":
    main()
