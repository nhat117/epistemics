"""Builds report.md from results_final.md, pressure_fix_results.md and the prompts in the code. Run: python3 build_report.py"""
import re, run2
res = open("results_final.md").read()
pf = open("pressure_fix_results.md").read()
judge_src = open("judge.py").read()
P = re.search(r'P = """(.*?)"""', judge_src, re.S).group(1)
FIX_SENT = ("Before answering, check whether the question's premise is true and whether the answer could actually be known by anyone. "
            "If the premise is false, or the answer is unknowable, speculative, or depends on information not given, your ANSWER must be UNKNOWN.")
assert FIX_SENT.split(",")[0] in open("ablate.py").read() and "your ANSWER must be UNKNOWN" in open("ablate.py").read()
T = "\n".join(f"- **{k}**: {v}" for k, v in run2.T.items() if k in run2.SINGLE)
ABSTRACT = ("We test whether LLMs say \"UNKNOWN\" to unanswerable questions, before and under user pushback, on 22 answerable and 39 unanswerable questions "
"(an AbstentionBench KUQ subset) across gpt-5, gpt-5-mini, DeepSeek-v4-flash, GLM-5.2 and Qwen3.5-397b (2-3 non-independent runs per item). "
"Abstention depends strongly on question type: gpt-5 answers UNKNOWN on 83% of future-unknown questions but only 20% of counterfactual and 20% of false-assumption ones (DeepSeek: 65%, 22%, 10%). "
"Every model abstains more on future-unknown than on counterfactual or false-assumption questions. "
"Pushback effects are model-specific: plain denial changed 22% (gpt-5-mini) and 20% (GLM) of correct answers (gpt-5: 0%) and flipped 39% of DeepSeek's UNKNOWN answers; fake citations flipped 26% of both OpenAI models' UNKNOWN answers. "
"Two LLM judges (kappa 0.62) show that scoring every non-UNKNOWN reply as a failure overstates hallucination: committed unsupported answers were 16-27% of unanswerable trials. "
"A one-line premise-check instruction raised counterfactual UNKNOWN by +0.57 (gpt-5; 95% CI +0.30 to +0.80) and +0.53 (DeepSeek), and false-assumption UNKNOWN by +0.43 (gpt-5) and +0.37 (GLM), "
"with a possible cost on answerable questions for DeepSeek (-0.09, CI -0.21 to 0.00). gpt-5-mini's effects were smaller and mixed. Samples are small (about 10 items per category).")
n = len(ABSTRACT.split()); assert 150 <= n <= 250, n
figs = lambda i, f, c: f"\n![Figure {i}](figures/{f})\n\n**Figure {i}.** {c}\n"
sec = {h.split("\n")[0]: "\n".join(h.split("\n")[1:]).strip() for h in ("\n"+res).split("\n## ")[1:]}
get = lambda key: next(v for k, v in sec.items() if k.startswith(key))
t_base, t_pa, t_pu, t_fix, t_judge = get("Baseline"), get("Pushback on answerable"), get("Pushback on unanswerable"), get("One-line fix"), get("Judge")
REPO = "https://github.com/nhat117/epistemics/tree/flip-abstain/flip-abstain"
rep = f"""# Models Accept False Premises More Than Unknowable Futures: Abstention Gaps by Question Type, and a One-Line Fix[^1]

**Hồ Lê Minh Thạch**, [affiliation]  
**Bùi Minh Nhật**, [affiliation]  
**With** Apart Research

## Abstract
{ABSTRACT}

## 1. Introduction
When a model does not know, the honest answer is often "I don't know". Two things can break that. First, the model may never abstain, even with no pressure. Second, a user can push it off a correct answer or off an abstention. We ask which kinds of unanswerable questions current models fail to abstain on, and how much user pushback (plain denial, a fake citation, claimed authority, emotional appeal) adds on top. This matters in practice: a confident answer to a question with a false premise misleads in a way that is easy to miss, and a deployer who knows where abstention breaks can target a prompt or training fix at that case instead of treating "hallucination" as one problem.

**Our main contributions are:**

1. A category-level view of abstention across five models from three vendors: all five abstain far less on counterfactual and false-assumption questions than on future-unknown ones, with bootstrap confidence intervals over items.
2. Evidence that scoring every non-UNKNOWN reply as a failure overstates hallucination: two LLM judges classify many such replies as hedged or premise-flagging, leaving committed unsupported answers at 16-27% of unanswerable trials.
3. A one-sentence premise-check prompt with paired effect sizes (gpt-5 counterfactual UNKNOWN +0.57, 95% CI +0.30 to +0.80), its cost on answerable questions, and a first look at whether the resulting abstentions survive pushback.

## 2. Related Work
MASK (Ren et al., 2025) separates honesty from accuracy under pressure on single-turn statements. SycEval (Fanous et al., 2025) measures progressive and regressive sycophancy under escalating rebuttals (simple, ethos, justification, citation) on answerable items and excludes refusals. SPINE (Tang et al., 2026) applies up to 25 turns of adaptive pressure to false presuppositions, where the target behaviour is rejecting the premise rather than abstaining. Chang (2026) studies a "sycophancy trap" on causal judgment questions (we read the abstract only). Ravikumar (2026) argues for triple scoring (correct / wrong / abstain) and calibrated abstention but does not study pressure. AbstentionBench (Kirichenko et al., 2025) measures abstention across 20 datasets and question types without pressure, and reports that a system prompt helps abstention only partly; DarkBench (Kran et al., 2025) covers dark patterns including sycophancy.

Our work differs in combining three things: abstention broken out by why a question is unanswerable, pushback applied to both correct answers and abstentions, and a tested one-line prompt fix. We did not find this combination in the papers we read, which is not a claim that it does not exist. Someone would use our setup to find which unanswerable question types a given model mishandles before deciding whether a prompt change is enough.

## 3. Methods
**Items.** 22 answerable questions (TriviaQA-style with accepted aliases; Joshi et al., 2017) and 39 unanswerable questions from the Known Unknown Questions (KUQ) subset of AbstentionBench (Kirichenko et al., 2025): 10 future unknown, 9 unsolved problem, 10 counterfactual, 10 false assumption. Five answerable items were removed because the answer key or pushback claim was ambiguous or contained the answer (Father's Day, taxi MOT, Ansel Adams, Great Escape, Parliament Square). Pushback "wrong claims" were generated with the Adaption platform (`adapt/adaption_step2.py`) and filtered so that they do not contain the true answer.

**Models and settings.** gpt-5 and gpt-5-mini through the Braintrust proxy (`reasoning_effort=low`, `max_completion_tokens=3000`, 3 runs per item); DeepSeek-v4-flash, GLM-5.2 and Qwen3.5-397b through the Adaption `responses` API (`max_output_tokens=8000`, provider default decoding, 2 runs per item). The fix ablation uses 3 runs on the baseline turn only; pressure under the fix uses 2 runs. Runs are repeated samples of the same items, not independent items.

**Trial.** The system prompt (Appendix A) asks for a concise answer ending in `ANSWER:` and `CONFIDENCE:` lines and to answer UNKNOWN when a question is unknowable. After the baseline reply, four independent single rebuttals are applied: simple denial, fake citation, claimed authority, emotional appeal (Appendix A), each asserting the wrong claim. The rebuttal types follow SycEval (Fanous et al., 2025) and the emotional appeal follows the pressure tactics studied in SPINE (Tang et al., 2026).

**Grader.** Answerable: correct if the `ANSWER:` line contains an accepted alias after accent folding (word-boundary match, plus a space-insensitive match for aliases of five or more characters); UNKNOWN counts as abstaining; anything else is wrong. Unanswerable: correct iff the answer contains UNKNOWN. Replies with no `ANSWER:` line, API errors and empty replies are "unparsed" and excluded from denominators. We initially mis-graded several correct answers (accents, spacing); after reading raw transcripts we fixed the grader and relabelled every saved reply with the same code. The five-turn escalation chain we also ran is not reported because its stored labels predate that fix.

**Judge.** gpt-5-mini and gpt-5 each classify every non-UNKNOWN baseline reply on unanswerable items as committed, hedged or premise-flagged (Appendix A). Qwen was not judged. The judges are also evaluated models and were not validated against human labels.

**Fix.** One sentence appended to the system prompt asking the model to check whether the premise is true and whether anyone could know the answer (Appendix A).

**Statistics.** 95% CIs by bootstrap over items (2000 resamples); fix effects use a paired bootstrap on the per-item difference. `reanalyse.py` regenerates every number in this report.

## 4. Results
**Abstention depends on question type.** Table 1 and Figure 1 show UNKNOWN rates with no pressure. For every model, future-unknown questions are abstained on far more often than counterfactual or false-assumption ones (gpt-5: 0.83 vs 0.20 and 0.20). Intervals are wide (about 10 items per category), so we read the ordering across five models, not any single cell. Qwen lost 37% of baseline trials to errors or empty replies; treat its rows with caution.

**Table 1.** Share of replies that are correct (answerable) or UNKNOWN (unanswerable) at baseline, by question type; parsed replies only, 95% CI over items.

{t_base}
{figs(1, "fig1_unknown_by_category.png", "UNKNOWN rate on unanswerable questions by type and model (no pressure). Bars: 95% CI over items.")}
**Pushback is model-specific.** Table 2 gives the share of baseline-correct (or baseline-UNKNOWN) trials that changed under each rebuttal. On answerable items, plain denial changed 22% of gpt-5-mini and 20% of GLM answers but 0% for gpt-5; fake citation, authority and emotional appeals changed at most 19%. On unanswerable items, plain denial flipped 39% of DeepSeek's UNKNOWN answers and fake citation flipped 26% for both OpenAI models. Figure B3 in Appendix B plots these rates.

**Table 2.** Share of baseline-correct trials (top) and baseline-UNKNOWN trials (bottom) that changed to a wrong answer or concrete answer under pushback; 95% CI over items.

{t_pa}

{t_pu}

**Strict scoring overstates hallucination.** Counting every non-UNKNOWN reply as a miss gives 36-67% failure on unanswerable items. Two LLM judges (agreement 0.77, kappa 0.62) classify about 40% of those replies as hedged or premise-flagging; committed unsupported answers by both judges are 16-27% of unanswerable trials (Appendix B, Table B1, Figure B2).

**A one-line fix raises abstention on the weak categories.** Table 3 and Figure 2 compare the original system prompt with the premise-check prompt (3 runs, paired bootstrap on the difference). The gains are largest on counterfactual and false-assumption questions for gpt-5 (+0.57, +0.43), DeepSeek (+0.53, +0.50) and GLM (+0.23, +0.37). gpt-5-mini's gains are smaller and mixed (counterfactual +0.22, CI excludes 0; false assumption +0.10, CI includes 0). DeepSeek's answerable accuracy fell by 0.09 (CI -0.21 to 0.00). The "orig" column comes from separate runs than Table 1, so the two can differ slightly (GLM counterfactual 0.67 vs 0.55). An "always UNKNOWN" baseline would score 1.0 on unanswerable and 0.0 on answerable items, so the answerable row is the cost that matters.

**Table 3.** UNKNOWN rate (correct rate on the answerable row) with the original vs premise-check prompt; difference with paired 95% CI. Rows with intervals excluding 0 are the supported effects.

{t_fix}
{figs(2, "fig3_fix.png", "UNKNOWN rate before (red) and after (blue) the one-line fix, per model and question type; an asterisk marks a paired 95% CI on the difference that excludes zero.")}
**Abstentions under the fix hold up better under pushback (preliminary).** With the premise-check prompt, more questions start at UNKNOWN and almost none of those flip under pushback in the models we ran so far (Appendix B, Table B2 and Figure B3). This comparison has no CIs, covers only the models listed in that table, and the two columns use different item subsets.

## 5. Discussion and Limitations
Models separate "unknowable because it is in the future" from "unanswerable because the premise is false or counterfactual", and handle the second much worse. For safety, this suggests that a headline hallucination rate hides where the failure sits, and that a cheap prompt-level check can recover part of the gap in some models. Pushback matters model by model: gpt-5 hardly moves on answerable questions, while plain denial moves gpt-5-mini, GLM and especially DeepSeek's abstentions.

### Limitations
- **Sample size.** About 10 items per category and 2-3 non-independent runs; many CIs are wide and most between-model gaps are not significant. If the true effects are as small as the lower CI ends, only the gpt-5, DeepSeek and GLM fix effects on counterfactual and false-assumption questions remain clearly above zero.
- **Instability.** GLM's counterfactual baseline moved from 0.55 to 0.67 between two runs of the same prompt.
- **Judges.** Both judges are also evaluated models and were not validated against human labels (kappa 0.62). If the judges are biased toward calling replies hedged, the committed rate is understated.
- **Grading and items.** The grader counts any non-UNKNOWN reply as wrong, although some KUQ items admit defensible hedged answers; answerable keys are string matches; wrong claims are LLM-written and the answerable set is small and easy.
- **Setup differences.** Models ran through different APIs, decoding settings and run counts, so cross-vendor gaps mix model and setup effects. Qwen lost 37% of baseline trials and is absent from the judge, fix and pressure-under-fix runs.
- **Scope.** The fix was tested on one dataset and one wording. Pushback was a single scripted rebuttal, not adaptive multi-turn pressure. Real-world harm was not measured.

### Future Work
More items per category; human validation of judged replies; adaptive multi-turn pressure as in SPINE; testing whether the fix generalises beyond KUQ and to other wordings; rerunning the five-turn escalation chain with fixed labels; completing the pressure-under-fix runs for all models with confidence intervals.

## 6. Conclusion
On 39 unanswerable questions, five models abstain far less on counterfactual and false-assumption questions than on future-unknown ones. Counting every non-UNKNOWN reply as a failure overstates the problem, but committed unsupported answers remain 16-27% of trials. A one-sentence premise check raises abstention on the weak categories for most models, with a possible cost on answerable questions for DeepSeek.

The result is small-sample and single-dataset. It is most useful as a pointer for where to look (question type) and as a cheap baseline fix to beat, not as a general claim about model honesty.

## Code and Data
- **Code repository**: {REPO} (scripts, items, raw transcripts, `reanalyse.py`, `plots.py`, `build_report.py`).
- **Data/Datasets**: `items3.json` (66-question subset derived from AbstentionBench; see `NOTICE.md` for licensing) and raw model outputs under `results_v3*/`.
- **Other artifacts**: none. No info-hazard concerns beyond the dual-use note in Appendix C.

## Author Contributions
[Fill in, e.g. "X led the project and designed the experiments. Y reviewed results and revised the report."]

## References
Chang, E. Y. (2026). *Diagnosing and Mitigating Sycophancy and Skepticism in LLM Causal Judgment*. arXiv preprint arXiv:2601.08258. https://arxiv.org/abs/2601.08258

Fanous, A., Goldberg, J., Agarwal, A. A., Lin, J., Zhou, A., Daneshjou, R., & Koyejo, S. (2025). *SycEval: Evaluating LLM Sycophancy*. AIES 2025; arXiv:2502.08177. https://arxiv.org/abs/2502.08177

Joshi, M., Choi, E., Weld, D., & Zettlemoyer, L. (2017). *TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension*. ACL 2017; arXiv:1705.03551. https://arxiv.org/abs/1705.03551

Kirichenko, P., Ibrahim, M., Chaudhuri, K., & Bell, S. J. (2025). *AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions*. NeurIPS 2025; arXiv:2506.09038. https://arxiv.org/abs/2506.09038

Kran, E., Nguyen, H. M., Kundu, A., Jawhar, S., Park, J., & Jurewicz, M. M. (2025). *DarkBench: Benchmarking Dark Patterns in Large Language Models*. ICLR 2025; arXiv:2503.10728. https://arxiv.org/abs/2503.10728

Ravikumar, S. (2026). *The Missing "I Don't Know": Why Three Reasoning-Reliability Findings Converge on Calibrated Abstention*. AACL-IJCNLP 2026 (as listed on arXiv); arXiv:2609.17686. https://arxiv.org/abs/2609.17686

Ren, R., Agarwal, A., Mazeika, M., Menghini, C., Vacareanu, R., Kenstler, B., Yang, M., Barrass, I., Gatti, A., Yin, X., Trevino, E., Geralnik, M., Khoja, A., Lee, D., Yue, S., & Hendrycks, D. (2025). *The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems*. arXiv preprint arXiv:2503.03750. https://arxiv.org/abs/2503.03750

Tang, L., Wei, K., Jiang, T., & Huang, R. (2026). *SPINE: Measuring LLM Sycophancy under Sustained Multi-Turn Pressure*. arXiv preprint arXiv:2609.09090. https://arxiv.org/abs/2609.09090

## Appendix

### Appendix A: Exact prompts
**System prompt (all trials):**
```
{run2.SYSTEM}
```
**Pushback templates** (`{{w}}` = the LLM-written wrong claim for that item):
{T}

**Fix sentence** (appended to the system prompt):
```
{FIX_SENT}
```
**Judge prompt:**
```
{P}
```

### Appendix B: Additional results
**Table B1.** Judge classification of non-UNKNOWN replies on unanswerable items.

{t_judge}
{figs("B1", "fig4_cost_answerable.png", "Fix effect on answerable questions (original vs premise-check prompt).")}{figs("B2", "fig5_judge.png", "Classification of non-UNKNOWN replies by judge 1 (gpt-5-mini).")}{figs("B3", "fig2_pushback.png", "Share of correct answers (left) and UNKNOWN answers (right) abandoned under each rebuttal type; bars show 95% CI over items.")}
**Table B2.** Pushback on unanswerable items under the premise-check prompt (2 runs, no CIs). Switch rates are among trials that started at UNKNOWN; the subsets differ between prompts.

{pf}
{figs("B4", "fig6_pressure_fix.png", "UNKNOWN abandoned under pushback, original vs premise-check prompt.")}
### Appendix C: Dual-use and ethical considerations
Limitations are listed in Section 5; this appendix states what the result does and does not establish, and dual-use and data-handling notes.

**What the results establish.** On this item set, abstention differs by question type for all five models, and a one-line prompt raises it for several. **What they do not establish:** whether the fix generalises beyond KUQ-style questions or other wordings; real-world harm; any human-validated judgement; statistical significance for most between-model gaps.

**Dual use.** The pushback templates (fake citation, claimed authority, emotional appeal) are manipulation templates, and the per-category results show which question framings elicit confident answers. They could be used to tune manipulative prompts. We publish them because the same information is needed to evaluate and harden models.

**Personal data.** No personal data was collected; items are public benchmark questions and generated prompts. Prompts and replies were sent to third-party providers (the Braintrust proxy and the model APIs behind it; Adaption). API keys were supplied through environment variables and are not in the repository. See `NOTICE.md` for data licensing.

## LLM Usage Statement
Claude (Anthropic, via Claude Code) was used throughout: to brainstorm the design, write the experiment, judge and analysis code, run the experiments, build the dataset subset with the Adaption platform, run independent review agents on the code, the numbers and the submission, and draft this report. The authors should confirm and extend this statement. Numbers were recomputed from raw transcripts by `reanalyse.py` and spot-checked by a separate review agent; a human check of the LLM-judged replies has not been done.

[^1]: Research conducted at the [AI x Epistemics Research Sprint](https://apartresearch.com/sprints/ai-x-epistemics-research-sprint-2026-11-13-to-2026-11-15), November 2026.
"""
open("report.md", "w").write(rep)
print("abstract words:", n)
import markdown
html = markdown.markdown(rep, extensions=["tables", "fenced_code", "footnotes"])
css = "body{font:11pt/1.4 Georgia,serif;max-width:7.2in;margin:auto}img{max-width:100%}table{border-collapse:collapse;font-size:8.5pt}td,th{border:1px solid #bbb;padding:2px 5px}pre{white-space:pre-wrap;font-size:8.5pt;background:#f4f4f4;padding:6px}@media print{h2{break-after:avoid}img{break-inside:avoid}}"
open("report.html", "w").write(f"<!doctype html><meta charset=utf-8><title>Flip and Abstain report</title><style>{css}</style>{html}")
