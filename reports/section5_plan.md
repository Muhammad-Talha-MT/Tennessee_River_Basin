# Plan for Section 5 (Results)

Built on `reports/paper.tex` (Sections 1-4 as of 21 Nov 2026) and `reports/section5_facts.md`. Every number below is from the facts sheet.

## 1. Is the set of results the right one?

Mostly yes. The facts sheet answers each research question with one paired contrast, on the same gauges, in three protocols, with an equivalence test. That matches the design promised in Sections 1 and 4. Four things need fixing before the text is written.

1. **RQ2 is half missing.** Routing (M1 - M4) has no numbers until M4 is evaluated, so Section 5.3 cannot be written yet. Message passing (M0 - M2) alone cannot answer "message passing or routing".
2. **The results are mostly small or equivalent differences.** This is a legitimate answer to the title question, but only if it is written as an equivalence result:
   - M0 - M1 is within the 0.02 margin in the gauged and random protocols.
   - M0 - M2 is within 0.01 when gauged and changes sign across protocols.
   - M0 - LSTM is small when gauged and random and larger when blocked, where n = 29.

   Section 5 should report each contrast as "different", "equivalent within 0.02" or "undetermined". Writing "M0 is better" for a +0.002 difference would not survive review.
3. **The every-reach evaluation favors the reach-scale models.** Their stage A and stage-B prior are fitted to the same NWM v3.0 retrospective that serves as the reach-level reference. The LSTM never sees it, and at reaches it runs with flow history hidden. Figure 7 can only show how agreement with the NWM changes with drainage area. It is not an RQ1 test. This belongs in its own subsection with that caveat, or in the supplement.
4. **The KGE result reverses NSE in the blocked protocol.**
   - **Blocked:** LSTM KGE 0.381 and NWM 0.385 against M0 0.334.
   - **Variability ratio:** M0 alpha is 0.530 against 0.626 for the LSTM.
   - **Section 2.4 already promised this check:** "NSE can reward forecasts that underestimate variability". It has to be reported, not left in Table 3.

## 2. Inconsistencies with Sections 1-4 to fix first

| Where | Issue | Fix |
|---|---|---|
| Section 1, RQ1 (line 107) | "a lumped LSTM with matched inputs and outputs" contradicts Section 4.4 ("It is not matched to them input for input") | "... than a lumped regional LSTM driven by the same meteorological products" |
| Section 1 (line 105) | "evaluated at gauges withheld at random, and ... entire basins" is fine; "the model and its ablations" implies M4 everywhere | add M4's coverage (random and blocked only) once its evaluation is done |
| Commented headings (line 402-405) | use H1-H3; the paper uses RQ1-RQ3 | use the RQ structure below |
| Section 4.5 VERIFY (line 390) | M4 coverage | settle after the M4 evaluation |
| Figure files | the paper's Figure 3 is `fig4_model.pdf`; the results figures are `fig4_skill_lead` ... `fig7_every_reach` | rename the schematic to `fig3_model.pdf`, then the results figures 4-7 match their numbers |
| Section 3.5 (line 222) | "we compare results by position only at similar drainage areas" | Section 5.4 must use the area-matched comparison (5c), not the raw nested/headwater split |

## 3. Structure of Section 5

One subsection per research question, in the order of Section 1. Each opens with the one-sentence answer, then the number, n, interval and test, then the lead dependence.

### 5.1 Overall skill and the NWM reference (Figure 4, Table 3)
- **Purpose:** context, not a research question. Lead curves show that all reach-scale variants and the LSTM lie close together in every protocol, and that skill falls with strictness of the hold-out.
- **Numbers:** days 1-5 median NSE for M0 is 0.430, 0.385 and 0.305 (gauged, random, blocked); the LSTM is 0.428, 0.367 and 0.280.
- **NWM:** it is lower in NSE (0.255, 0.255, 0.183) but competitive in KGE. It is nudged to observations, so it is a reference, not a competitor. It is better than M0 at 6, 17 and 5 gauges.
- **Order:** state the KGE reversal here (point 4 above), because it qualifies every later NSE statement.

### 5.2 RQ1: the reach-scale model against the LSTM (Figure 5, row M0 - LSTM; Table S6)
- **Gauged:** +0.020 [+0.007, +0.035], Holm p 0.053.
- **Random hold-out:** +0.015 [-0.010, +0.037]. The mean and median disagree (median +0.035, 66% of gauges better, Holm p 0.010), so report both.
- **Blocked:** +0.066 [+0.026, +0.105], Holm p 0.072, 69% of gauges better, n = 29 in three basins.
- **By lead:** in the random hold-out the gain shrinks from +0.031 at 24 h to -0.003 at 120 h; in the blocked protocol it is steady (+0.06 to +0.08).
- **Answer:** a small gain that grows with the strictness of the hold-out. It is a system difference, not only the river network (Section 4.4).

### 5.3 RQ2: message passing or routing (Figure 5, rows M0 - M2 and M1 - M4; area strata)
- **Message passing (M0 - M2):**
  - gauged +0.005, equivalent within 0.01;
  - random +0.023 [-0.002, +0.054], median +0.001;
  - blocked -0.013 [-0.038, +0.010].
- **Answer for message passing:** no consistent effect.
- **Routing (M1 - M4):** waiting for M4.
- **Routing sanity check (one sentence):** learned travel times scale with river length (Spearman 0.88). The implied celerity is about 1.1 m/s, a physically plausible value.
- **High flows:** all models underestimate peaks by about 40%; the reach-scale models time short-lead peaks more closely (median 3 h against 6 h for the LSTM and NWM). Mention this only if routing (M1 - M4) shows a difference.

### 5.4 RQ3: upstream observations (Figure 6, Table S6)
- **Overall:** M0 - M1 is within the 0.02 margin in the gauged and random protocols. In the blocked protocol insertion cannot act (0 of 29 gauges have a visible upstream gauge), so the +0.012 there measures run-to-run differences, not insertion.
- **Where insertion can act** (random hold-out, the 40 nested gauges):
  - a gain at 6 h, about +0.05 (Figure 6a);
  - no gain at days 1-5: -0.003 [-0.018, +0.015].
- **Area-matched:** 12 nested-headwater pairs within a factor 1.41; difference of gains about 0.
- **Area ratio:** the gain at days 1-2 falls as the upstream gauge's share of the area rises (Spearman -0.43, p 0.005, random hold-out). It shows no relation in the gauged protocol (+0.10). Report it as observed, without a mechanism.
- **Within-gauge test:** not identifiable, because upstream gauges are visible at almost every issue. One sentence, in the text or the supplement.
- **Answer:** upstream observations routed downstream help only in the first hours.

### 5.5 Behavior at every reach (Figure 7)
- **Framing:** agreement with the NWM retrospective by drainage area, for M0 and the LSTM only. State the caveat in point 3 in the first sentence.
- **Content:** agreement is highest at 500-5,000 km2 and lowest in the smallest (< 5 km2) and largest (> 5,000 km2) classes.
- **Placement:** keep it short or move it to the supplement if space is tight.

### Supplement
- **Figure S3 (the example event):** the M0 - M1 median gauge, by fixed rule.
- **Table S6:** every contrast.
- **Section 5.4:** the within-gauge test and the area bins.

## 4. Presenting the current figures better (no new figures)

- **Figure 4 (lead skill):** keep. In the text, point to it once for the overall ranking; the lines overlap, so no contrast should be read off it.
- **Figure 5 (contrasts):** the central figure.
  - Order the rows by research question: M0 - LSTM (RQ1); M0 - M2 and M1 - M4 (RQ2); M0 - M1 (RQ3).
  - Move M0 - NWM to Table 3 and Table S6: its off-scale values compress the scale for the others, and it is not an RQ contrast.
  - Label the row groups RQ1-RQ3 on the left.

  These are layout edits to the existing script; the data stay the same.
- **Figure 6 (upstream observations):** keep both panels. The caption says panel (a) shows means with bootstrap 5-95% bands; panel (b) carries the rank correlations.
- **Figure 7 (every reach):** keep. The caption states that the reference is the NWM retrospective, which the reach-scale models were pretrained on.
- **Figure 8:** becomes Figure S3.
- **Table 3:** keep, with KGE beside NSE, so the blocked reversal is visible.
- **Table S6:** keep.

## 5. Writing rules for Section 5

- **Sentence pattern:** answer, number with interval and n, test, lead dependence.
- **Equivalent:** call a contrast equivalent only when the Holm-corrected TOST p is below 0.05 at that margin. A contrast whose interval includes zero but is not shown equivalent is "undetermined".
- **Seeds:** every interval is across gauges, not training runs. Say so once at the start of the section (Section 4.6 already states it).
- **No mechanisms:** those go in Section 6, the Discussion.

## 6. What must happen before writing

1. **M4 evaluation:** after the blocked runs finish, about 1 h on 4 GPUs. Then rerun `scripts/section5/s5_facts.py`, the figure scripts and `tables.py`; no code changes are needed.
2. **Seed 2 on the blocked basins:** recommended, about 31 h. The blocked protocol carries the largest differences (M0 - LSTM +0.066, M0 - M2 -0.013) and has only 29 gauges in three basins.
3. **Section 1 RQ1 wording:** fix it, and set the figure file names.

## 7. Revisions adopted on 22 Nov 2026 (supersede the conclusions above where they differ)

**Conclusions tightened to the numbers**
- **RQ1:** none of the three M0 - LSTM contrasts survives Holm correction (p = 0.053, n.s., 0.072), the random-hold-out interval includes zero and the intervals overlap, so the status is *undetermined* in all three protocols (`status` column of `s5_3_contrasts.csv`). The blocked result (+0.066, n = 29, three basins) is suggestive only, and the LSTM has the higher blocked KGE. Write that, not a trend with the strictness of the hold-out.
- **Area-ratio result (5d):** exploratory, supplement only, no main-text sentence. Robustness is in 5f of the facts sheet (leave-one-out, without the three largest gains, bootstrap); it runs opposite to the expectation in Section 2.3 and changes sign between protocols.
- **Peak underestimation:** reported regardless of routing, as a statement in 5.1 or 5.5 (Section 1 opens with flood warning). Numbers: facts sheet 6c.

**Additions**
1. **Noise floor** (facts sheet 5e): in the blocked protocol insertion cannot act, so the M0 - M1 difference (+0.012; B1 +0.004, B2 +0.019, B3 +0.018) is a one-pair estimate of training-to-training differences. Compare every "within 0.02" claim with it.
2. **Seeds first, cheap before expensive** (facts sheet 10): LSTM seeds 2-5 (about 1.5 h plus the CPU benchmark), a second stage-B seed for M0, M1, M2 in the gauged protocol (about 9 h), then blocked seed 2 (about 31 h).
3. **Intervals and margin:** 90% (5-95%) gauge-bootstrap intervals in Figure 5, stated in the caption as the usual display for equivalence tests; justify +-0.02 in one sentence from the seed spread of item 2; primary quantity = mean paired difference, medians alongside.
4. **Every reach:** within-family contrasts (M0 - M1, M0 - M2 by area class) are fair because the same reference was used for pretraining; only the LSTM comparison is biased. This needs reach-level runs of M1, M2, M4 (existing models, about 2.2 h of evaluation); until then Figure 7 shows M0 and the LSTM with the LSTM line flagged.
5. **Gauge-based skill by area stratum** (facts sheet 5g): the only observation-based evidence below 200 km2 (smallest gauge 13.9 km2; 4 gauges under 50 km2, 27 under 200 km2).

**Sequencing:** write 5.1, 5.2, 5.4 and 5.5 now; 5.3 waits for the M4 evaluations (running). Settle the framing first: the honest summary may be that the river network gives no clear benefit beyond the first hours, and the abstract and Discussion would be written around that.
