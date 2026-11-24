
---

# PART D. Open items (must be resolved or stated in the text)

1. **M4 and routing (RQ2).** M4 is evaluated for the random hold-out and the blocked basins (done 2026-11-22); there is no gauged-protocol M4, so M1 - M4 exists in those two protocols only.
2. **Seeds.** One seed per model and protocol. Planned in order of cost: LSTM seeds 2-5 (about 1.5 h plus the CPU benchmark); a second stage-B seed for gauged M0, M1, M2 (about 9 h); then blocked seed 2 (about 31 h) and random hold-out seed 2 (about 50 h). The +-0.02 equivalence margin should be justified from the seed spread once these exist; until then say that it is a pre-chosen margin and give the noise floor (PART B 5e).
3. **Every-reach contrasts within the model family** (M0 - M1, M0 - M2 by area class) need reach-level runs of M1, M2, M4 (existing models, about 2.2 h of evaluation); until then Figure 8 and Figures S5-S6 show M0, the LSTM and the NWM forecast only.
4. **Figures** are final in layout (journal style); Figure 5 already groups the rows by research question and leaves M0 - NWM to Table 3 and Table S6.
5. **Paper housekeeping:** (a) Introduction, RQ1 says "a lumped LSTM with matched inputs and outputs", which contradicts Section 4.4: reword as in A3. (b) The paper's Figure 3 is the model schematic (file `fig4_model`); rename it `fig3_model` so the results figures are 4-10. (c) Section 4.5 contains a VERIFY comment on M4 coverage: adjust the sentence and Table 2 to the runs that exist: M4 in the random hold-out and blocked protocols only (inventory in PART B, section 1).
6. **Numbers not in this file** (e.g. any comparison not in PART B): say they are not available; do not compute or estimate them.
