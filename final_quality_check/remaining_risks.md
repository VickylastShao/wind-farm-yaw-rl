# Remaining Risks Before Submission

## Moderate Risks

1. **λrate Config-E mismatch**: Dynamic wind λrate analysis uses Config-A (J=1, no positions) policies. Config-E λrate retraining would strengthen this claim. *Can be fixed: retrain with Config-E + rate penalties (~3.5h GPU).*

2. **Gated DRL is evaluation-only**: The gating strategies are applied post-hoc to the trained Config-E policy. A policy explicitly trained with gated operation may perform differently. *Acceptable for current submission; note as limitation.*

3. **No wind-rose AEP**: The paper lacks wind-rose-weighted AEP analysis requested by Wind Energy reviewers. *Can use ERA5 data or synthetic wind rose as illustrative analysis.*

4. **No secondary load proxy**: Yaw travel is the only actuator cost metric. A yaw-misalignment aerodynamic load proxy would strengthen the actuator-aware contribution. *Can be added as simplified proxy with clear limitations.*

## Minor Risks

5. **Figure reproducibility**: Not all figures have automated generation scripts. make_figures.py covers the 3 new figures; older figures rely on existing scripts. *Acceptable; most figures have generation scripts.*

6. **Single-seed BC evaluation**: BC baseline evaluated on 1 seed only. Multi-seed BC training would add confidence. *Low impact; BC result is strongly negative regardless.*

7. **KL ablation results not analyzed**: KL ON/OFF checkpoints exist but comparison metrics not extracted. *Low impact; supplementary material only.*

## Fatal Risks: NONE
