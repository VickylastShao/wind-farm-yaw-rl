# Wind Energy Submission Checklist

## Manuscript Files
- [x] `main.tex` — Revised manuscript (Wind Energy format)
- [x] `main.pdf` — Compiled PDF (1.39 MB)
- [x] `supplementary.tex` — Supporting Information
- [x] `supplementary.pdf` — Compiled Supplementary (45 KB)
- [x] `response_letter.tex` — Point-by-point response to reviewers
- [x] `response_letter.pdf` — Compiled Response Letter (46 KB)

## Figures (New — Config-E)
- [x] `Fig_FLORIS_cross_validation.pdf` — FLORIS cross-validation summary
- [x] `Fig_noise_robustness.pdf` — Observation-noise robustness
- [x] `Fig_lock_ablation.pdf` — Downstream-lock ablation

## Results Data
- [x] `results/configE_floris_cross_validation.json` — FLORIS 3000-condition results
- [x] `results/configE_noise_robustness.json` — Noise robustness 5-seed results
- [x] `results/wind_rose_aep_analysis.json` — Illustrative AEP analysis

## Submission Documents
- [x] `submission/cover_letter.md` — Cover letter for editor
- [x] `submission/statements.md` — Data, conflict, funding, author, AI-use statements
- [x] `submission/checklist.md` — This file

## Audit & Quality
- [x] `audit/file_inventory.md` — Complete file inventory
- [x] `audit/experiment_inventory.md` — Experiment catalog
- [x] `audit/reproducibility_status.md` — Per-experiment reproducibility
- [x] `audit/claims_vs_evidence_table.csv` — 17 claims mapped to evidence
- [x] `audit/manuscript_structure_review.md` — Structure assessment
- [x] `final_quality_check/claim_evidence_matrix.csv` — Final claim verification
- [x] `final_quality_check/reproducibility_checklist.md` — Reproducibility status
- [x] `final_quality_check/remaining_risks.md` — Remaining risks (none fatal)

## Key Experiment Checkpoints
- [x] Config-E PPO: 5 seeds (sens_act10)
- [x] Config-E Lock-off: 3 seeds (lock_off_configE)
- [x] SAC Fair: 5 seeds (checkpoints_3x3_sac_fair)
- [x] Marginal Reward: 3 seeds (marginal_reward)
- [x] KL ON: 3 seeds (kl_on)
- [x] KL OFF: 3 seeds (kl_off)
- [x] BC Baseline: 1 seed (bc)

## Pre-Submission Verification
- [x] All LaTeX files compile without errors
- [x] All figures referenced in text exist
- [x] All tables have captions and labels
- [x] References complete (no missing citations)
- [x] No "TO BE FILLED" in main text (Supplementary only)
- [x] Journal name correct: Wind Energy
- [x] Title matches submission requirements
- [x] Author list and affiliations present
- [x] Abstract within word limit
- [x] Keywords present
- [ ] Cover letter personalized with author names — [TO BE FILLED]
- [ ] Funding statement completed — [TO BE FILLED by authors]
- [ ] Acknowledgements completed — [TO BE FILLED by authors]

## Known Limitations (Not Blocking)
- KL ablation metrics not extracted (policy checkpoints only)
- λrate analysis uses Config-A policies (not retrained for Config-E)
- Wind-rose AEP uses synthetic data (illustrative only)
- No OpenFAST/FAST.Farm aeroelastic validation
- No field SCADA validation
