# SUSTech-SYSU fresh test (tier TF) — single locked evaluation, 2026-10-30

Gaoyao People's Hospital and Zhongshan Ophthalmic Center, Guangdong, China (Lin L et al., Sci Data 2020, doi:10.1038/s41597-020-00755-0; dataset
figshare doi:10.6084/m9.figshare.12570770, CC BY 4.0); Topcon TRC-50DX. Registered as reserved before download; images extracted blind (labels and
lesion annotations sealed); identity-reviewed against 136,074 training, validation and exposed-test images (no exclusion). `lock.json` (31 bound
artifacts, `labels_read: false`) preceded the one evaluation `report.json` (curated: identifiers replaced by counts) and the pre-registered secondary
endpoint `secondary_report.json` (site-calibration rule R2, selected on exposed sites before this lock; script SHA-256 recorded in the bound plan).

Primary population: 1,145 images (1,219; 68 laser-treated "category 5" images reported separately; 6 practically identical copies with conflicting
labels excluded), 490 referable (ICDR ≥ 2 in the release's ICDR column), 1,118 clusters (same-retina copies merged).
Spectrum caveat: case-control design — the 631 grade-0 images come from non-diabetic volunteers, so specificity is measured on easy negatives and is
optimistic for a diabetic screening population. All logits computed on the laptop CPU.
