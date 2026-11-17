# Paraguay fresh test (tier TF) — single locked evaluation, 2026-10-30

Hospital de Clínicas, Asunción, Paraguay (Data in Brief 2021, PMC8257963; dataset Zenodo doi:10.5281/zenodo.4647952, CC BY 4.0); Zeiss Visucam 500.
Registered as reserved before download; images extracted blind (neutral content-hash names, labels sealed); identity-reviewed against 136,074 training,
validation and exposed-test images (no exclusion). `lock.json` (30 bound artifacts, `labels_read: false`) preceded the one evaluation `report.json`
(curated copy: per-image identifiers replaced by counts). Pre-registration: MODEL_IMPROVEMENT_PLAN_v6 §2–3 and Addenda 6.1–6.3 (private notes).

Primary population: 705 images (740 unique; 35 practically identical copies with conflicting labels excluded, 17 byte-identical copies merged),
519 referable (ICDR ≥ 2; folder categories mapped as declared), 647 clusters (same-retina copies merged).
Spectrum caveats: the authors excluded eyes with other retinal disease and the sample is severity-enriched (74% referable; 461 severe NPDR/PDR),
so specificity rests on 186 non-referable images and the set is easier than a screening population.
E_768 member logits were computed on the owner's Mac (MPS) and passed the lock's CPU spot check (max |Δ| 1.7e-3 and 1.0e-3, tolerance 5e-3);
the legacy model's logits were computed on the laptop CPU.
