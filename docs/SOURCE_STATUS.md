# Manuscript source status

The supplied project archive contains `Stellar_evolution_FINAL_REVISED_v2.tex`, but that file is **not** the source of the supplied final 20-page paper. It corresponds to a different 25-page manuscript version and contains superseded numerical/methodological statements, including an older bootstrap configuration and different mixture-model framing.

The final 20-page PDF in `paper/final_research_paper.pdf` was therefore corrected directly at the PDF level because the exact matching source was not present in the supplied archive. The correction changed only reproducibility wording and seed documentation; it did not alter the study's numerical tables, figures, or scientific claims.

The superseded `.tex` and older manuscript PDF are retained under `archive/legacy_manuscript/` for provenance and are explicitly not part of the active build.

The cleaned `retrieve_matched_gaia_parameters.py` and the ADQL query files were added to make the data acquisition pipeline explicit. They reconstruct the documented final data operation; they are not presented as the missing historical notebook source.
