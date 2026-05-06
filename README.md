# PhD-Theory

This repository is organized around three complementary layers:

1. Python packages in `src/analysis/<analysis_name>/` contain the reusable analysis code.
2. Jupyter notebooks in `src/analysis/<analysis_name>/` act as scientific companions: they introduce the problem, explain the theory, run the package code, and discuss the results.
3. The web layer reuses the notebooks to present the work in a clearer, more accessible, and more pedagogical way.

The intent is to keep the scientific logic in importable Python modules, while the notebooks remain the narrative and validation surface. This makes the analyses easier to test, reuse, and expose later in the web interface without duplicating the core implementation.a

## Repository Structure

- `src/analysis/`: importable analysis packages.
- `src/analysis/<analysis_name>/`: analysis-specific Python modules, helpers, and notebook companions.
- `web/`: the public-facing layer that presents the analyses in a more accessible order.
- `docs/`: reports, slides and other non-code deliverables.