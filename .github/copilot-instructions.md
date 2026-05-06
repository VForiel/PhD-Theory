# Copilot Instructions for PhD-Theory

Before making any change in this repository, always read [README.md](../README.md) first.

Repository work should follow this structure:

- Put reusable analysis logic in `src/analysis/<analysis_name>/...` Python files.
- Use notebooks in `src/analysis/<analysis_name>/...` as scientific companions that explain the problem, show the theory, run the code, and discuss the results.
- Use the web layer to present those analyses in a clearer and more accessible form without duplicating the core scientific implementation.

If a task touches analysis code, prefer changing the Python module first and then update the notebook narrative or web presentation as needed.