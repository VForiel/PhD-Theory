# 🌐 Thesis Web Interface Guidelines

This document outlines the purpose, structure, and development guidelines for the interactive thesis website.

## 🎯 Goal
The primary goal of this website is to present the thesis in its entirety. Each page focuses on a specific aspect of the research, typically accompanied by an **interactive demonstration** to foster intuitive understanding of the concepts.

## 🏗️ Page Structure
Pages usually follow this standard layout pattern:

1.  **Overview**: 
    *   Introduction to the topic.
    *   General theoretical explanation.
    *   Key concepts required to understand the subsequent simulation.
    *   Key interests of the research.
2.  **Simulation**:
    *   Interactive area managed by the **Context Widget** (see below).
    *   Specific simulation parameters (overriding the context).
    *   Visual results (plots, maps, metrics).
3.  **Discussion**:
    *   Analysis of the results.
    *   Scientific interpretation.
4.  **Footer Details** (in Expander/Dropdowns):
    *   🔎 **Theoretical Details**: Deeper dive into equations or physics.
    *   🔧 **Technical Details**: Implementation specifics, algorithm choices.
    *   🤝 **Acknowledgements**: Credits to collaborators or tools.
    *   📚 **References**: Bibliographic citations.

## ⚙️ PHISE integration
The thesis relies heavily on the **PHISE** library, with the `Context` object being its core component.

*   **Context Widget**: When a page involves simulations based on the global physical context (telescope, atmosphere, optical chain), it **MUST** include the `context_widget` script.
*   **Configuration Flow**:
    1.  **Default Context**: The page defines a "default context" relevant to the specific demo.
    2.  **Widget**: The `context_widget` allows the user to tweak global parameters (overriding defaults).
    3.  **Specific Parameters**: Below the widget, the page exposes simulation-specific parameters (e.g., scanning range, specific modulation settings) that further override or complement the context.
    4.  **Execution**: The simulation runs using this aggregated configuration.

## ✨ Highlighting Contributions
Scientific and technical contributions specific to this thesis are highlighted to distinguish them from general state-of-the-art knowledge.

*   **Format**: Use a Streamlit info/success box or a distinct Markdown block.
*   **Icon**: The block contains the ✨ emoji to be instantly recognizable.
