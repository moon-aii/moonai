# Specification: Analyze PDF Documents and Update Webpage

## Context
The project has several PDF documents in `documents/` containing detailed information about the MoonAI simulation. The current webpage is minimal. We need to extract rich content from these documents and populate the webpage with new cards to make it a better "digital poster".

## Goal
1.  Systematically analyze all PDF files in `documents/`.
2.  Create a summary file `conductor/project_notes.md` with key insights.
3.  Update `index.html` with new cards displaying this information (e.g., specific algorithms used, experiment results, design details).

## Requirements
- **Source Files:**
    - `documents/ProjectProposal.pdf`
    - `documents/ProjectSpecification.pdf`
    - `documents/AnalysisReport.pdf`
    - `documents/HighLevelDesignReport.pdf`
    - `documents/Poster.pdf`
- **Output:**
    - `conductor/project_notes.md`: Intermediate synthesis.
    - `index.html`: Updated with new cards.
- **Constraints:**
    - Do NOT use comments in HTML.
    - Use existing CSS classes (`card`, `grid`, etc.).
    - Do NOT modify existing "Team", "Repositories", or "Documents" cards (unless fixing links, but the goal is *adding* new ones).
