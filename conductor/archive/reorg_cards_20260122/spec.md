# Specification: Reorganization of Cards and Content Refinement

## Context
The current webpage layout and content organization need refinement to better utilize screen space and present information more logically. The "Project Overview" card is redundant or misplaced, and several technical cards are fragmented. We also need to leverage the rich data in `project_notes.md` to enhance the "Technologies Used" section.

## Goal
1.  **Widen Layout:** Increase the main container width to better accommodate content.
2.  **Consolidate & Prune:** Remove the "Project Overview" card after redistributing its unique content. Merge four specific technical cards into one cohesive "Project Details" card.
3.  **Enrich Content:** Expand the "Technologies Used" card with detailed, categorized information from project notes.

## Requirements
-   **CSS Change:**
    -   Update `.container` in `styles.css` to have a `max-width` of `72rem`.
-   **Content Redistribution (Project Overview):**
    -   Analyze the text in the current "Project Overview" card.
    -   Move any non-redundant information to "Motivation", "Objective", "Approach", or the new "Project Details" card.
    -   Delete the "Project Overview" card from `index.html`.
-   **Card Consolidation (Project Details):**
    -   Create a new `grid-full` card named "Project Details".
    -   Merge the content of "Evolutionary Core", "Simulation Environment", "Heterogeneous Architecture", and "Real-Time Analytics" into this single card.
    -   Use the old card titles as `<h3>` subheadings within this new card.
    -   Place this new card in the position previously occupied by "Project Overview" (or effectively after the Motivation/Objective/Approach row).
-   **Content Expansion (Technologies Used):**
    -   Read `conductor/project_notes.md` to identify all mentioned technologies.
    -   Update the "Technologies Used" card to use a **categorized list** format with subheadings (e.g., "Core Languages", "Simulation & Graphics", "AI & Compute", "Data Analysis").

## Constraints
-   **No Comments:** strictly prohibited in HTML/CSS.
-   **Style Consistency:** Use existing `card` and `grid-full` classes.
-   **Verification:** Ensure no information is lost during the deletion of "Project Overview".
