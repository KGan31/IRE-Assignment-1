# MATH AND MARKDOWN RENDERING CONSTRAINTS
You are operating within a Antigravity side panel where the Markdown parser aggressively intercepts text before the LaTeX renderer. To prevent rendering failures, you MUST obey these formatting rules:

1. **Inline vs. Block Math:** 
   - ALWAYS use a single `$` for inline math (e.g., `for $Q$ queries`). 
   - NEVER use double `$$` inline within a paragraph. Reserve `$$` strictly for standalone block equations on their own lines.

2. **No Underscores in LaTeX:** 
   - NEVER put code variables with underscores inside LaTeX blocks (e.g., avoid `$\text{top_k}$`). The Markdown parser will treat `_` as italics and permanently corrupt the equation.
   - **Fix:** Keep programming variables entirely outside of math blocks using standard backticks (e.g., `$k = \min($` `top_k` `, ` `num_docs` `$)$`), OR explicitly escape them (`$\mathrm{top\_k}$`).

3. **Hyphen and Delimiter Separation:** 
   - NEVER attach hyphens or text directly to the outside of a `$` delimiter (e.g., NEVER write `top-$K$`). This breaks delimiter pairing.
   - **Fix:** Move the prefix into the math block itself (e.g., `$\text{top-}K$`) or use standard Markdown formatting (e.g., `top-`*`K`*).

4. **Parentheses:** 
   - When enclosing mathematical coordinates or shapes, keep the parentheses inside the math block (e.g., use `shape $(Q, D)$`, NEVER `shape $(Q, D)$)`).
   
## For Running Tests: 
Command: python -m pytest src/tests/
