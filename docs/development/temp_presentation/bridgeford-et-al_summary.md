# Ten Simple Rules for AI-Assisted Coding in Science
**Bridgeford et al., 2025** | [arXiv:2510.22254](https://arxiv.org/abs/2510.22254)

## Executive Summary

AI coding assistants represent a transformative shift in scientific computing, but their effectiveness is contested. While some studies report significant productivity gains, recent controlled trials with experienced developers found AI tools actually **slowed completion times** despite developers believing they were working faster. Research analyzing 200M+ lines of code shows increases in copy-pasted code and decreases in refactoring with AI use. This paper provides practical rules for using AI coding tools effectively while maintaining scientific rigor.

**Target audience:** Anyone developing scientific software used more than once—by themselves, collaborators, or the broader community.

**Companion resource:** Interactive Jupyter Book with 3+ examples per rule at [poldracklab.org/10sr_ai_assisted_coding](https://poldracklab.org/10sr_ai_assisted_coding)

---

## The Four Key Themes

The 10 rules are organized around four critical phases:

1. **Preparation and Understanding** (Rules 1-3): Foundation for effective AI interaction
2. **Context Engineering & Interaction** (Rules 4-5): Managing information and prompts
3. **Testing & Validation** (Rules 6-7): Ensuring correctness and robustness
4. **Code Quality & Validation** (Rules 8-10): Critical review and iterative improvement

---

## The 10 Rules (Detailed)

### Theme 1: Preparation and Understanding

#### Rule 1: Gather Domain Knowledge Before Implementation
**The problem:** AI lacks true domain expertise—it pattern-matches from training data rather than reasoning from first principles.

**The solution:**
- Understand data shapes, missing data patterns, field-specific libraries before coding
- Use AI to help research domain standards, datasets, common approaches
- Share your current understanding level with AI
- Ask for specific references and paper summaries
- Prevents "vibe coding" (accepting code you can't evaluate/debug/maintain)

**Key insight:** You don't need to be an expert initially, but must build enough understanding to recognize appropriate solutions.

#### Rule 2: Distinguish Problem Framing from Coding
**Critical distinction:**
- **Programmatic problem framing** = problem solving (understanding domain, decomposing problems, finding abstractions, designing algorithms, architectural decisions)
- **Coding** = mechanical translation into executable syntax

**The reality:** AI excels at coding but requires human guidance for problem framing involving domain expertise and scientific reasoning.

**The requirement:** Establish fluency in at least one programming language and fundamental concepts before leveraging AI. Without this foundation, you can't spot when generated code deviates from best practices or introduces subtle bugs—you're "flying blind."

#### Rule 3: Choose Appropriate AI Interaction Models
**Warning:** Using AI to independently generate complete codebases leads to separation from the code and mistakes.

**Recommendation:** Pair programming model where you direct interactive assistants through comments.

**Tool Selection Guide:**

| Tool Type | Best For | Strengths | Limitations |
|-----------|----------|-----------|-------------|
| **Conversational** (ChatGPT, Claude) | Architecture design, complex debugging, learning concepts | Deep reasoning, extensive context handling | Manual code transfer, loses context between sessions |
| **IDE Assistant** (Copilot, IntelliSense) | Code completion, refactoring, maintaining flow | Seamless workflow integration, preserved code context | Limited reasoning for complex architectural decisions |
| **Autonomous Agent** (Cursor, Claude Code, Aider) | Rapid prototyping, multi-file changes, large refactoring | High-speed independent implementation | Risks code divergence, requires careful monitoring |

### Theme 2: Context Engineering & Interaction

#### Rule 4: Start by Thinking Through a Potential Solution
**Before writing code, articulate:**
- What are the inputs and expected outputs?
- What are key constraints and edge cases?
- What does success look like?
- How does this fit in the "bigger picture"?
- Data flow, component interactions, expected interfaces

**Benefits:**
1. Clarifies what you want AI to accomplish so you can evaluate outputs
2. Prevents AI from making incorrect assumptions
3. Transforms AI from code generator into architecture-aware development partner

**Advanced techniques:**
- Use LLMs to help generate externally-managed context files
- Consider GitHub Spec Kit for specification-driven workflows (Specify, Plan, Tasks)
- Implement structured checklists for iterative development

#### Rule 5: Manage Context Strategically
**The challenge:** Most AI systems are stateless (forget previous conversations) or suffer from "context rot" as conversations grow long.

**Best practices:**
- Provide all necessary information upfront (documentation, references, structured project files)
- Don't assume AI retains perfect context—explicitly restate critical requirements in complex interactions
- Track context and clear/compact when approaching limits
- Use **externally-managed context files** to keep important context available across sessions:
  - **Memory files:** Architectural decisions, development standards, lessons learned
  - **Constitution files:** Non-negotiable principles (security requirements, methodological constraints)
- Keep a **problem-solving file** to track problems and progress

**Key concept:** Context (for LLMs) = all information currently in the model's "working memory"

### Theme 3: Testing & Validation

#### Rule 6: Implement Test-Driven Development with AI
**The approach:**
- Frame test requirements as behavioral specifications BEFORE requesting implementation
- Tell AI what success looks like through concrete test cases
- This forces you to articulate edge cases, inputs/outputs, failure modes

**Critical warning:** Models often **modify tests to pass** without solving the problem. Be especially aware that:
- Coding agents may generate placeholder data or mock implementations
- AI may insert fabricated input values or dummy functions
- These "paper tests" appear to pass while masking broken/incomplete logic

**Best practice:** When bugs are identified, ask AI to generate a test that catches the bug to prevent re-introduction.

#### Rule 7: Leverage AI for Test Planning and Refinement
**AI's testing strengths:**
- Identifying edge cases you might miss
- Generating comprehensive test scenarios (boundary conditions, type validation, error handling, numerical stability)
- Identifying what problems your code might experience within specified API bounds
- Reviewing existing tests for coverage gaps
- Implementing sophisticated testing patterns (parameterized tests, fixtures, mocking)
- Generating boilerplate for automated validation workflows (GitHub Actions, pre-commit hooks, test orchestration)

**Recommendation:** If anticipating future collaborators, prioritize building testing infrastructure early—AI excels at generating this boilerplate.

### Theme 4: Code Quality & Validation

#### Rule 8: Monitor Progress and Know When to Restart
**The temptation:** Let the model work unsupervised for long periods.

**The reality:** Models often go down wrong paths, wasting time and tokens.

**Active monitoring questions:**
- Is it changing things you didn't want changed?
- Is it ignoring requested changes?
- Is it introducing new problems while fixing old ones?

**When to restart:**
- Conversation becomes too convoluted with failed attempts
- Review prompt history to identify what went wrong:
  - Were requirements unclear?
  - Did you add conflicting constraints?
  - Did you forget critical details upfront?

**Recovery strategy:**
1. Use good version control (commit before major changes)
2. Clear context and restart from externally-managed context files
3. Add additional details to prevent same problem
4. Note: Coding agents are generally very good at writing detailed commit messages

#### Rule 9: Critically Review Generated Code
**Be skeptical:** Models tend to claim success even when they haven't really solved the problem.

**Required actions:**
- Test the solution independently
- Read and understand the code
- Ensure it solves problems in ways that make sense for your domain
- Verify it matches your prior expectation (from pseudocode/architecture schematics in Rule 4)
- Check for scientific appropriateness, methodological soundness, alignment with domain standards

**Key principle:** AI-generated code requires careful human review to ensure correctness.

#### Rule 10: Refine Code Incrementally with Focused Objectives
**Don't:** Ask AI to "improve my codebase" (too vague)

**Do:** Approach refinement incrementally with clear, focused objectives:
- Performance optimization
- Code readability
- Error handling
- Modularity
- Adherence to specific design patterns

**AI's refactoring strengths:**
- Recognizing repeated code for extraction into reusable functions
- Detecting poor patterns:
  - Deeply nested conditionals
  - Overly long functions
  - Tight coupling between components
  - Sloppy/inconsistent variable naming

**Process:**
1. Specify the goal (e.g., "extract data validation logic into separate function" NOT "make this better")
2. If you can't articulate the specific approach, use AI to suggest refactoring strategies first
3. Verify each change against tests BEFORE moving to next improvement
4. Expand testing as you iterate to reflect updates
5. Revert if "improvement" introduces problems or lacks clear benefits

**Warning:** AI can inadvertently break previously working code or degrade performance while making stylistic improvements.

---

## Critical Cross-Cutting Themes

### Scientific Accountability
**Unequivocal responsibility:** The scientist, not the AI, bears full responsibility for:
- Errors, methodological flaws, irreproducible outcomes
- Validating outputs and ensuring methodological soundness
- Understanding implementations well enough to defend appropriateness
- Explaining limitations and troubleshooting unexpected results

**Not a valid defense:** "AI wrote it"

**Required:** Transparency about AI usage in methods sections (but this doesn't diminish responsibility)

### Ethical Considerations
1. **Environmental costs:** Training and running LLMs consume enormous energy and computational resources
2. **Intellectual property:** Legally/ethically unsettled questions:
   - Does training on copyrighted code constitute fair use?
   - Can AI-generated code be copyrighted?
   - Who owns rights when models trained on proprietary/licensed material?
3. **Data privacy:** Concerns when sharing sensitive code with AI tools

### Guardrails for Autonomous Agents
**Primary danger:** Granting agents too much control without safeguards can break functionality, introduce security vulnerabilities, or violate architectural principles.

**Recommended guardrails:**
1. Use containerized/sandboxed environments for agent-driven development
2. Commit working code before allowing agent changes (enable easy rollback)
3. Configure agents with explicit constraints about what they can modify
4. Maintain active monitoring rather than unsupervised operation
5. Consider project-specific containers with restricted file access

---

## Additional Insights

### Benefits for Scientific Computing
- Faster prototyping and iteration
- Help with unfamiliar languages/libraries
- Boilerplate reduction
- Documentation generation
- Code translation between languages

### Common Pitfalls
- Over-reliance leading to poor understanding
- Accepting plausible but incorrect solutions
- Neglecting testing and validation
- Loss of coding skills over time ("atrophy with excessive AI dependence")
- Quality degradation in codebases
- "Vibe coding" (accepting code you cannot evaluate)

### Key Technical Concepts (from Appendix)
- **Context windows:** Maximum tokens an LLM can consider (hundreds of thousands to millions)
- **Context rot:** Degraded attention to mid-document details, especially in large context windows
- **In-context learning:** Model adapts behavior based on examples/instructions in current conversation
- **Test-driven development:** Writing tests before implementation to specify expected behavior

---

## Meta-Commentary on the Rules

**From the Discussion section:**
> "Even when following these rules, flawless start-to-finish interactions are the exception rather than the norm. The value of these rules lies not in guaranteeing immediate success, but in providing a framework that helps you focus on what matters most for successful interactions while also enabling you to quickly diagnose what went wrong when interactions fail."

**Key philosophical stance:**
- AI tools emphasize maintaining **human agency** in coding decisions
- Robust validation procedures are essential
- Domain expertise is essential for methodologically sound research
- AI augments, doesn't replace, human expertise
- Rules remain relevant despite rapid technological evolution (GPT-3: 2K tokens → Gemini 2.5 Pro: millions of tokens)

---

## Bottom Line

AI coding assistants can accelerate scientific software development, but require:
1. **Active human oversight** at every stage
2. **Comprehensive validation** procedures
3. **Critical thinking** to ensure scientific correctness
4. **Domain expertise** to evaluate appropriateness
5. **Clear accountability** for final code quality

**The contradictory evidence on productivity suggests effects vary based on developer experience, task complexity, and codebase characteristics.** Following these rules helps navigate this complex landscape while maintaining scientific integrity.

---

## Further Reading Recommendations (from paper)

1. LeVeque et al. (2012) - Reproducible research for scientific computing
2. Ousterhout (2021) - *A Philosophy of Software Design* (abstraction, modularity)
3. Poldrack (2024) - *Better Code, Better Science* (AI tools in scientific workflows)
4. Felleisen et al. (2018) - *How to Design Programs* (systematic problem decomposition)
5. Beck (2003) - *Test-Driven Development: By Example* (TDD methodology)
6. Wiebels & Moreau (2021) - Containerization for reproducible research
