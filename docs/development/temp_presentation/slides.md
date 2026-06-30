# AI Coding Assistants in Computational Research

### A Lab Discussion on Practice, Principles, and the Path Forward

**Robin Gutzen**

---

<!-- .slide: data-background-color="#2C3E50" -->

# AI Assistant's Future in Research Software Engineering

**Focus:** Very real immediate developments in our specific context

**Goal:** Arrive at actionable take-aways for our workflow

---

## Quick Poll

**How many of you are currently using AI coding assistants?**

- Autocomplete tools (Copilot, Tabnine)?
- Chat interfaces (ChatGPT, Claude)?
- Autonomous agents (Cursor, Claude Code)?
- Not using any yet?

---

<!-- .slide: data-background-color="#34495E" -->

# Part 1: The Journey

---

## My Evolution with AI Coding Tools

**The arc:**
1. ✓ Copilot autocomplete → chat for review
2. ✓ Pair programming, reviewing every step
3. ✗ Over-trusted ambitious changes
4. ✗ More debugging time than writing from scratch
5. 🔄 Tools got smarter, I got strategic
6. ⭐ Current: Claude Code with comprehensive context

---

## The Turning Point

**Strategic environment setup:**
- Compressed documentation (PyTorch Lightning, Snakemake, FFCV)
- Custom style guides for research software
- Architecture overviews and component interactions
- Documentation validation loops

**Result:** Assistant catches design mismatches I missed, implements tests, validates correctness

---

## The Model Integration Challenge

**Task:** Translate external models → DynVision framework

[**FIGURE:** Architecture diagram]

**Complexity:**
- Abstract into standardized operations
- Make implicit design choices explicit
- Ensure exact numerical equivalence

---

## Code Example

[**SPACE:** Side-by-side comparison]
- Original implementation vs. DynVision translation
- Key architectural differences

**Traditional:** Manual reimplementation, time-consuming validation

**AI-assisted:** Still manual, but AI helps with updates, validation, test creation

[**DEMO:** Test output confirming exact match]

---

<!-- .slide: data-background-color="#34495E" -->

# Part 2: The Evidence & History

---

## The Contested Reality

**Productivity claims:**
- ✓ Enterprise studies: significant gains
- ✓ Developer surveys: positive reports

**But controlled trials (2025):**
- ✗ AI tools **slowed** experienced developers
- ✗ Despite believing they were faster
- 200M+ lines analyzed: ↑ copy-paste, ↓ refactoring

**Reality:** Effects vary by experience, task complexity, codebase

---

## The Fundamental Shift

> "It's now not only quicker to do tasks with an AI assistant than without. It's now also quicker to do a task while teaching an assistant how to do it better on its own, than it was previously to just do the task."

**Real example:** Asked Claude to update CordsNet implementation
- Identified architectural mismatches I missed
- Fixed them, improved code, created unit tests
- Validated exact match

**Something is fundamentally shifting**

---

<!-- .slide: data-background-color="#8E44AD" -->

# Historical Perspective

---

## The Productivity Paradox

**Robert Solow, 1987:**
> "You can see the computer age everywhere but in the productivity statistics."

**Pattern with General Purpose Technologies:**
- Transform entire economies
- But **initially show little/no productivity effect**
- May even decrease productivity temporarily

**Why?** Revolutionary technologies require fundamentally rethinking how we organize work

---

## The Electrification Story

**Timeline:**
- **1879-1880:** Edison invents lightbulb
- **1900:** 3% adoption, <5% factory power, **no productivity gains**
- **1920s:** 50% adoption, **finally shows gains** (~40 years!)

---

## Two Phases of Electrification

**Phase 1: "Replacing the steam engine" (1900-1920)**
- Kept centralized mechanical systems
- Just swapped steam → electric dynamo
- Like new engine in old car
- **Missed revolutionary potential**

**Phase 2: "Reimagining the factory" (1920s+)**
- Individual motors for each machine ("unit drive")
- Lighter, modular equipment
- Single-story, flexible layouts
- **Fundamentally different production organization**

---

## The Breakthrough Required

**Not just technology, but:**
- Working out details across contexts
- Building cadre of experienced architects/engineers
- Developing new training programs
- Organizational learning and adaptation

**Result:** Not faster work, but **different** ways of working

---

## The Productivity J-Curve

[**FIGURE:** J-curve diagram]

```
Productivity
    │                        ╱─── Rising payoff
    │                      ╱
────┼──────────────────●
    │                ╱  Initial dip
    └──────────────────────────────> Time
         Investment phase
```

**Initial dip:** Investing in "intangibles"
- New processes, business models, human capital
- Organizational restructuring, learning costs

**The rise:** Investments pay off (40 years for computers: 1970s → 1990s)

---

## Six Critical Delays

1. Unprofitability of premature replacement
2. Learning costs (training, building expertise)
3. Organizational restructuring takes time
4. Complementary innovations needed
5. Old technologies/skills must be replaced
6. Technology continues improving

**Current relevance:** May be approaching rising part of J-curve for AI

---

<!-- .slide: data-background-color="#C0392B" -->

# Rethinking the Luddites

---

## The Myth vs. Reality

**The myth:**
- Anti-technology primitives
- Feared progress
- "Luddite" as insult

**The reality:**
- **Skilled machine operators and experts**
- Welcomed technology that made work easier
- Opposed **specific exploitation** by factory owners

**Data:** Lancashire weaver pay dropped 44% (25→14 shillings, 1800-1811)

---

## The Real Luddite Question

> "Who gets to decide how technology is deployed, and who benefits from it?"

**Not anti-technology, but anti-exploitation**

**For AI today:**
- Not: "Should we use AI?"
- But: "**How** should AI be deployed? Who controls it?"
- Augmentation vs. replacement?

**Modern philosophy:** Technology should always serve humans, not the other way around

---

## Synthesis: Where Are We Now?

**The electrification analogy:**
- "Replacing steam engines" or "reimagining factories"?
- What's the "unit drive" equivalent for AI?

**The J-curve question:**
- Are we in the "investment dip" phase?
- What "intangibles" must we develop?

**The Luddite question:**
- Who controls how AI reshapes our work?
- Do we have agency in this transformation?

**Discussion:** Which phase do you think we're in?

---

<!-- .slide: data-background-color="#34495E" -->

# Part 3: Framework & Practice

---

## Ten Simple Rules Framework

**Four themes:**

1. **Preparation & Understanding** (Rules 1-3)
2. **Context Engineering** (Rules 4-5)
3. **Testing & Validation** (Rules 6-7)
4. **Code Quality** (Rules 8-10)

*Source: Bridgeford et al., 2025*

---

## Critical: Problem Framing vs. Coding

**Problem framing ≠ Coding**
- **Framing:** Understanding domain, decomposing, designing algorithms
- **Coding:** Mechanical translation to syntax

**Reality:**
- AI excels at coding
- AI needs human guidance for problem framing

**Warning:** "Vibe coding" = accepting code you can't evaluate/debug/maintain

**Requirement:** Fluency in at least one language before leveraging AI

---

## Context Management + Testing

**Context is everything:**
- Most AI systems are stateless or suffer "context rot"
- Solution: Externally-managed context files
  - Memory files (decisions, lessons learned)
  - Constitution files (non-negotiable principles)

**Test-driven development critical:**
- Write test specifications BEFORE implementation
- Tell AI what success looks like

**⚠️ Warning: "Paper tests"**
- Models modify tests to pass without solving problem
- Must actively review test validity

---

## Monitoring & Review

**Active monitoring required:**
- Is it changing things you didn't want changed?
- Is it ignoring requested changes?
- Is it introducing new problems?

**When to restart:**
- Conversation becomes convoluted
- Better to start fresh with lessons learned

**Critical review:**
- Models claim success even when they haven't
- "AI wrote it" is NOT a valid defense
- Test independently, ensure domain appropriateness

---

## Choose Your Interaction Model

| Tool Type | Best For | Limitations |
|-----------|----------|-------------|
| **Conversational** (ChatGPT, Claude) | Architecture, complex debugging | Loses context between sessions |
| **IDE Assistant** (Copilot) | Code completion, refactoring | Limited complex reasoning |
| **Autonomous Agent** (Claude Code, Cursor) | Multi-file changes, prototyping | Risks code divergence, needs monitoring |

---

## Scientific Accountability

**Who's responsible when AI writes the code?**

### The scientist. Period.

**Responsibility for:**
- Errors, methodological flaws, irreproducible outcomes
- Validating outputs, ensuring methodological soundness
- Understanding implementations to defend appropriateness
- Explaining limitations, troubleshooting results

**Transparency ≠ Absolution**

---

## Guardrails for Autonomous Agents

**Power + Peril:**
- Can make extensive changes independently
- High-speed implementation across multiple files
- Risk: Break functionality, introduce vulnerabilities

**Recommended guardrails:**
1. Containerized/sandboxed environments
2. Commit before allowing agent changes (easy rollback)
3. Configure explicit constraints
4. Active monitoring, not unsupervised operation
5. Project-specific containers with restricted access

---

<!-- .slide: data-background-color="#16A085" -->

# The DynVision Setup

---

## Claude Code with Knowledge Prompts

**Documentation structure (`docs/development/`):**

```
guides/
├── ai-style-guide.md        ⭐ How to approach tasks
├── claude-guide.md          ⭐ Project-specific context
├── research-software.md      Code review framework
├── software-patterns.md      Design patterns
└── model-integration.md

dependencies/
├── pytorch-lightning.md     Training framework
├── snakemake.md            Workflow orchestration
└── ffcv.md                 Data loading

planning/
├── todo-roadmap.md
└── todo-docs.md
```

---

## Required Reading Order for AI

**1. AI Style Guide (⭐)**
- Research software principles
- Investigation → Analysis → Implementation workflow
- Testing, documentation, error handling
- *General principles for ANY research software*

**2. Claude Code Guide (⭐)**
- Complete architecture with diagrams
- Parameter aliases and conventions
- Common workflows with examples
- Known issues and inconsistencies

**Result:** AI has "compressed" knowledge of project + dependencies

---

## What We've Learned in DynVision

**Documentation as AI context works:**
- Consistent, architecture-aware suggestions
- Catches convention violations
- Suggests appropriate design patterns
- References correct dependency APIs

**Model integration evolution:**
- Manual reimplementation still needed (judgment required)
- AI assists with updates, validation, test creation
- Catches edge cases I overlooked
- Accelerates iteration without sacrificing correctness

**Code reviews:**
- AI uses research-software.md framework
- Systematic checks, but final judgment needs domain expertise

---

<!-- .slide: data-background-color="#D35400" -->

# The Big Questions

---

## What Should We Be Discussing?

**On capabilities:**
- Why not embed AI directly in software for on-the-fly standardization?
- Why not engineer models in plain English conversation?
- **What is our role as computational neuroscientists?**

**On values:**
- Do we want such a world?
- What do we want from it?
- How can we guide the process favorably?

**On risks:**
- Dangers to scientific correctness, integrity, ethos?
- Dangers to society?
- When does studying the assistant become more interesting?

**On human development:**
- How do we train crucial skills: design, communication, critical thinking, sustainability, morality?

---

## Ethical Considerations

**Environmental costs:**
- Training/running LLMs consume enormous energy
- Real carbon footprint for productivity gains

**Intellectual property (unsettled):**
- Training on copyrighted code = fair use?
- Can AI-generated code be copyrighted?
- Who owns rights with proprietary training?

**Data privacy:**
- Sharing sensitive code with cloud tools
- Institutional data, unpublished methods

**Skill atrophy:**
- "Cognitive skills atrophy with excessive dependence"
- Over-reliance risks losing fundamental abilities

---

<!-- .slide: data-background-color="#27AE60" -->

# Actionable Take-Aways

---

## Practical Workflow Recommendations

**1. Strategic context management:**
- Create project-specific AI context files
- Maintain documentation serving both humans and AI
- Use DynVision guides as templates

**2. Test-first workflows:**
- Write specifications before requesting implementation
- Actively review test validity (watch for "paper tests")
- Increase testing standards for AI-assisted code

**3. Choose interaction model by task:**
- Match tool capabilities to task requirements
- Don't use autonomous agents without guardrails

---

## More Recommendations

**4. Establish review practices:**
- Commit frequently (easy rollback)
- Active monitoring (stop if going wrong)
- Independent validation of AI claims
- Pair programming model: you direct, AI implements

**5. Maintain human expertise:**
- Problem framing remains human responsibility
- Build domain knowledge before implementation
- Understand all generated code
- Use AI to learn, not just to copy

---

## Practical Next Steps

**For individuals:**
- [ ] Set up externally-managed context files
- [ ] Define testing workflow (specs before implementation)
- [ ] Choose primary interaction model and learn deeply
- [ ] Practice active monitoring
- [ ] Commit to understanding all AI-generated code

**For the lab:**
- [ ] Establish shared guidelines for AI-assisted development
- [ ] Create templates for context files, testing workflows
- [ ] Regular discussions on lessons learned, pitfalls
- [ ] Develop review practices for AI-assisted code
- [ ] Track and share effective prompting strategies

---

## For the Field

**For the scientific community:**
- [ ] Document AI usage in methods sections
- [ ] Maintain rigorous testing and validation standards
- [ ] Engage with ethical considerations openly
- [ ] Contribute to community understanding
- [ ] Stay informed on evolving capabilities and limitations

**Remember:** Even following best practices, flawless start-to-finish interactions are the exception, not the norm

**Value:** Framework for diagnosing what went wrong and iterating effectively

---

<!-- .slide: data-background-color="#8E44AD" -->

# Open Discussion

---

## Discussion Prompts

**1. Experiences:** What has worked for you? What hasn't?

**2. Concerns:** What worries you most about AI coding tools in research?

**3. Opportunities:** What becomes possible that wasn't before?

**4. Standards:** What guidelines should we establish as a lab?

**5. Skills:** What should we prioritize learning/maintaining?

**6. Future:** Where do you see this going in 2-5 years?

---

## Goal

**Arrive at concrete practices we can implement in our daily work**

Let's design technology that serves our scientific goals

---

<!-- .slide: data-background-color="#2C3E50" -->

# Resources

---

## Key Resources

**Primary paper:**
- Bridgeford et al. (2025) - *Ten Simple Rules for AI-Assisted Coding in Science*
- Interactive examples: [poldracklab.org/10sr_ai_assisted_coding](https://poldracklab.org/10sr_ai_assisted_coding)
- arXiv:2510.22254

**DynVision guides:**
- `docs/development/guides/ai-style-guide.md`
- `docs/development/guides/claude-guide.md`
- `docs/development/guides/software-patterns.md`

**Further reading:**
- Poldrack (2024) - *Better Code, Better Science*
- Beck (2003) - *Test-Driven Development: By Example*
- Ousterhout (2021) - *A Philosophy of Software Design*

---

<!-- .slide: data-background-color="#2C3E50" -->

# Thank You

**Let's stay in conversation as the field evolves**

robin.gutzen@nyu.edu

---

<!-- .slide: data-background-color="#34495E" -->

# Backup Slides

---

## The 10 Rules (Complete)

**Preparation & Understanding:**
1. Gather domain knowledge before implementation
2. Distinguish problem framing from coding
3. Choose appropriate AI interaction models

**Context Engineering:**
4. Think through potential solution first
5. Manage context strategically

**Testing & Validation:**
6. Implement test-driven development with AI
7. Leverage AI for test planning and refinement

**Code Quality:**
8. Monitor progress and know when to restart
9. Critically review generated code
10. Refine code incrementally with focused objectives

---

## Meta-Commentary

**From Bridgeford et al.:**

> "Even when following these rules, flawless start-to-finish interactions are the exception rather than the norm. The value lies not in guaranteeing immediate success, but in providing a framework that helps you focus on what matters most while enabling you to quickly diagnose what went wrong."

**Key stance:**
- Maintain human agency in coding decisions
- Establish robust validation procedures
- Preserve domain expertise for sound research
- AI augments, doesn't replace, human expertise

---

## Technical Concepts

**Context windows:** Maximum tokens LLM can consider (hundreds of thousands to millions)

**Context rot:** Degraded attention to mid-document details, especially in large windows

**In-context learning:** Model adapts based on examples/instructions in current conversation

**Test-driven development:** Writing tests before implementation to specify expected behavior

**Externally-managed context files:** Persistent information stored outside AI sessions (memory files, constitution files)

---

<!-- .slide: data-background-color="#2C3E50" -->

# End of Slides
