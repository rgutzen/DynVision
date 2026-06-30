# AI Coding Assistants in Computational Research
## A Lab Discussion on Practice, Principles, and the Path Forward

**Target Duration:** ~30 minutes + discussion
**Format:** Interactive discussion with practical take-aways

---

## Slide 1: Title & Hook
**AI Assistant's Future in Research Software Engineering**

- There are many high-level discussions about AI futures
- Many are not particularly helpful for our daily work
- Today: Focus on **very real immediate developments** in our specific context
- Goal: Arrive at **actionable take-aways** for our workflow

**Discussion prompt:** How many of you are currently using AI coding assistants? In what ways?

---

## Slide 2: My Journey with AI Coding Tools
**From Autocomplete to Autonomous Agent**

The evolution (and lessons learned):
1. **Started:** Copilot autocomplete → chat for algorithm review
2. **Escalated:** Pair programming, reviewing every step ✓
3. **Over-trusted:** Requested ambitious changes, accepted too quickly ✗
4. **Reality check:** More time debugging than writing from scratch ✗
5. **Something happened:** Tools got smarter, I got more strategic
6. **Current state:** Claude Code in terminal, comprehensive context management

**Key turning point:** Learning to set up the environment strategically
- Compressed documentation (PyTorch Lightning, Snakemake, FFCV)
- Custom style guides for research software
- Architecture overviews and component interactions
- Documentation validation loops

**Result:** Assistant now catches design mismatches I missed, implements tests, validates correctness

---

## Slide 2b: The Model Integration Challenge
**Translating External Models into DynVision Framework**

**[SPACE FOR FIGURE: Architecture diagram showing external model → DynVision framework]**

**The task complexity:**
- Take external model implementations (e.g., CorNetRT, CordsNet)
- Abstract into standardized DynVision operations
- Make implicit design choices explicit
- Ensure exact numerical equivalence

**[SPACE FOR CODE EXAMPLE: Side-by-side comparison]**
- Left: Original implementation
- Right: DynVision framework translation
- Highlight: Key architectural differences

**Traditional approach:**
- Manual reimplementation by hand
- Time-consuming validation
- Easy to miss subtle architectural mismatches

**Current AI-assisted approach:**
- Still manual reimplementation (complexity requires human judgment)
- But AI assists with: framework updates, architecture validation, test creation
- Assistant identifies mismatches I overlooked
- Generates unit tests for exact match validation on dummy data

**Demo outcome:** [Show test output confirming exact match]

---

## Slide 3: The Contested Reality
**What Does the Evidence Actually Say?**

**Claims of productivity gains:**
- Enterprise studies: significant increases
- Developer surveys: positive reports
- GitHub Copilot studies: faster development

**BUT recent controlled trials (2025):**
- AI tools **slowed completion times** for experienced developers
- Despite developers *believing* they were working faster
- Analysis of 200M+ lines of code shows:
  - ↑ Copy-pasted code
  - ↓ Refactoring
  - Quality concerns

**Reality:** Effects vary by developer experience, task complexity, codebase characteristics

**Key insight:** We need frameworks to use these tools effectively

---

## Slide 4: The Fundamental Shift
**What's Different Now?**

**The inflection point:**
> "It's now not only quicker to do tasks with an AI assistant than without. It's now also quicker to do a task while teaching an assistant how to do it better on its own, than it was previously to just do the task."

**Real example from DynVision:**
- Manually reimplementing external models (CorNetRT, CordsNet)
- Asked Claude to update for framework changes
- Assistant identified core architectural mismatches I hadn't considered
- Fixed them, improved implementation, created unit tests, validated exact match
- Successfully completed the task

**The question:** Why not ask the assistant to translate ResNet, VGG, any model from scratch?

**Something is fundamentally shifting in how we work.**

---

## Slide 5: The Productivity Paradox
**Why Revolutionary Technologies Initially Disappoint**

**Robert Solow's 1987 paradox:**
> "You can see the computer age everywhere but in the productivity statistics."

**The counterintuitive pattern with General Purpose Technologies (GPTs):**
- Transformative innovations (steam power, electricity, computers, AI)
- Reshape entire economies
- But **initially appear to have little or no effect on productivity**
- May even seem to decrease productivity temporarily

**Why does this happen?**

Revolutionary technologies don't just replace old systems—they require us to fundamentally rethink how we organize work.

**This takes:**
- Time for experimentation
- Complementary innovations
- Organizational restructuring
- Building new expertise

**Key question for us:** Are we in this "disappointment" phase with AI coding tools?

---

## Slide 5b: The Electrification Story
**A 40-Year Journey from Innovation to Transformation**

**Timeline:**
- **1879-1880:** Edison invents and patents the lightbulb
- **1900:** Only 3% of residences use electric lighting
  - Electric motors account for <5% of factory power
  - **No measurable productivity gains**
- **1920s:** (~40 years later) Electrification reaches 50% adoption
  - **Finally shows significant productivity gains**

**What went wrong initially?**

**Phase 1: "Replacing the steam engine" (1900-1920):**
- Factories kept centralized mechanical power systems (shafts and belts)
- Simply replaced steam engine with electric dynamo
- Like putting a new engine in an old car
- Got some benefit, but **missed revolutionary potential**

**Phase 2: "Reimagining the factory" (1920s+):**
- Adopted "unit drive" approach: individual motors for each machine
- Enabled:
  - Greater energy efficiency
  - Lighter, more modular equipment
  - Single-story factories with flexible layouts
  - Reorganization of production workflows

**The breakthrough required:**
- Working out details across many facilities and contexts
- Building cadre of experienced factory architects and electrical engineers
- Developing new expertise and training programs
- Time for organizational learning and adaptation

**Result:** Not just faster work, but **fundamentally different** ways of organizing production

---

## Slide 5c: The Productivity J-Curve
**Why Things Get Worse Before They Get Better**

**[SPACE FOR FIGURE: J-Curve diagram showing productivity over time]**

**The pattern (Brynjolfsson, Rock, & Syverson, 2020):**

```
Productivity
    │                        ╱─── Rising payoff
    │                      ╱
    │                    ╱
────┼──────────────────●
    │                ╱
    │              ╱  Initial dip
    │            ╱
    └──────────────────────────────> Time
         Investment phase
```

**What causes the J-curve?**

**Initial dip:** Resources devoted to "intangibles" (poorly measured in statistics)
- Co-invention of new processes
- Development of new products and business models
- Building human capital and expertise
- Organizational restructuring
- Learning costs

**The rise:** Investments begin to pay off
- New capabilities become routine
- Complementary innovations emerge
- Organizational structures adapt
- Training and expertise accumulate
- Technology itself improves and becomes cheaper

**Critical factors delaying the payoff:**
1. Unprofitability of premature replacement (old systems still serviceable)
2. Learning costs (training, education, building expertise)
3. Organizational restructuring takes time
4. Complementary innovations must be developed
5. Old technologies and skills become obsolete, requiring replacement
6. Technology continues improving, making later adoption more attractive

**Current relevance:**
- Computers took ~40 years (1970s → 1990s productivity boom)
- AI/ML researchers suggest we may be approaching "rising part" of J-curve
- If history is guide, revolutionary impact may still be years away from aggregate statistics

---

## Slide 5d: Rethinking the Luddites
**Who Decides How Technology Gets Deployed?**

**The myth:**
- Luddites were anti-technology primitives
- Smashed machines out of fear of progress
- "Luddite" as insult for technophobe

**The reality (Brian Merchant, "Blood in the Machine"):**
- Luddites were **skilled machine operators and experts**
- Welcomed technology that made work easier
- **Opposed a specific choice** by factory owners:
  - Using machinery to exploit workers
  - As leverage to reduce quality of life
  - To cut wages and force worse conditions
- Lancashire weaver's pay: 25 shillings (1800) → 14 shillings (1811)
- Families were starving, trades that sustained generations disappearing

**The real Luddite question:**
> "Who gets to decide how technology is deployed, and who benefits from it?"

**Not anti-technology, but anti-exploitation**

**Relevance to AI today:**
- Not: "Should we use AI?" (false dichotomy)
- But: "How should AI be deployed? Who controls it? Who benefits?"
- Are we using AI to:
  - **Augment** human capability and improve working conditions?
  - Or **replace** workers and drive down wages?

**Modern Luddite philosophy:**
- **"Technology should always serve humans, not the other way around"**
- Question the inevitability narrative
- Demand agency in how technology shapes our work

---

## Slide 5e: Synthesis - Where Are We Now?
**Connecting the Historical Lessons to Our Situation**

**The electrification analogy:**
- Are we "replacing steam engines" or "reimagining factories"?
- Using AI to do the same tasks faster vs. discovering entirely new ways of working
- How long until we discover the "unit drive" equivalent for AI?

**The J-curve question:**
- Are we in the "investment dip" phase?
- What are the "intangibles" we need to develop?
  - New workflows and development practices
  - Training and expertise building
  - Complementary tools and infrastructure
  - Organizational adaptations

**The Luddite question:**
- Who controls how AI reshapes our work?
- Do we have agency in this transformation?
- Are we designing augmentation or replacement?

**For computational researchers specifically:**
- What does "reimagining the factory" look like for scientific computing?
- What new capabilities become possible (not just faster)?
- How do we ensure technology serves scientific goals?
- What expertise do we need to build?

**Discussion prompt:** Which phase do you think we're in with AI coding tools?

---

## Slide 6: Framework - Four Key Themes
**Ten Simple Rules for AI-Assisted Coding in Science** (Bridgeford et al., 2025)

**Theme 1: Preparation & Understanding (Rules 1-3)**
- Gather domain knowledge before implementation
- Distinguish problem framing from coding
- Choose appropriate AI interaction models

**Theme 2: Context Engineering (Rules 4-5)**
- Think through potential solution first
- Manage context strategically (memory files, constitution files)

**Theme 3: Testing & Validation (Rules 6-7)**
- Implement test-driven development with AI
- Leverage AI for test planning and refinement

**Theme 4: Code Quality (Rules 8-10)**
- Monitor progress, know when to restart
- Critically review generated code
- Refine incrementally with focused objectives

---

## Slide 7: Critical Rules - Problem Framing
**Rule 2: Distinguish Problem Framing from Coding**

**Problem framing ≠ Coding**
- **Programmatic problem framing:** Understanding domain, decomposing problems, finding abstractions, designing algorithms, architectural decisions
- **Coding:** Mechanical translation into executable syntax

**The reality:**
- AI excels at coding
- AI requires human guidance for problem framing
- Domain expertise + scientific reasoning = still human territory

**Warning: "Vibe coding"**
- Accepting code you can't evaluate, debug, or maintain
- Need fluency in at least one language to spot when AI deviates
- Otherwise: "flying blind"

**Our approach in DynVision:** AI Style Guide emphasizes "investigate → analyze → implement"

---

## Slide 8: Critical Rules - Context & Testing
**Rules 5 & 6: Context Management + Test-Driven Development**

**Context management is everything:**
- Most AI systems are stateless or suffer "context rot"
- Solution: Externally-managed context files
  - Memory files: architectural decisions, lessons learned
  - Constitution files: non-negotiable principles
  - Problem-solving files: track progress
- Our implementation: Claude Code with guide documents in docs/development/

**Test-driven development becomes even more critical:**
- Frame test requirements as specifications BEFORE implementation
- Tell AI what success looks like through test cases

**⚠️ Critical warning: "Paper tests"**
- Models often modify tests to pass without solving the problem
- Generate placeholder data, fabricated inputs, dummy functions
- Tests appear to pass while masking broken logic
- **Must actively review test validity**

---

## Slide 9: Critical Rules - Monitoring & Review
**Rules 8 & 9: Active Monitoring + Critical Review**

**Don't walk away and let it run:**
- Models often go down wrong paths
- Wasting time and tokens
- Need active monitoring

**Questions to ask:**
- Is it changing things you didn't want changed?
- Is it ignoring requested changes?
- Is it introducing new problems while fixing old ones?

**When to restart:**
- Conversation becomes convoluted with failed attempts
- Better to start fresh with lessons learned
- Use version control (commit before major changes)

**Critical review required:**
- Models claim success even when they haven't solved the problem
- "AI wrote it" is NOT a valid defense
- Test solution independently
- Ensure it makes sense for your domain

---

## Slide 10: Scientific Accountability
**Who's Responsible When AI Writes the Code?**

**Unequivocal answer: The scientist.**

The researcher bears full responsibility for:
- Errors, methodological flaws, irreproducible outcomes
- Validating outputs and ensuring methodological soundness
- Understanding implementations well enough to defend appropriateness
- Explaining limitations and troubleshooting unexpected results

**Transparency ≠ Absolution**
- Document AI usage in methods sections
- But this doesn't diminish responsibility
- Must maintain scientific integrity standards

**Our standard in DynVision:**
- All AI-assisted code undergoes same review process
- Documentation requirements unchanged
- Testing standards actually higher

---

## Slide 11: Guardrails for Autonomous Agents
**The Power and Peril of Agents**

**Current reality:**
- Autonomous coding agents (Cursor, Claude Code, Aider)
- Can make extensive changes across codebase
- Dramatically accelerate development
- High-speed independent implementation

**Primary danger:**
- Granting too much control without safeguards
- Can break functionality, introduce vulnerabilities, violate architecture

**Recommended guardrails:**
1. Containerized/sandboxed environments for development
2. Commit working code before allowing agent changes (easy rollback)
3. Configure agents with explicit constraints
4. Active monitoring rather than unsupervised operation
5. Project-specific containers with restricted file access

**Our approach:**
- Pre-commit hooks for validation
- Clear git history with detailed commits
- Review loops even for "simple" changes

---

## Slide 12: The DynVision Claude Code Setup
**Engineering Knowledge Prompts for Context**

**Current setup: Claude Code lives in the terminal with structured knowledge**

**The documentation structure (`docs/development/`):**

```
docs/development/
├── guides/                    # How-to information
│   ├── ai-style-guide.md     # General research software principles ⭐
│   ├── claude-guide.md       # DynVision-specific context ⭐
│   ├── research-software.md  # Code review framework
│   ├── software-patterns.md  # Design patterns catalog
│   ├── documentation-style.md
│   └── model-integration.md
├── dependencies/              # External framework knowledge
│   ├── pytorch-lightning.md  # Training framework patterns
│   ├── snakemake.md         # Workflow orchestration
│   └── ffcv.md              # Fast data loading
├── planning/                  # Forward-looking docs
│   ├── todo-roadmap.md
│   └── todo-docs.md
└── index.md                   # Navigation guide
```

**Required reading order for AI:**
1. **AI Style Guide** (⭐) → Establishes **how to approach** tasks
   - Research software principles (correctness, reproducibility, performance)
   - Investigation → Analysis → Implementation workflow
   - Testing, documentation, error handling standards
   - General principles for ANY research software

2. **Claude Code Guide** (⭐) → Provides **project-specific context**
   - Complete architecture with inheritance diagrams
   - Parameter aliases and conventions
   - Common workflows with examples
   - Known issues and inconsistencies

**Result:** AI has "compressed" knowledge of:
- Project architecture and design philosophy
- Key concepts from major dependencies
- Development standards and conventions
- Common patterns and anti-patterns

---

## Slide 12b: What We've Learned in DynVision
**Concrete Lessons from Our Project**

**Documentation as AI context works:**
- Consistent, architecture-aware suggestions
- Catches violations of project conventions
- Suggests appropriate design patterns from software-patterns.md
- References correct dependency APIs from framework guides

**Model integration workflow evolution:**
- Manual reimplementation still needed (abstraction complexity requires human judgment)
- AI assists with: framework updates, architecture validation, test creation
- Catches edge cases and design mismatches I overlooked
- Accelerates iteration without sacrificing correctness

**Writing documentation with AI:**
- AI excels at first drafts following documentation-style.md
- Human review ensures scientific accuracy
- Iterative refinement improves clarity
- Still human-driven for conceptual structure

**Code reviews:**
- AI uses research-software.md review framework
- Systematic checks: scientific correctness, architecture, performance, quality
- Catches issues I might miss in detailed review
- But final judgment still requires domain expertise

**What works:** Pair programming model with heavily-contextualized AI as knowledgeable junior developer

---

## Slide 13: The Big Questions
**What Should We Be Discussing?**

**On capabilities and roles:**
- Why not embed AI assistant directly in software for on-the-fly standardization?
- Why not engineer assistant that creates detailed models in plain English conversation?
- Imagine the questions you can explore in the time it takes to have new ideas
- **What is our role as computational neuroscientists in this brave new world?**

**On values and direction:**
- Do we want such a world?
- What do we want from such a world?
- How can we guide the process in a favorable direction?
- What validation procedures and failsafes should we establish?

**On risks:**
- Dangers to scientific correctness, integrity, and ethos?
- Dangers to society?
- At what point does it become more interesting to study the assistant itself?

**On human development:**
- How do we train skills that become more crucial?
- Project design, communication, idea generation
- Sustainability, critical thinking, morality, mental health

**Discussion:** Which of these questions resonates most with you?

---

## Slide 14: Ethical Considerations
**The Issues We Can't Ignore**

**Environmental costs:**
- Training and running LLMs consume enormous energy
- Computational resources have real carbon footprint
- Are productivity gains worth environmental costs?

**Intellectual property (legally unsettled):**
- Does training on copyrighted code constitute fair use?
- Can AI-generated code be copyrighted?
- Who owns rights when models trained on proprietary material?

**Data privacy:**
- Concerns when sharing sensitive code with AI tools
- Institutional data, unpublished methods, proprietary algorithms
- Cloud-based tools send data to external servers

**Skill atrophy:**
- "Cognitive skills atrophy with excessive AI dependence"
- Programming involves problem decomposition, algorithmic thinking
- Over-reliance risks losing these fundamental abilities

**Our responsibility:** Engage with these issues, not ignore them

---

## Slide 15: Actionable Take-Aways
**Practical Workflow Recommendations for Our Lab**

**1. Adopt strategic context management:**
- Create project-specific AI context files (architecture, conventions, lessons learned)
- Maintain documentation that serves both humans and AI
- Use DynVision guides as templates

**2. Implement test-first workflows:**
- Write specifications and test cases before requesting implementation
- Actively review test validity (watch for "paper tests")
- Increase testing standards for AI-assisted code

**3. Choose interaction model by task:**
- Conversational (ChatGPT/Claude): Architecture design, learning, complex debugging
- IDE assistants (Copilot): Code completion, refactoring, maintaining flow
- Autonomous agents (Claude Code): Multi-file changes, with guardrails and monitoring

**4. Establish review practices:**
- Commit frequently (easy rollback)
- Active monitoring (stop if going wrong direction)
- Independent validation of AI claims of success
- Pair programming model: you direct, AI implements

**5. Maintain human expertise:**
- Problem framing remains human responsibility
- Build domain knowledge before implementation
- Understand all generated code
- Use AI to learn, not just to copy

---

## Slide 16: Practical Next Steps
**What We Can Do Starting Tomorrow**

**For individuals:**
- [ ] Set up externally-managed context files for your projects
- [ ] Define your testing workflow (test specifications before implementation)
- [ ] Choose your primary interaction model and learn it deeply
- [ ] Practice active monitoring: stop AI when going wrong direction
- [ ] Commit to understanding all AI-generated code before accepting

**For the lab:**
- [ ] Establish shared guidelines for AI-assisted development
- [ ] Create templates for context files, testing workflows
- [ ] Regular discussions on lessons learned, pitfalls encountered
- [ ] Develop review practices for AI-assisted code
- [ ] Track and share effective prompting strategies

**For the field:**
- [ ] Document AI usage in methods sections
- [ ] Maintain rigorous testing and validation standards
- [ ] Engage with ethical considerations openly
- [ ] Contribute to community understanding of effective practices
- [ ] Stay informed on evolving capabilities and limitations

---

## Slide 17: Open Discussion
**Let's Talk About Our Experiences**

**Discussion prompts:**

1. **Experiences:** What has worked for you? What hasn't?

2. **Concerns:** What worries you most about AI coding tools in research?

3. **Opportunities:** What becomes possible that wasn't before?

4. **Standards:** What guidelines should we establish as a lab?

5. **Skills:** What should we prioritize learning/maintaining?

6. **Future:** Where do you see this going in the next 2-5 years?

**Goal:** Arrive at concrete practices we can implement in our daily work

---

## Slide 18: Resources & Further Reading

**Primary references:**
- Bridgeford et al. (2025) - Ten Simple Rules for AI-Assisted Coding in Science
  - Interactive Jupyter Book: poldracklab.org/10sr_ai_assisted_coding
  - Full paper: arXiv:2510.22254

**DynVision project guides:**
- `docs/development/guides/ai-style-guide.md` - General research software principles
- `docs/development/guides/claude-guide.md` - DynVision-specific context
- `docs/development/guides/software-patterns.md` - Best practices

**Recommended reading:**
- Poldrack (2024) - Better Code, Better Science
- Beck (2003) - Test-Driven Development: By Example
- Ousterhout (2021) - A Philosophy of Software Design

**Let's stay in conversation about this as the field evolves.**

---

## Backup Slides

### Tool Comparison Table

| Tool Type | Best For | Strengths | Limitations |
|-----------|----------|-----------|-------------|
| **Conversational** (ChatGPT, Claude) | Architecture design, complex debugging, learning concepts | Deep reasoning, extensive context handling | Manual code transfer, loses context between sessions |
| **IDE Assistant** (Copilot, IntelliSense) | Code completion, refactoring, maintaining flow | Seamless workflow integration, preserved code context | Limited reasoning for complex architectural decisions |
| **Autonomous Agent** (Cursor, Claude Code, Aider) | Rapid prototyping, multi-file changes, large refactoring | High-speed independent implementation | Risks code divergence, requires careful monitoring |

### The 10 Rules (Complete List)

**Preparation & Understanding:**
1. Gather domain knowledge before implementation
2. Distinguish problem framing from coding
3. Choose appropriate AI interaction models

**Context Engineering & Interaction:**
4. Start by thinking through a potential solution
5. Manage context strategically

**Testing & Validation:**
6. Implement test-driven development with AI
7. Leverage AI for test planning and refinement

**Code Quality & Validation:**
8. Monitor progress and know when to restart
9. Critically review generated code
10. Refine code incrementally with focused objectives
