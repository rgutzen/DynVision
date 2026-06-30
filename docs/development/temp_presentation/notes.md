## AI Assistant's future in RSE dev
There are many high-level discussions going on about AI futures, many are not particularly helpful. But there are very real immediate developments, that should be discussed in for our specific context to come up with **actionable take-aways**.

I started using copilot autocomplete. then using the chat to review and optimize algorithms. Then I described implementation intentions and went with the bot through the development, reviewing every step. Then I just talked with the bot, requesting ambitious changes, and accepting to quickly. Then I spend more time debugging and manually rewriting as I would have needed to write it myself in the first place. Then something happend. Then AI assistants kept becoming smarter, getting larger context windows, better workflow integration, and I got more strategic with settings and prompts. 
Claude Code now lives in my terminal and is intimatly familiar with the toolbox. It has a memory context of the compressed documentation of my main dependencies pytorch-lightning, snakemake, ffcv. It also has a style-guide tailored for the special demands of research software development, which the bot previously summarized from a collection of talks i once gave. It has an overview, of the toolbox, its components and how they interact, and reads and checks the documentation.
Now, as I still reimplement external models in the DynVision framework by hand because of the complexity to abstract the original implementation into a sequence of standardized operations making implicit design choices explicit. But as I ask Claude to update the reimplementation to incorporate some recent changes in the toolbox it imidatly  pinpoints some core mismatches in the architectures that I haven't thought of in the first place, fixes them, makes the implementation much better, implements a unit test to validate an exact match on dummy data, runs the test to confirm, and wishes me a good day.
At this point, why not ask the Assisstent to translate and reimplement ResNet, VGG, or any other model. Something is fundamentally shifting in the way we work. And we should talk about that. There are completely novel possibilities and completely novel challenges.
Why not embed an AI Assisstent directly into the software that standardizes you model implementation on the fly. Why not engineer an Assisstent who creates a detailed and formalized biologically realistic computer vision model in conversation in plain english. Image the questions you can explore in a timeframe that it takes you to come up with new ideas. What is our role as computational neuroscientist in face of this brave new world?

It's now not only quicker to do task with an ai asisstent than without. It's now also quicker to do a task while teaching an assisstent how to do it better on its own, then it was previously to just do the task.

Do we want such a world?
What do we want from such a world?
How can we guide the process in a favorable direction?
What validation procedures and failsafes should we establish?
What are dangers to scientific correctness, integrity, and ethos?
What are dangers to society?
At what point does it become more interesting to study the Assisstent itself instead?
How do we train ourselves on the skills that will become so much more crucial? Like Project design, communication, idea generation, sustainability, critical thinking, morality, mental health.

Looking at previous technological breakthroughs: 
When electricity became widespread available, people started replacing their machines running on steam engines with electricity. It still took 10-15 years to see an actual productivity increase when people discovered the new possibilities that the technology brought, such as decentralized engines and parallelization such as the assembly line.

Luddites philosophy: Technology should always sever humans not the other way around

Writing documentation