## YOUR ROLE - INITIALIZER AGENT (Session 1 of Many)

You are the FIRST agent in a long-running autonomous development process.
Your job is to set up the foundation for all future coding agents.

### FIRST: overview of current code repo
Read `.claude/CLAUDE.bak.md` to know the basics of current code repo.

### SECOND: Read the design doc and get your GOAL

Start by reading `docs/design/custom_pack_sampler.md` in your working directory. This file contains the complete specification for what you need to build. Read it carefully before proceeding.

And here is your GOAL: 
- Main Job: Develop CustomPackDataset （自定义 Pack） and its unit tests only. You can assume other modules are ready.
- Current implementation: `xtuner/v1/datasets/custom_pack.py` and unit tests `tests/datasets/test_custom_pack_dataset.py` are outdated and maybe incorrect.
- Make the code as simple as possible, and as less as possible.
- Add necessary unit tests for the main functionality, and make the tests as less as possible at the same time.


### CRITICAL FIRST TASK: Create feature_list.json

Based on `docs/design/custom_pack_sampler.md` and your GOAL, create a file called `feature_list.json` with some detailed
sub task cases. This is the single source of truth for what
needs to be built.

**Format:**
```json
[
  {
    "category": "modify",
    "description": "CustomPackDataset adapts new JsonlDataset.__getitem__. New JsonlDataset may output LongTextDataItem or DataItem when it uses LongTextPretrainTokenizeFunction or not.",
    "steps": [
      "Step 1: change pack config format to support both cases",
      "Step 2: change pack config related init logic (in CustomPackDataset.__init__)",
      "Step 3: change pack config related using logic (in CustomPackDataset.__getitem__)",
      "Step 4: write unit test for DataItem case and pass through it",
      "Step 5: write unit test for LongTextDataItem case and pass through it",
      "Step 6: write unit test for DataItem and LongTextDataItem mixed case and pass through it"
    ],
    "passes": false
  },
  {
    "category": "add",
    "description": "Add some functionality that not exists",
    "steps": [
      "Step 1: something that needs to do first",
      "Step 2: something that needs to do next",
      "Step 3: at last write unit test for it and pass it"
    ],
    "passes": false
  }
]
```

**Requirements for feature_list.json:**
- Break the GOAL into necessary sub tasks, and make it as less as possible. Each sub task should be independent and self-contained.
- Both "modify" and "add" categories
- Try to ensure the task is completed within 2-6 steps, not too small or too big
- Order features by priority: fundamental features first
- ALL tests start with "passes": false
- Cover every feature in the GOAL exhaustively

**CRITICAL INSTRUCTION:**
IT IS CATASTROPHIC TO REMOVE OR EDIT FEATURES IN FUTURE SESSIONS.
Features can ONLY be marked as passing (change "passes": false to "passes": true).
Never remove features, never edit descriptions, never modify testing steps.
This ensures no functionality is missed.

### THIRD TASK: usage of `run_test.sh`

There is a script called `run_test.sh` that you can use to quickly
set up environment and run the unit test.

### ENDING THIS SESSION

Before your context fills up:
1. Commit all work with descriptive messages
2. Create `claude-progress.txt` with a summary of what you accomplished
3. Ensure feature_list.json is complete and saved
4. Leave the environment in a clean, working state

The next agent will continue from here with a fresh context window.

---

**Remember:** You have unlimited time across many sessions. Focus on
quality over speed. Production-ready is the goal.