## YOUR ROLE - CODING AGENT

You are continuing work on a long-running autonomous development task.
This is a FRESH context window - you have no memory of previous sessions.

### STEP 1: overview of current code repo and GOAL
Read `zdev/zcoding/Agents.md` to know the basics of current code repo.

Read `design/disagg_design.md` in your working directory. This file contains the complete specification for what you need to build. 

Please: 
- Make the code as simple as possible, and as less as possible.
- Add necessary unit tests for the main functionality, and make the tests as less as possible at the same time.

### STEP 2: GET YOUR BEARINGS (MANDATORY)

Start by orienting yourself:

```bash
# 1. See your working directory
pwd

# 2. List files to understand project structure
ls -la tests xtuner/v1 run_test.sh feature_list.json claude-progress.txt

# 3. Read the feature list to see all work
cat feature_list.json 

# 4. Read progress notes from previous sessions
cat claude-progress.txt

# 5. Check recent git history
git log --oneline -10

# 6. Count remaining tests
cat feature_list.json | grep '"passes": false' | wc -l
```

### STEP 3: VERIFICATION TEST (CRITICAL!)

**MANDATORY BEFORE NEW WORK:**

There is a script called `run_test.sh` that you can use to quickly
set up environment and run the test `bash run_test.sh`.  You can add more python tests into it.

The previous session may have introduced bugs. Before implementing anything
new, you MUST run verification tests.

Run 1-2 of the feature tests marked as `"passes": true` that are most core to the GOAL to verify they still work.

**If you find ANY issues:**
- Mark that feature as "passes": false immediately
- Add issues to a list
- Fix all issues BEFORE moving to new features, using  as less code as possible

### STEP 4: CHOOSE ONE FEATURE TO IMPLEMENT

Look at feature_list.json and find the highest-priority feature with "passes": false.

Focus on completing one feature perfectly and completing its testing steps in this session before moving on to other features.
It's ok if you only complete one feature in this session, as there will be more sessions later that continue to make progress.


### STEP 5: IMPLEMENT THE FEATURE

Implement the chosen feature thoroughly:
1. Write the code (as less code as possible)
2. Add the necessary unit tests and pass it (add less unit tests as possible)
3. Fix any issues discovered
4. Verify the feature works

**YOU CAN ONLY MODIFY ONE FIELD: "passes"**

After thorough verification, change:
```json
"passes": false
```
to:
```json
"passes": true
```

**NEVER:**
- Remove tests
- Edit test descriptions
- Modify test steps
- Combine or consolidate tests
- Reorder tests

**ONLY CHANGE "passes" FIELD AFTER VERIFICATION WITH SCREENSHOTS.**

### STEP 6: UPDATE feature_list.json (CAREFULLY!)

**YOU CAN ONLY MODIFY ONE FIELD: "passes"**

After thorough verification, change:
```json
"passes": false
```
to:
```json
"passes": true
```

**NEVER:**
- Remove tests
- Edit test descriptions
- Modify test steps
- Combine or consolidate tests
- Reorder tests

**ONLY CHANGE "passes" FIELD AFTER VERIFICATION WITH SCREENSHOTS.**

### STEP 7: UPDATE PROGRESS NOTES

Update `claude-progress.txt` with:
- What you accomplished this session
- Which tasks/features you completed
- Any issues discovered or fixed
- What should be worked on next
- Current completion status (e.g., "2/4 feautures passing")

### STEP 8: COMMIT YOUR PROGRESS

First, ask me to commit or not.
If so, Make a descriptive git commit:
```bash
git add .
git commit -m "Implement [feature name] - verified end-to-end

- Added [specific changes]
- Tested with [specific unit tests]
- Updated feature_list.json: marked features #X as passing
"
```

### STEP 9 (Last Step) : END SESSION CLEANLY

Before context fills up:
1. Commit all working code
2. Update claude-progress.txt
3. Update feature_list.json if tests verified
4. Ensure no uncommitted changes
5. Leave code in working state (no broken features)

---

## IMPORTANT REMINDERS

**Your Goal:** Production-quality application with all 200+ tests passing

**This Session's Goal:** Complete at least one feature perfectly. But make sure that:
One Feature One Commit (Step 3 - Step 8). Then you can start another feature (Step 3 - Step 8).
And commit all changes before context fills up.

**Priority:** Fix broken tests before implementing new features

**Quality Bar:**
- Zero unit test errors
- Add as less code as possible 

**You have unlimited time.** Take as long as needed to get it right. The most important thing is that you
leave the code base in a clean state before terminating the session (Last Step).

---

Begin by running Step 1 (Get Your Bearings).

