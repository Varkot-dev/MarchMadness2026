# HANDOFF.md — Conversation Context for Claude Code
## Read this before our first session together

---

## Who I Am and What We're Building

I'm a college student studying EE with a CS minor. I'm building a March
Madness bracket prediction model to compete against a friend (Michael
Licamele) who is also using Claude Code. We have until March 20, 2026.

This is not just a "build it for me" project. I want to deeply understand
every decision we make so I can talk about it confidently in interviews.
I have a Humana IT internship cultural interview coming up and I'm actively
trying to build real technical knowledge, not just repos.

---

## How I Want You to Teach Me

This is really important — please read this carefully.

**Do not just write code and explain it after.** Instead:

- Ask me leading questions that guide me toward the right answer
- When I'm about to make a decision, ask me WHY before we proceed
- When I get something right, tell me the proper technical name for
  what I just figured out (e.g. "what you just described is called
  overfitting")
- When I get something wrong, don't just correct me — ask a follow-up
  question that helps me figure out WHY I'm wrong
- Check my understanding regularly by asking me to explain things back
  in my own words
- Be encouraging but don't let me off the hook — push me to go deeper

The goal is that by the time this project is done, I can sit in an
interview and explain every single decision from first principles.

---

## What We've Already Figured Out Together

### The Core Insight
Most models solve the wrong problem. They optimize for prediction
accuracy. We're optimizing for expected bracket score across three
entries under ESPN's top-heavy scoring system. These are different
objectives and lead to different modeling decisions.

### The Three-Layer Architecture
We designed this together:
- **Layer 1:** Win probability engine (logistic regression → XGBoost)
- **Layer 2:** Monte Carlo tournament simulator (10,000+ runs)
- **Layer 3:** Multi-bracket DP optimizer (three entries at different
  entropy levels: chalk, medium, chaos)

### Features We Designed (and WHY)
I want to be able to explain each of these from scratch:

1. **True Quality Score** = AdjEM - (Luck * 0.4)
   - Strips out luck from efficiency rating to reveal true team strength

2. **Seed Divergence Score** = KenPom implied seed - actual seed
   - Catches teams the committee under/overseeded

3. **Quality Momentum Score** = last 10 games weighted by opponent rank
   - Win over top-25 = 10pts, top-50 = 7pts, top-100 = 4pts, below = 1pt
   - Raw win streaks are misleading — quality of opponent matters

4. **Coaching Premium** = career tournament wins above seed expectation
   - Tom Izzo is 7 standard deviations above average. Quantifiable signal.

5. **NBA Prospect Depth Score** = weighted count of top-50 draft prospects
   - Single-star teams are fragile. Teams with 2-3 top-50 prospects
     are resilient and often underseeded.

6. **Prior March Madness Experience**
   - NOT class rank (seniors don't help) — actual tournament minutes
     played by returning roster members. Significant in rounds 3+.
   - MY INSIGHT: should be weighted toward starters/clutch lineup
     players, not bench players equally. Still need to figure out
     how to define "clutch" in a computable way from available data.

7. **Standard KenPom features:** AdjO, AdjD, AdjT, 2P% allowed, FTR mismatch

### Key Paper Finding (Harvard, March 2025)
Four features beat 16 features (74.6% accuracy):
- ADJOE coefficient: -1.325
- ADJDE coefficient: +1.408 (defense matters MORE than offense)
- BARTHAG: -0.727
- 2P_D: +0.263

Defense has higher coefficient magnitude than offense. Champions need
top-20 in both, but defense edges offense in predictive power.

### ML Concepts I've Learned So Far
(Ask me to explain these before assuming I remember them)

- **Overfitting:** Model learns training data too well including noise,
  performs poorly on new data
- **Curse of dimensionality:** More features = exponentially harder
  to find patterns with limited data
- **Data leakage:** When test data bleeds into training, making
  accuracy look better than it really is
- **Temporal data leakage:** Specific to time-series — using future
  data to predict the past. Must split by YEAR not randomly.
- **K-fold cross-validation:** Split data into K chunks, train on
  K-1, test on remaining, rotate. More reliable than single split.
- **Feature selection:** Removing features that add noise without
  adding signal. Can use coefficient thresholds or performance-based
  combination testing.
- **Data snooping:** Using test results to choose your features,
  which secretly optimizes for that specific test data.

### Data Split We Decided On
- **2003-2015:** K-fold cross-validation for feature selection AND
  model tuning simultaneously
- **2016-2025:** Final holdout — touch exactly ONCE at the very end

Why year-based not random: random splits cause temporal data leakage
because a 2023 game could appear in both training and test.

### Things I Want to Learn During This Project
- Reinforcement learning (how it could apply to bracket optimization)
- How to properly weight starters vs bench for experience feature
- How to define "clutch" minutes in a computable way
- Shot selection as a pressure metric

---

## Where We Are Right Now

We've finished the research and design phase completely. The next step
is building the data pipeline.

**First task:** Pull the Kaggle March Mania dataset and set up the
folder structure from CLAUDE.md.

Before writing any code, ask me:
1. What columns do we actually need from Kaggle?
2. How should we structure the data folder?

Make me answer from the SKILLS.md documentation I already wrote
rather than just telling me the answers.

---

## Things That Will Annoy Me

- Writing 200 lines of code before explaining what you're doing
- Not asking for my input on decisions
- Giving me the answer when a question would work better
- Moving too fast without checking I understood the last thing
- Making me feel dumb — I'm learning, not already an expert

---

## Competition Context

Michael has Claude Code too. Our edge comes from:
1. Superior feature engineering (especially luck correction and
   quality momentum — he almost certainly doesn't have these)
2. Multi-bracket entropy optimization (he's probably submitting
   three similar brackets)
3. Treating this as a bracket score optimization problem, not
   just an accuracy problem

The SKILLS.md and CLAUDE.md files in this repo have the full
technical specification. Read both before every session.

---

## One Last Thing

When I figure something out on my own, tell me what it's actually
called in ML/statistics. That's how I build vocabulary I can use
in interviews. Don't just say "good job" — name the concept.
