# **Engineering Principles and Practices**

## **Preamble**

This document is the single source of truth for the engineering principles, cognitive
frameworks, technical defaults, and workflow disciplines that guide how we (humans and AI
agents) plan, build, test, and ship software.

Our priorities are **consistency**, **simplicity**, and **iterative complexity**. This guide
ensures that we build maintainable, reliable systems that deliver value quickly and sustainably.

**Rule 0:** Make things easier to change. Everything else supports that.

---

## **1. Core Principles**

These principles guide decision-making under uncertainty. Simplicity is our foundation; complexity
must be justified by real-world need.

### **1.1 Gall’s Law: Iterative Complexity**

> “A complex system that works is invariably found to have evolved from a simple system that
> worked. A complex system designed from scratch never works.” – John Gall

**Concept:** Systems evolve successfully through working feedback loops, not big-bang design.

**Implications:**

* Ship the smallest working vertical slice (a "walking skeleton") first.
* Add capability incrementally once stability is proven.
* Defer irreversible decisions until they are required.

**Examples:**

* *Positive:* Deploy to a simple managed service first; scale only when proven bottlenecks appear.
* *Negative:* Spending months building a generic “platform” before delivering a single feature.

---

### **1.2 Pragmatic Simplicity (BSSN + YAGNI)**

> “The Best Simple System for Now is the simplest system that meets the needs of the product right
> now.” – Dan North
>
> “Always implement things when you actually need them, never when you just foresee that you need
> them.” – XP

**Concept:** Build only what is necessary *now*, to the right quality level. Simplicity is achieved
by solving today’s real problem cleanly, not by guessing tomorrow’s.

**Implications:**

* Avoid speculative hooks and “just-in-case” flexibility.
* Maintain flexibility through clean seams, not complex abstractions.
* Delay non-essential generalization until it’s proven useful.

**Examples:**

* *Positive:* Use Postgres full-text search until proven insufficient—then consider Elasticsearch.
* *Negative:* Building a multi-provider auth system when only email/password is needed.

---

### **1.3 Mahoney’s Simplicity Creed (Data Primacy)**

> “Simpler systems tend to be more reliable, faster to change, and easier to work with... Data is
> more important than code.” – Kevin Mahoney

**Concept:** Prioritize data integrity and consistency over code cleverness. Data outlives code.

**Implications:**

* Design **data-first**: normalized, constrained schemas.
* Enforce integrity at the database layer: NOT NULL, FOREIGN KEYS, CHECK constraints, and RLS.
* Favor consistency over novelty; fewer technologies = less complexity.

**Examples:**

* *Positive:* Using database constraints for referential integrity.
* *Negative:* Polyglot persistence (Mongo + Redis + Postgres) without proven need.

---

### **1.4 Make Invalid States Unrepresentable**

**Concept:** Structure types, schemas, and APIs so invalid or contradictory states are impossible.

**Implications:**

* Prefer discriminated unions/enums to boolean flags.
* Use non-nullable types for required data.
* Let type systems and constraints catch errors early.

**Example (TypeScript):**

```ts
type ApiState<T> =
  | { status: 'loading' }
  | { status: 'success', data: T }
  | { status: 'error', error: Error };
```

Prevents impossible states like loading + error simultaneously.

---

### **1.5 Refactor When It Hurts (Duplication, Deletion, and Abstraction)**

**Concept:** Combine the Rule of Three and Optimize for Deletion: delay abstraction until patterns
are proven, and design so features can be safely removed.

**Heuristic:**

1. First use: just do it.
2. Second use: duplicate.
3. Third use: abstract.

**Implications:**

* Refactor only after genuine repetition.
* Keep components modular and cohesive.
* Minimize dependencies and use clear boundaries.

**Examples:**

* *Positive:* Extracting a shared date-formatting utility after the third duplicate.
* *Negative:* Building a flexible date library before a single concrete use case.
* *Positive:* Wrapping a third-party API behind an internal interface for easy replacement.
* *Negative:* Scattering a feature’s logic across unrelated modules.

---

### **1.6 Throw Away the First Draft (The Spike Rule)**

> “Keeping the code, and knowing that you might, completely changes the psychology of the
> prototyping phase.” – N. Tietz

**Concept:** Spikes exist to learn, not to ship. Code written to explore uncertainty must be
deleted.

**Implications:**

* Time-box spikes; delete afterward.
* Hardcode and skip testing in spikes.
* Never merge exploratory code.

**Examples:**

* *Positive:* Spiking a quick API call to learn behavior, then rewriting cleanly.
* *Negative:* “Cleaning up” and merging a spike into production.

---

### **1.7 Ship Quickly, Deliberately (Pragmatic Velocity)**

> “Velocity comes from minimizing complexity and using mastered tools.” – Evan Hahn

**Concept:** True speed arises from simplicity, tight scope, and familiarity—not cutting quality.

**Implications:**

* Use boring, reliable tech.
* Reduce scope before compromising quality.
* Match quality to lifespan (one-off vs. core system).

**Examples:**

* *Positive:* Small, independent PRs reviewed quickly.
* *Negative:* Adopting shiny new tech to “learn it” at the cost of delivery.

---

### **1.8 Better Today is Better Than Perfect Tomorrow**

> “The most effective way to cope with change is to increase the rate of change.” – Kent Beck

**Concept:** **Seek small, high-leverage improvements constantly** rather than waiting for the
opportunity to implement a flawless, large-scale solution. Optimizing for "perfect" often leads to
paralysis.

**Implications:**

* Focus daily on debt-reduction and clarity, even if only for a few lines of code.
* A single, clean commit to improve a function's name is more valuable than planning a month-long
  refactor.
* **Good enough now** with high quality and clean seams is better than **perfect later** that
  delays customer value.

**Examples:**

* *Positive:* Taking 30 minutes to improve logging traceability in a hot path instead of waiting
  for a full observability platform overhaul.
* *Negative:* Blocking a feature deployment because a related, non-critical service hasn't been
  re-architected.

---

### **1.9 A System Under Load is a Different System**

**Concept:** Systems behave fundamentally differently under **real-world traffic, concurrency, and
failure modes** than they do in development or synthetic stress tests. Production is the final,
most reliable testing environment.

**Implications:**

* **Measure everything** and watch for degradation (see **Optimization Trigger**).
* **Latency matters**—average latency hides critical tail latency, which is what customers
  experience during peak load.
* Never assume performance based on local machine tests; load test against realistic,
  production-like data and traffic.
* Failure modes (network partitions, slow dependencies) must be explicitly tested and handled.

**Examples:**

* *Positive:* Using a tool like **p99** (99th percentile) latency as the primary performance
  metric.
* *Negative:* Assuming a database query optimized for 10 rows will perform the same when
  retrieving 10,000 concurrent rows.

---

## **2. Seams for Replaceability**

> "A seam is a place where you can change behavior without editing the code." — Michael Feathers,
> Working Effectively with Legacy Code

**Concept:** A seam is an intentional boundary where behavior can be swapped, isolated, or tested
without touching the surrounding system.

Every part of a system is easy to build the first time but hard to change later. Seams are the
antidote: they give you safe "grip points" for refactoring, testing, or replacing components as
your understanding evolves (Gall's Law).

**The Golden Rule of Seams:** Seams are not abstractions-you-don't-need. A seam is valuable only if
it allows:

* Independent testing: Can I verify logic without spinning up a database?
* Independent replacement: Can I swap a vendor without rewriting business logic?
* Independent reasoning: Can I understand the core logic without knowing the implementation details
  of the dependency?

If it does none of these, it's not a seam—it's an unnecessary abstraction.

**Heuristics:**

* Isolate Volatility: A seam is only justified if the other side is risky, slow, or likely to
  change (e.g., third-party APIs, auth providers, network calls, randomness).
* Keep it Thin: Prefer simple interfaces and pure functions over tangled, stateful modules.
* Data over Classes: Pass data structures in/out of boundaries rather than cementing behavior into
  sprawling class hierarchies.
* Push Decisions to Edges: Move configuration and wiring (Env, DI, Factories) to the entry point of
  the app, keeping the core logic pure.

**Examples:**

✅ Positive: The Happy Path — Swapping a storage backend without touching business logic.

```python
from typing import Protocol

# 1. The Seam (Protocol)
# The business logic depends on this contract, not the implementation
class FileStore(Protocol):
    def save(self, key: str, data: bytes) -> None:
        ...

# 2. The Logic
class UploadService:
    def __init__(self, store: FileStore):
        self.store = store

    def upload(self, file_data: bytes):
        # Logic doesn't care if this is S3 or Disk
        self.store.save('profile.png', file_data)

# 3. The Test (in CI)
# No HTTP mocking. No S3 latency. No flakiness
def test_uploads_file():
    # Simple in-memory dict implementation
    class FakeStore:
        def __init__(self): self.saved = {}
        def save(self, key, data): self.saved[key] = data

    fake = FakeStore()
    service = UploadService(fake)

    service.upload(b'ok')

    assert 'profile.png' in fake.saved
```

❌ Negative: Premature Abstraction — Creating architecture for a future that hasn't arrived.

```python
class AbstractSmsManager(ABC, RetryMixin, Observable):
    """... 500 lines of extensible architecture ..."""
    pass
```

Why this fails: Nobody needed the abstraction. Nobody can reason about it. Nobody can change it
without fear.

Fix: Just write a simple send_sms(to, body) function.

---

## **3. The Threshold for Complexity: A Synthesis**

Complexity must always be earned. Introduce it only when the simpler approach demonstrably fails.

| Decision | Default | Trigger | Key Heuristic |
| --- | --- | --- | --- |
| **Abstraction** | Duplicate first | Third repetition | Prefer duplication over wrong abstraction |
| **Optimization** | Clarity first | Measured degradation | Don’t optimize what you can’t measure |
| **Scaling** | Simple infra | Proven load bottlenecks | Solve the problems you *have* |
| **Decoupling** | Modular monolith | Team friction | Microservices only when pain beats complexity |

---

## **4. Prescriptive Technical Defaults**

Defaults eliminate bikeshedding and ensure consistency. Deviations must be justified via ADR.

### **4.1 Data Integrity & Model Hygiene**

* **Primary Keys:** UUIDv7 (time-ordered, sortable, safe).
* **Timestamps:** TIMESTAMPTZ (UTC); auto-set `created_at` and `updated_at`.
* **Constraints:** Enforce with NOT NULL, FOREIGN KEY, CHECK, and explicit ON DELETE.
* **Enums:** Use Postgres ENUM for fixed sets.

Enums over string CHECKs: For any "fixed set of values" column, define a single canonical enum (Python Enum + DB enum) and map the column to it. Avoid hand-written string CHECK constraints and hardcoded string literals in code. This centralizes allowed values, improves type-safety, and makes invalid states harder to represent.

---

### **4.2 Reliability & Resilience**

* **Timeouts:** Always explicit; never infinite.
* **Retries:** Idempotent ops use exponential backoff + jitter.
* **Idempotency:** Critical POSTs accept `Idempotency-Key`.

---

### **4.3 API Hygiene**

* **Errors:** Use RFC 7807 (`application/problem+json`).
* **Time Format:** RFC 3339 UTC.
* **Pagination:** Cursor-based (not OFFSET).
* **Casing:** snake_case in DB, camelCase in JSON (convert at boundary).

---

## **5. Workflow Discipline**

### **5.1 Development Loop**

1. **Understand & Scope:** Define smallest complete outcome (Gall’s Law, BSSN).
2. **Spike if Unknown:** Explore, then delete (Spike Rule).
3. **Implement Cleanly:** Build the Best Simple System for Now.
4. **Validate:** Meet Quality Bar.
5. **Document as You Go:** Update docs in same PR.
6. **Log Defensively:** Use structured logs with correlation IDs.
7. **Commit/PR:** Keep tight scope (Optimize for Deletion).
8. **Manage Follow-ups:** Log deferred improvements separately.

---

### **5.2 PR & Quality Standards**

**Title:** Follow Conventional Commits (`feat|fix|refactor|docs|chore`).

**Body:**

* **Problem:** What issue is being solved?
* **Approach:** Why this method? (reference principles)
* **Validation:** How was it tested? Include output/screenshots.
* **Risks:** What to review carefully?
* **Policy Notes:** Mention RLS/security changes.
* **Follow-ups:** Link deferred tasks.

**Quality Bar:**

* ✅ Clean: No dead code, TODOs without tickets.
* ✅ Validated: Lint, type-check, 100% coverage for new logic.
* ✅ Structured: Cohesive modules; small functions (<25 lines).
* ✅ Data-Safe: Enforce integrity; avoid invalid states.
* ✅ Resilient: External calls have explicit timeouts + retry docs.
* ✅ Reviewable: Focused diffs (<250 LOC). Don’t mix refactors + features.

---

## **6. Communication Requirements (for Agents)**

* **Brevity:** Status > storytelling.
* **Context First:** Name file(s) + action (e.g., “Editing src/api/users.py to add POST”).
* **Focus:** Stay on assigned task; log deviations.
* **No Fluff:** No apologies or filler.
* **Transparency:** Show exact commands when using tools.

---

## **7. Appendix: Principle Summary Table**

| Principle              | Core Idea                    | Anti-Pattern              |
| ---------------------- | ---------------------------- | ------------------------- |
| Gall’s Law             | Evolve from simple systems   | Big-bang design           |
| Pragmatic Simplicity   | Build only what’s needed now | Speculative abstraction   |
| Mahoney’s Creed        | Data > Code                  | App-level-only validation |
| Invalid States         | Types enforce correctness    | Ambiguous flags           |
| Refactor When It Hurts | Abstract after repetition    | Premature abstraction     |
| Spike Rule             | Explore, then delete         | Shipping prototype code   |
| Pragmatic Velocity     | Speed via simplicity         | Resume-driven tech        |

---

**Rule 0 Reminder:** Make things easier to change. Everything in this document exists to serve
that.
