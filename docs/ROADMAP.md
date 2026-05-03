# gemma4good — Roadmap (PoC → MVP)

Time is short. This roadmap is ruthless about scope. Anything not on the critical path for the demo is deferred.

## Guiding cuts
- **Single conversation, single topic** at PoC. Multi-convo is MVP, not PoC.
- **Text-only**. No image/PDF upload until after MVP.
- **No web search, no graph/equation tool, no file read** until after MVP. They conflict with "offline" and aren't needed to prove the core loop.
- **No follow-up recommendations** until after MVP.
- **CLI + UI in parallel from M1.** CLI stays as the harness for fast iteration; UI grows alongside so the demo target is real from day one. M3 finishes the UI; it does not start it.
- **One Gemma 4 variant** picked early and frozen. Don't shop variants.
- **LangGraph introduced at M2, not before.** M1's loop is straight-line; LangGraph would be pure overhead. M2 adds verification + safety branching, retries, and shared state — that's where an explicit graph starts paying for itself, and it doubles as the boundary the M3 UI talks to.

## North-star demo
A 90-second clip:
1. User picks a topic.
2. Agent gives a structured overview.
3. Agent asks 3 MCQs.
4. User answers; one wrong.
5. Conversation continues; the wrong question reappears 2–3 turns later.
6. User answers correctly; it stops reappearing as often.

If that runs offline on Gemma 4, the PoC is done.

---

## M0 — Spike (smallest possible)
**Goal:** prove Gemma 4 runs locally end-to-end and can produce a structured MCQ.

- [x] uv-runnable smoke test (already landed: [scripts/test_gemma4.py](../scripts/test_gemma4.py)).
- [ ] Pick variant + quantization that fits target hardware. Freeze it.
- [ ] Prompt that returns an MCQ as **strict JSON** (`{question, choices[4], answer_index, rationale}`).
- [ ] Parse + validate JSON. Retry once on parse failure.

**Done when:** one shell command emits a valid MCQ JSON about a user-supplied topic.

## M1 — Core loop (PoC)
**Goal:** the north-star demo, in a CLI **and** a minimal UI.

Loop:
- [ ] Topic intake → overview generation (short, structured).
- [ ] Quiz generation: N MCQs from the overview, JSON-validated.
- [ ] Grading: exact index match.
- [ ] In-memory review queue: incorrect answers go in with high priority.
- [ ] **Turn-based interleaver**: every K user turns, pop one item from the queue and ask it; correct answer demotes priority, wrong promotes it.
- [ ] Persist queue + history to a single SQLite file so a session can resume.

UI (single-conversation shell):
- [ ] Pick the smallest workable stack and freeze it (FastAPI + single HTML page is the default — no deliberation).
- [ ] Chat pane: user input box, message list.
- [ ] MCQs render as clickable choices; selection sends an answer event.
- [ ] UI talks to the **same** loop the CLI drives — no parallel implementations.

**Done when:** the north-star demo runs both in a terminal and in the browser, offline.

**Explicitly out:** multi-conversation, materials upload, verification node, safety node, LangGraph.

## M2 — Verification + safety + LangGraph migration
**Goal:** stop the obvious failure modes that would tank a demo, and put the loop on rails so M3 has a clean API to build a UI against.

Graph + nodes:
- [ ] Migrate M1's loop to **LangGraph**. Nodes: `overview`, `quiz_gen`, `verify`, `safety`, `grade`, `review_inject`. Shared state object replaces the ad-hoc dict from M1.
- [ ] Verification node: re-prompt Gemma to check `answer_index` is consistent with `question + choices`. Drop items that fail.
- [ ] Safety node: a single classifier prompt that rejects unsafe topics/questions before they reach the user.
- [ ] Conditional edges: failed verify → retry once, then drop; failed safety → short-circuit to a polite refusal.
- [ ] Logging: every node's input/output to a JSONL file for debugging.

UI surfaces:
- [ ] Show a small status indicator when an item is dropped by verify/safety (so the user understands why a topic was refused or a question regenerated).
- [ ] Persist conversation+queue to SQLite so reload restores state in the UI, not just the CLI.

**Done when:** running the demo on a list of 20 varied topics produces zero malformed quizzes and zero unsafe outputs, and the UI faithfully reflects verify/safety outcomes.

## M3 — UI completion (MVP)
**Goal:** finish the UI on top of the M2 graph. Everything user-visible lands here.

- [ ] Data model finalized: `Conversation`, `Message`, `QuizItem`, `ReviewState`. SQLite (already on disk from M2; this milestone hardens the schema).
- [ ] Conversation picker (sidebar list, "new conversation" button).
- [ ] Conversation switcher preserves per-conversation state without losing in-flight quizzes.
- [ ] Cross-conversation review pool: incorrect items from any conversation can re-surface in any conversation; UI shows which conversation a re-surfaced item originated from.
- [ ] Empty-state, loading, and error states for the chat pane and the picker.
- [ ] Visual polish pass: keyboard nav for MCQ choices, responsive layout, no console errors.

**Done when:** a non-developer can open the app, run two topics, switch between them, and see review questions interleave — including review items that crossed conversations.

## M4 — Submission polish
- [ ] README with run instructions (uv-only).
- [ ] 90-second screen recording.
- [ ] Short writeup (problem, approach, agent graph diagram, limitations).
- [ ] Public repo cleanup (license already present, add `.gitignore`, prune dead code).

**Done when:** submission link works from a clean machine following only the README.

---

## Deferred to post-submission (do not start until M4 ships)
- Multimodal materials upload (image/PDF → overview/quiz).
- Tooling: web search (cached corpus), file read, graph/equation generator.
- Spaced-repetition algorithm upgrade (SM-2 / FSRS) — until then, simple priority queue with cooldown counter is enough.
- Follow-up topic recommendations.
- Adaptation tuning beyond "wrong-answer priority."

## Risks (and the cheap mitigation)
- **Variant doesn't fit hardware** → fall back to smaller variant or 4-bit; decided in M0, not M3.
- **JSON parsing flakiness** → strict schema + one retry + verification node; do not invent a parser DSL.
- **Scope creep into agent frameworks** → LangGraph is scheduled for M2 only, when verify/safety branching makes plain functions painful. No LlamaIndex, no other framework, no earlier adoption. If M1 tempts you to add it for "cleanliness," resist.
- **LangGraph migration overruns its budget** → if the M2 migration isn't passing the M1 demo within a short timebox, fall back to plain functions and ship M2's verify/safety nodes without the graph. Demo > framework.
- **Offline + web search conflict** → web search is deferred; do not re-litigate.
