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
- [x] Pick variant + quantization that fits target hardware. Freeze it.
- [x] Prompt that returns an MCQ as **strict JSON** (`{question, choices[4], answer_index, rationale}`).
- [x] Parse + validate JSON. Retry once on parse failure.

**Done when:** one shell command emits a valid MCQ JSON about a user-supplied topic.

## M1 — Core loop (PoC)
**Goal:** the north-star demo, in a CLI **and** a minimal UI.

Loop:
- [x] Topic intake → overview generation (short, structured).
- [x] Quiz generation: N MCQs from the overview, JSON-validated.
- [x] Grading: exact index match.
- [x] In-memory review queue: incorrect answers go in with high priority.
- [x] **Turn-based interleaver**: every K user turns, pop one item from the queue and ask it; correct answer demotes priority, wrong promotes it.
- [x] Persist queue + history to a single SQLite file so a session can resume.

UI (single-conversation shell):
- [x] Pick the smallest workable stack and freeze it (FastAPI + single HTML page is the default — no deliberation).
- [x] Chat pane: user input box, message list.
- [x] MCQs render as clickable choices; selection sends an answer event.
- [x] UI talks to the **same** loop the CLI drives — no parallel implementations.

**Done when:** the north-star demo runs both in a terminal and in the browser, offline.

**Explicitly out:** multi-conversation, materials upload, verification node, safety node, LangGraph.

## M2 — Verification + safety + LangGraph migration
**Goal:** stop the obvious failure modes that would tank a demo, and put the loop on rails so M3 has a clean API to build a UI against.

Graph + nodes:
- [x] Migrate M1's loop to **LangGraph**. Nodes: `overview`, `quiz_gen`, `verify`, `safety`, `grade`, `review_inject`. Shared state object replaces the ad-hoc dict from M1.
- [ ] Verification node: re-prompt Gemma to check `answer_index` is consistent with `question + choices`. Drop items that fail.
- [ ] Safety node: a single classifier prompt that rejects unsafe topics/questions before they reach the user.
- [ ] Conditional edges: failed verify → retry once, then drop; failed safety → short-circuit to a polite refusal.
- [x] Logging: every node's input/output to a JSONL file for debugging.

UI surfaces:
- [ ] Show a small status indicator when an item is dropped by verify/safety (so the user understands why a topic was refused or a question regenerated).
- [x] Persist conversation+queue to SQLite so reload restores state in the UI, not just the CLI.

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

## Ideas for future development

- **Bullet-targeted MCQ generation**: instead of generating questions from the topic string, assign each MCQ to a specific bullet point from the overview. Eliminates duplicate/similar questions and ensures full coverage of the overview. Partially addressed by the structured overview format; the next step is passing the target bullet directly to the quiz-gen prompt.
- **Bullets as sub-topics**: use each overview bullet point as the seed topic for the *next round* of MCQs. Lets the user drill progressively deeper on whichever fact they got wrong.
- **Stop button for generation**: a cancel button that aborts the in-flight request while Gemma is generating. Frontend-side abort is trivial via `AbortController`; true server-side cancellation requires streaming responses (SSE or chunked transfer) so the server knows when the client disconnects and can interrupt the model mid-generation.

## Deferred to post-submission (do not start until M4 ships)
- Multimodal materials upload (image/PDF → overview/quiz).
- Tooling: web search (cached corpus), file read, graph/equation generator.
- Spaced-repetition algorithm upgrade (SM-2 / FSRS) — until then, simple priority queue with cooldown counter is enough.
- Follow-up topic recommendations.
- Adaptation tuning beyond "wrong-answer priority."

## Generation quality concerns (not yet addressed)

These are known output-quality failure modes observed during M1 development. None are blocking the PoC demo, but they should be tackled before MVP or they will degrade the learning experience.

**Content diversity**
- **Duplicate or near-duplicate questions**: Gemma tends to generate questions about the same obvious fact. Mitigation candidate: assign each MCQ to a specific overview bullet (see Ideas section); M2 verify node could also reject questions too similar to existing ones.
- **Redundant overview bullets**: the overview can repeat the same concept in slightly different wording. Mitigation candidate: post-generation deduplication pass, or prompt engineering to force distinct aspects per bullet.

**Answer choice quality**
- **Multiple choices that are arguably correct**: Gemma sometimes generates distractors that are also defensible answers. The M2 verify node should check this explicitly — re-prompt Gemma to confirm only one choice is unambiguously correct.
- **Similar or non-discriminating choices**: all four choices may be semantically similar, making the question trivial. Verify node candidate.
- **Gemma labelling choices with A/B/C/D**: model sometimes prefixes choice text with its own letter labels; prompt mitigation in place but not fully reliable.

**Lifecycle and data hygiene**
- **No way to discard bad generated content**: if a question or overview is clearly wrong, there is no admin tool to delete or flag it. Needed before multi-user or persistent-session use.
- **SQLite accumulates stale sessions**: switching topics leaves orphaned quiz items with no cleanup path. A `reset` or `clear-session` endpoint/CLI command is needed.
- **Overview/question versioning**: re-generating a topic overwrites the old session with no history. If the new generation is worse, the old one is lost.

## Risks (and the cheap mitigation)
- **Variant doesn't fit hardware** → fall back to smaller variant or 4-bit; decided in M0, not M3.
- **JSON parsing flakiness** → strict schema + one retry + verification node; do not invent a parser DSL.
- **Generation quality degrades on niche topics** → log all raw model outputs (already in place for MCQ and overview); review logs to identify prompt improvements before M2.
- **Scope creep into agent frameworks** → LangGraph is scheduled for M2 only, when verify/safety branching makes plain functions painful. No LlamaIndex, no other framework, no earlier adoption. If M1 tempts you to add it for "cleanliness," resist.
- **LangGraph migration overruns its budget** → if the M2 migration isn't passing the M1 demo within a short timebox, fall back to plain functions and ship M2's verify/safety nodes without the graph. Demo > framework.
- **Offline + web search conflict** → web search is deferred; do not re-litigate.
