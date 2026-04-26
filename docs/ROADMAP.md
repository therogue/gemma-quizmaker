# gemma4good — Roadmap (PoC → MVP)

Time is short. This roadmap is ruthless about scope. Anything not on the critical path for the demo is deferred.

## Guiding cuts
- **Single conversation, single topic** at PoC. Multi-convo is MVP, not PoC.
- **Text-only**. No image/PDF upload until after MVP.
- **No web search, no graph/equation tool, no file read** until after MVP. They conflict with "offline" and aren't needed to prove the core loop.
- **No follow-up recommendations** until after MVP.
- **CLI first**, UI second. UI is presentation; the loop is the product.
- **One Gemma 4 variant** picked early and frozen. Don't shop variants.

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
**Goal:** the north-star demo, in a CLI.

- [ ] Topic intake → overview generation (short, structured).
- [ ] Quiz generation: N MCQs from the overview, JSON-validated.
- [ ] Grading: exact index match.
- [ ] In-memory review queue: incorrect answers go in with high priority.
- [ ] **Turn-based interleaver**: every K user turns, pop one item from the queue and ask it; correct answer demotes priority, wrong promotes it.
- [ ] Persist queue + history to a single SQLite file so a session can resume.

**Done when:** the north-star demo runs in a terminal, offline.

**Explicitly out:** multi-conversation, UI, materials upload, verification node, safety node.

## M2 — Verification + safety (still PoC-grade)
**Goal:** stop the obvious failure modes that would tank a demo.

- [ ] Verification node: re-prompt Gemma to check `answer_index` is consistent with `question + choices`. Drop items that fail.
- [ ] Safety node: a single classifier prompt that rejects unsafe topics/questions before they reach the user.
- [ ] Logging: every node's input/output to a JSONL file for debugging.

**Done when:** running the demo on a list of 20 varied topics produces zero malformed quizzes and zero unsafe outputs.

## M3 — Multi-conversation + minimal UI (MVP)
**Goal:** the thing the judges actually click on.

- [ ] Data model: `Conversation`, `Message`, `QuizItem`, `ReviewState`. SQLite.
- [ ] Conversation picker (sidebar list, "new conversation" button).
- [ ] Chat pane that renders MCQs as clickable choices.
- [ ] Cross-conversation review pool: incorrect items from any conversation can re-surface in any conversation.
- [ ] Pick framework: smallest thing that works (FastAPI + a single HTML page, or a Tauri/Electron shell — pick one, don't deliberate).

**Done when:** a non-developer can open the app, run two topics, switch between them, and see review questions interleave.

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
- **Scope creep into agent frameworks** → no LangGraph/LlamaIndex unless something in M1 actually breaks without it. Plain Python functions until proven insufficient.
- **Offline + web search conflict** → web search is deferred; do not re-litigate.
