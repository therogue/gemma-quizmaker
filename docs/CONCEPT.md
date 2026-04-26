# gemma4good — Concept

## One-liner
Offline, single-user learning agent built on Gemma 4: explains topics, generates quizzes, and uses turn-based spaced repetition to reinforce retention across conversations.

## Competition framing
Track: **Future of Education**. Brief: *"multi-tool agents that adapt to the individual and empower the educator through seamless integration."*

This project targets the **adapt-to-the-individual** half. Educator-facing features are out of scope.

## Goals
User can:
- Ask about a topic and get a structured overview.
- Receive a short MCQ quiz to reinforce the overview.
- Drill deeper in the same conversation.
- Maintain multiple conversations, one per topic.
- Receive turn-based interleaved review questions:
  - From old quizzes in the *same* conversation (spaced repetition).
  - From any past quiz where answers were incorrect (priority replay).
- Upload own materials (files, images) — agent parses and produces overview + quiz.
- Receive follow-up topic recommendations.

## Adaptation signal
- Quiz correctness per question, per topic.
- Incorrect answers re-enter the review pool with higher priority.
- Pace and depth of follow-up questions adjusted from running accuracy.

## Non-goals
- Educator dashboards, multi-user, classrooms.
- Auth / accounts (local single-user).
- Cloud deploy, online-only features.
- Free-text grading.

## Deliverables
- Local app, fully offline at inference time.
- Backend + local DB (conversations, quizzes, review state, materials).
- Chat UI with conversation picker.
- Tooling surface: web search, file read, graph/equation generator.
- Public repo + writeup for submission.

## Agent shape (high level)
Multi-tool agent with explicit nodes:
- **Planner / router** — decides next action (explain, quiz, review, tool call, recommend).
- **Retrieval** — over uploaded materials and past conversation context.
- **Tool nodes** — web search, file read, graph/equation generator.
- **Quiz generator** — MCQ, exact-match grading.
- **Verification node** — hallucination guard before content reaches user.
- **Safety node** — content safety check, separate from verification.
- **Review scheduler** — turn-based, injects spaced + incorrect-answer questions.

Detailed graph design: TBD in design doc.

## Quiz spec
- Format: multiple choice.
- Grading: exact match.
- Hallucination guard: verification node before delivery.

## Spaced repetition
- Trigger: turn-based interleaving inside an active conversation.
- Pool: prior quiz items from same topic + globally-incorrect items.
- Algorithm: TBD (SM-2 / FSRS / Leitner candidates).

## Model
- **Gemma 4** (competition requirement).
- Variant: TBD — depends on local hardware budget vs. multimodal needs (image/PDF ingestion implies multimodal variant).

## Stack
- TBD. Constraints: offline, local DB, runs Gemma 4 locally.

## Open questions
- Spaced-repetition algorithm choice.
- Gemma 4 variant + quantization for target hardware.
- Storage schema for review state.
- Tool provider choices (web search offline-capable? cached corpus?).

## Success criteria
- User self-report (no objective retention benchmark planned).
