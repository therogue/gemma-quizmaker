# M3 Backend Contract

This is the current FastAPI contract for the multi-conversation MVP.

## Conversations

Create an empty conversation:

```http
POST /conversations
```

List conversations:

```http
GET /conversations
```

Load one conversation:

```http
GET /conversations/{id}
```

The detail response includes conversation metadata, persisted transcript messages,
the current overview, and active questions.

## Start Topic

```http
POST /conversations/{id}/start-topic
```

Request:

```json
{
  "topic": "photosynthesis",
  "quiz_count": 3
}
```

Behavior:

- Updates the conversation's current focus topic and overview.
- Preserves prior messages and quiz items in the same conversation.
- Counts as one turn.
- Does not auto-generate topic suggestions.
- Stores quiz question transcript messages without `answer_index` or `rationale`.

Question response shape:

```json
{
  "item_id": 1,
  "is_review": false,
  "topic": "photosynthesis",
  "question": "What is the primary energy source used by plants?",
  "choices": ["A", "B", "C", "D"]
}
```

## Answer Question

```http
POST /conversations/{id}/answer
```

Request:

```json
{
  "item_id": 1,
  "choice_index": 2
}
```

Behavior:

- Rejects quiz items from other conversations.
- Counts as one turn.
- Wrong answers are promoted into the per-conversation review queue.
- The answer message stores `correct_index` and `rationale` only after the user answers.
- May return a review question if the turn crosses the review interval.

Response:

```json
{
  "item_id": 1,
  "is_correct": true,
  "correct_index": 2,
  "rationale": "Photosynthesis converts light energy into chemical energy.",
  "review": null
}
```

## Advance Turn

```http
POST /conversations/{id}/turn
```

Manual turn advancement. Counts as one turn and may return a review question.

## Chat

```http
POST /conversations/{id}/chat
```

Request:

```json
{
  "text": "Can you explain the Calvin cycle?"
}
```

Behavior:

- Plain chat only; it does not secretly generate quizzes.
- Stores user and assistant messages in `messages`.
- Counts as one turn.
- Sends Gemma the current focus topic, current overview, and last 10 messages.
- May return a review question if the turn crosses the review interval.

## Actions

```http
POST /conversations/{id}/actions
```

More questions:

```json
{
  "action": "more_questions",
  "count": 3
}
```

Generates additional questions from the conversation's current focus topic and
overview.

Suggest topics:

```json
{
  "action": "suggest_topics",
  "count": 4
}
```

Suggestions are explicit and on-demand only.

## Messages vs Logs

- `messages`: user-visible transcript source of truth.
- `logs`: internal/debug events, including persisted graph node records.
