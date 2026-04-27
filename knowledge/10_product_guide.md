# Product Guide

This guide explains the project from a product manager's point of view. For the engineering-facing view, see [09_developer_guide.md](./09_developer_guide.md).

## Executive Summary

Layered Memory Architecture (LMA) is a research prototype for a more stateful kind of AI assistant.

Its thesis is simple:

> A good assistant should not treat every user message as an isolated search query.

Instead, it should:

- recognize what kind of request it is
- decide whether retrieval is even necessary
- remember stable user facts
- maintain conversational state
- separate topics into threads
- surface prior decisions and insights when they become relevant again

From a product perspective, this is not just "RAG chat." It is an attempt to build an inspectable memory system for conversational AI.

## What The Product Is Today

Based on the current code, docs, and UI, the product already supports:

- a chat experience with streaming responses
- persistent conversations
- profile memory and profile recall
- privacy and "what do you know about me?" flows
- topic threading within a conversation
- research-insight extraction and concept linking
- detailed observability of why the system responded the way it did
- experiment toggles and telemetry for subsystem evaluation

The UI makes this visible through:

- conversation history in the sidebar
- a profile and memory modal
- a thread side panel
- a research dashboard
- per-message retrieval details
- optional debug mode

## What The Product Is Not

The current implementation is not yet:

- a consumer-ready general assistant
- a multi-tenant SaaS with authentication and permissions
- a polished enterprise knowledge platform
- a proven memory system with strong real-world validation

It is best understood as a strong research prototype and reference architecture.

## Inferred Product Framing

The following framing is inferred from the README, codebase, UI copy, and API surface. It is not a formal product spec written elsewhere in the repo.

### Most likely category

An AI knowledge workspace for long-running technical conversations.

### Most likely early users

- AI engineers building stateful assistants
- teams experimenting with long-horizon memory for copilots
- internal knowledge assistant builders
- researchers evaluating whether structured memory beats plain RAG

### Most likely wedge

"Use this when normal RAG feels stateless, noisy, or impossible to debug."

## The User Problems It Is Trying To Solve

## 1. Standard RAG feels forgetful

Normal RAG systems often treat each message as an isolated retrieval event. Follow-ups, prior decisions, and long-running topics get lost.

LMA's answer:

- continuation gating
- curated history
- thread tracking
- cross-turn Q&A recall

## 2. Standard chat memory is opaque

Many assistants "seem" to remember, but users and builders cannot inspect what was stored or why it was retrieved.

LMA's answer:

- explicit profile storage
- explicit research insights
- explicit concept links
- visible retrieval panels and debug mode

## 3. Retrieval is often overused

Greeting messages, lightweight follow-ups, or testing/probing queries do not always need retrieval. Standard systems still pay the cost.

LMA's answer:

- classifier before retrieval
- behavior engine retrieval skipping
- policy-based context injection

## 4. Product teams cannot tell which subsystem matters

Many AI products accumulate complexity without proof that it improves outcomes.

LMA's answer:

- telemetry by stage
- runtime subsystem toggles
- baseline mode
- A/B experiment harness

## The Product's Core Experience Pillars

These are the clearest product pillars visible in the repo.

| Pillar | User-facing meaning | Main implementation |
|---|---|---|
| Selective memory | the system remembers only what should matter | intent classifier, policy engine |
| Conversational continuity | follow-ups feel connected | curated history, same-conversation Q&A |
| Structured topic awareness | conversations can branch without collapsing context | topic threading |
| Personal memory | user facts can be remembered and recalled | profile memory |
| Research recall | decisions and insights can resurface later | research memory |
| Transparency | users and builders can inspect what happened | telemetry, debug UI, retrieval panels |

## Product Walkthrough By User Intent

The intent taxonomy is one of the best ways to understand the product behavior.

### `general`

Examples:

- greetings
- casual conversation
- broad questions

Product behavior:

- minimal or adaptive retrieval
- fast, low-friction response

### `knowledge_base`

Examples:

- technical or factual questions
- documentation lookups
- architecture comparisons

Product behavior:

- retrieve knowledge base chunks
- optionally add prior related Q&A
- answer from internal context plus model knowledge

### `continuation`

Examples:

- "why?"
- "what about the second option?"
- "can you elaborate on that?"

Product behavior:

- use recent and semantically relevant earlier messages
- try to preserve local continuity
- avoid false continuation if topic drift is detected

### `profile`

Examples:

- "My name is Alex"
- "I prefer Python"
- "What do you remember about my background?"

Product behavior:

- save personal facts when shared
- inject stored profile when queried

### `privacy`

Examples:

- "What data do you store about me?"
- "Delete my data"

Product behavior:

- answer transparently
- expose stored profile information
- frame the response explicitly as a privacy interaction

This intent is especially product-important because it turns memory from a hidden capability into an inspectable contract with the user.

## Why The Product Is Differentiated

Compared with plain chat + RAG systems, the differentiators are:

- retrieval is policy-bound instead of unconditional
- memory is split into layers instead of a single transcript
- topic continuity is explicit
- behavioral adaptation exists as a separate subsystem
- debugging and measurement are first-class

In plain language:

Most RAG systems are answer engines. This project is trying to be a conversational cognition engine.

## The Current Product Strengths

## 1. Strong architecture narrative

The repo has a clear thesis, not just a pile of features. That makes it easier to explain, test, and position.

## 2. Excellent inspectability

This is one of the strongest product advantages in the codebase. Builders can see:

- intent
- route
- retrieval details
- policy decision
- thread resolution
- research context

That is valuable for trust, debugging, demos, and enterprise buyers.

## 3. Good "system thinking" for multi-turn use cases

The combination of:

- behavior engine
- history curation
- threading
- structured memory

is well aligned with long-running technical conversations.

## 4. Built-in experimentation culture

The repo is unusually honest about proving value instead of assuming it. That is rare and product-healthy.

## The Current Product Weaknesses

## 1. It is still more prototype than platform

There is no auth layer, no workspace model, no org model, no permissions, and no production-grade background job system.

## 2. Memory value is uneven across subsystems

The product promise is strongest for:

- conversational continuity
- inspectability
- thread coherence

It is weaker, today, for:

- research memory ROI
- hybrid retrieval justification on small corpora

## 3. Product surface is broader than validated value

The system exposes:

- threads
- insights
- concepts
- debug panels
- retrieval dashboards

But the experiments suggest not every one of those layers is yet earning its cost.

## 4. Terminology may be too research-heavy for end users

Phrases like:

- research memory
- concept links
- thread cohesion
- behavioral routing

work well for technical audiences, but would likely need simplification for broader product messaging.

## What The Evidence Says Right Now

The repo already includes published findings in [README.md](../README.md). That matters because this product is explicitly evidence-driven.

### Positive signals

- behavior adaptation triggered on a meaningful minority of queries
- thread attachment rate was high in the reported experiment
- tail latency improved in the full pipeline because some queries skipped retrieval
- the system can demonstrate continuity and inspectability better than plain RAG

### Negative or cautionary signals

- research memory hit rate was effectively zero in the cited run
- hybrid search did not outperform vector baseline on the current small corpus
- reranking added latency without earning it in retrieval-only evaluation
- rapid-fire and repetition detectors under-activated in the published tests

### Product implication

The product already has a compelling story, but the best current evidence supports a narrower claim than "all advanced memory layers improve quality."

A more honest current claim would be:

> The architecture clearly improves inspectability and likely helps multi-turn continuity, but some advanced retrieval and research-memory layers still need stronger validation.

## Current Product Positioning Recommendation

If this were being pitched today, I would position it as:

### Primary positioning

A reference architecture for stateful AI assistants with inspectable memory.

### Secondary positioning

A research workbench for measuring which memory and retrieval layers actually improve conversations.

### I would not position it as

- "the best general chatbot"
- "enterprise-ready memory platform"
- "drop-in production memory layer" without qualification

## Product Risks And Launch Blockers

If this were moving toward broader release, these are the main PM risks.

### 1. Trust and privacy expectations

Memory products create strong user expectations. The current product does have privacy flows, which is good, but it lacks:

- authentication
- tenant separation
- deletion workflows beyond basic profile removal
- governance features

### 2. Complexity creep

The architecture can become harder to explain than the value it creates. That is especially risky if some layers are not measurably useful.

### 3. Small-corpus mismatch

The retrieval stack is more sophisticated than the current corpus size appears to justify. That can make the product feel over-engineered until the knowledge base is much larger.

### 4. Discoverability of value

A user may see "chat + dashboard" but not immediately understand why threads, concepts, and insights help them. Onboarding and explanation matter.

## Suggested Product Metrics

If I were PM-ing this project, I would track four classes of metrics.

### Experience metrics

- response latency
- P95 latency
- conversation length
- return usage across sessions

### Quality metrics

- off-topic injection rate
- thread attachment quality
- retrieval precision and recall
- answer faithfulness and relevance

### Memory metrics

- profile recall usefulness
- research insight hit rate
- concept-link reuse rate
- number of conversations where memory changed the answer

### Trust metrics

- privacy query success rate
- deletion success rate
- user-reported trust in memory behavior
- debug/inspection panel usage by builders

## What I Would Prioritize Next

This prioritization is based on both the current product surface and the experiment results.

### Tier 1: strengthen the core claim

1. Improve and validate the multi-turn continuity story.
2. Tighten thread quality and reduce fragmentation.
3. Prove when research memory helps, or simplify it if it does not.

### Tier 2: harden product fundamentals

1. Add authentication and user identity handling.
2. Add clear data deletion and memory management flows.
3. Add a more robust job system for post-response processing.

### Tier 3: expand only after proof

1. file upload ingestion
2. web search as a policy-routed source
3. tool calling and action execution
4. richer exports of thread summaries and insights

## A Good PM Reading Of The Roadmap

The roadmap in [README.md](../README.md) shows a healthy direction:

- measure first
- keep only what earns its complexity
- expand ingestion and tools later

The strongest PM principle already present in this repo is:

> Do not confuse architectural sophistication with product value.

That principle should stay central.

## Messaging Drafts

If you need short positioning language, here are grounded options based on the current product.

### One-line version

An inspectable memory architecture for AI assistants that need more than plain RAG.

### Demo-day version

This system shows not only what an AI answered, but why it answered that way, what it remembered, what it retrieved, and which conversation thread it thinks you are in.

### Technical buyer version

LMA is a reference implementation for policy-gated retrieval, structured conversational memory, and subsystem-level AI evaluation on top of FastAPI, React, PostgreSQL, and pgvector.

## Final Product Read

The clearest product truth in this repo is not "we built a smarter chatbot."

It is this:

> We built a system for testing whether explicit memory architecture can make AI conversations more coherent, more controllable, and more debuggable than standard RAG.

That is a strong product thesis.

What remains is narrowing the claim to the parts the evidence already supports, then either proving or pruning the layers that do not yet earn their keep.
