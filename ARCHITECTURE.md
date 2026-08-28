# Recoup — Architecture

A bounded revenue-recovery agent for Razorpay merchants.
Razorpay AI Buildathon 2026 · Track 3, AI Revenue Recovery.

---

## 1. The problem, stated precisely

A merchant on Razorpay loses revenue in three ways that look different but are
the same shape: a payment is attempted and fails, a checkout is started and
abandoned, an invoice is issued and goes past due. In each case the customer
wanted to buy, the merchant wanted to sell, and the money did not move.

The naive response is to chase all of it. That fails for a reason worth stating
plainly: **a large fraction of that revenue arrives anyway.** The bank outage
ends, the customer retries, the invoice gets paid on the 5th. A recovery system
that emails everyone and counts what comes back will report a large number, and
most of that number will be revenue it had nothing to do with.

So the real problem is not *how do I recover more?* It is:

1. **Which** at-risk revenue is actually recoverable by acting?
2. **What** action, specifically, for this failure?
3. **How much** of what came back can I honestly claim?
4. **What is the system allowed to do** while it finds out?

Recoup is organised around those four questions, in that order.

---

## 2. The two claims this architecture exists to support

Everything below is downstream of two decisions. If you read nothing else, read
this section.

### 2.1 Incremental measurement, via a randomised holdout

30% of events are randomly assigned to a **control arm** at generation time.
Control events are scored and decided *exactly* as treatment events are — the
full reasoning chain is computed and persisted — and execution is suppressed at
the final step.

This is deliberately more expensive than an untouched control group. It means
that for every held-out event we know precisely **what Recoup would have done**,
not merely that it did nothing. The comparison is against an inspectable
counterfactual rather than a population we hope is comparable.

The headline number is the **difference between arms**. Gross recovery is
reported beside it, explicitly labelled as the number a system without a holdout
would have claimed, so the gap between the flattering figure and the honest one
is visible rather than hidden.

Cost: ~₹11.7L of at-risk value goes untouched in the reference dataset. At 600
events a 10% holdout would leave the lift estimate too noisy to support any
claim, and a system that cannot measure itself is not one worth deploying.

### 2.2 Bounded autonomy, enforced in code rather than in a prompt

The agent proposes. A deterministic **policy engine** disposes.

Thirteen bounds — plus an input-validation gate that fails closed — run *after*
the model has spoken, on every decision, with no short-circuiting. The model has no way to reach around them, because they are
not instructions it was given — they are code that runs on its output.

This matters more than it first appears. Prompt-level guardrails fail
*silently*: a model told "never discount more than 15%" will mostly comply, and
the one time it doesn't, nothing catches it and nothing records it. A rule that
runs afterward catches it every time and writes down that it caught it.

Every check is persisted — **the passes as well as the failures**. "This action
was permitted because these thirteen bounds were checked and cleared" is a claim
the audit trail must be able to support, and it cannot be made from a list of
failures alone.

---

## 3. System shape

```
                    ┌─────────────────┐
                    │  taxonomy.py    │  16 reason codes → strategy,
                    │  (domain core)  │  attempt ceiling, wait, incentive
                    └────────┬────────┘  eligibility
                             │ every stage reads from here
   ┌─────────────────────────┼─────────────────────────────────────┐
   ▼                         ▼                                     ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌────────────┐
│ detect/  │→ │  agent/  │→ │ policy/  │→ │ execute/  │→ │   eval/    │
│  ASSESS  │  │  DECIDE  │  │  REVIEW  │  │    ACT    │  │  MEASURE   │
└──────────┘  └──────────┘  └──────────┘  └───────────┘  └────────────┘
     │             │             │              │              │
     ▼             ▼             ▼              ▼              ▼
 Assessment    Decision    PolicyReview     ActionRun       Outcome
     └─────────────┴─────────────┴──────────────┴──────────────┘
                              │
                    one immutable row per stage
                              │
                              ▼
                     ┌─────────────────┐
                     │   api/ + web/   │  replay any event end to end
                     └─────────────────┘
```

`pipeline.py` is the only place these meet. It is deliberately thin — each stage
is tested in isolation, so what the orchestrator must get right is the *order*
and the *timing*.

---

## 4. The domain core: `taxonomy.py`

The single most load-bearing file, and the one that separates this from a
generic "LLM reads a CSV" project.

**A failed payment is not one thing.** An issuer outage and a fraud decline both
surface as "payment failed", but one should be retried untouched in two hours
and the other must never be retried at all. Systems that treat them alike burn
money and — worse — push risk-declined transactions back through the rails,
earning the merchant a higher decline rate and a chargeback problem.

Sixteen reason codes, mapped against Razorpay's `error_source` / `error_step`
schema, each carrying:

| Field | Why it exists |
|---|---|
| `strategy` | Retry now / retry delayed / retry on liquidity / switch rail / persuade / **do not retry** |
| `base_recoverability` | Prior for the scorer, superseded by fitted rates once outcomes exist |
| `retry_after_minutes` | Acting earlier wastes the attempt — the balance hasn't changed, the outage hasn't ended |
| `max_attempts` | Hard per-reason ceiling |
| `incentive_eligible` | Whether spending money could plausibly change the outcome |
| `switch_to` | Which rails are worth offering instead |

**Only 2 of 16 reason codes permit spending money** — `payment_cancelled` and
`checkout_abandoned`. That constraint is derived from the domain, not from a
budget setting: a discount can only move a failure whose cause was *intent*.
Discounting a bank outage is pure margin burn — you paid a customer to do what
they were already going to do.

Unrecognised codes fail closed: no action, surfaced for a human, counted as a
coverage metric rather than quietly dropped.

---

## 5. Assess — `detect/`

Deterministic. No model, no randomness, no network.

`features.py` extracts observable signals only. The interesting one:

**`liquidity_window()`** — salaried accounts in India are credited around the
last working day of the month, and the balance survives into the first few days
of the next. So the recoverable window for an `insufficient_funds` failure is
roughly the 1st–5th, and a retry on the 22nd is close to guaranteed waste. Not
because the strategy is wrong — because the money isn't there yet. Getting this
right is most of the value in that bucket.

`scorer.py` combines the taxonomy prior with customer history, ticket size,
attempt number and time decay to produce P(recovered | we act) and an expected
value. It also carries `fit_from_outcomes()`, which estimates per-reason rates
from *observed treatment-arm outcomes* and shrinks them toward the prior in
proportion to how little evidence exists (25 pseudo-observations). That stops a
bucket with four events and three lucky recoveries from declaring itself a 75%
opportunity.

Every score records which basis produced it — `prior` or `fitted:n` — and that
label is surfaced in the UI.

---

## 6. Decide — `agent/`

**The LLM is deliberately not in the hot path.**

`router.py` escalates to the model only where the deterministic path is
genuinely underdetermined: incentive-eligible reasons (how deep to discount is a
judgement call), high exposure where a wrong call is costly, conflicting signals,
and repeat attempts where the obvious action already failed. Everything else is
settled by `rules_engine.py` straight from the taxonomy.

Calling a model 600 times to be told what a lookup table already knew is
expensive theatre. Knowing when *not* to call a model is part of the design.

`brain.py` uses `claude-opus-5` with adaptive thinking and **structured output**,
so a decision is machine-valid rather than parsed out of prose. What comes back
is validated anyway — unknown action types rejected, incentives clamped,
ineligible incentives dropped. The policy engine is the real gate; this just
keeps malformed proposals out of the audit trail.

`prompts.py` renders the bounds **from `policy.rules.Bounds` directly** rather
than restating them. A hardcoded copy would drift out of sync with the real
rules, and the failure mode is silent: the model would propose inside limits
that no longer exist.

**Degradation:** with no API key, `brain` falls back to the rules engine and
records the decision as `RULES`, not `LLM`. Counting a fallback as an LLM
decision would inflate exactly the number used to argue the model is used
sparingly.

### 6.1 The model is a component, not the architecture

`providers.py` carries two adapters: the Anthropic SDK, and any
OpenAI-compatible `/chat/completions` endpoint. The second covers OpenAI, Groq,
Google Gemini, OpenRouter, Together, DeepSeek and a local Ollama, because they
all speak that protocol — so switching is two lines of `.env`, not a rewrite.

Structured-output support is uneven in exactly the place it matters. Full
`json_schema` works on some endpoints, `json_object` on more, and an unsupported
value is usually a 400 rather than a graceful degradation. The adapter
negotiates downward: `json_schema` → `json_object` → plain text with the schema
restated in the prompt. That only changes how often the model gets it right
first time, never whether a bad proposal can get through — **the floor is the
validator, not the response format.**

Replies wrapped in ```` ```json ```` fences are unwrapped, because the endpoints
without schema support are precisely the ones that fence. A fenced body that is
not JSON is still rejected.

**The policy engine never learns which provider answered.** That is the useful
form of "the model proposes, the policy engine disposes": it means the answer to
*what if the model is bad?* is structural rather than reassuring. The claim is
tested — the same event decided through two providers must produce an identical
validated proposal, and if that ever fails the model has stopped being a
component.

---

## 7. Review — `policy/`

The bounds, in one reviewable dataclass:

| Bound | Limit |
|---|---|
| Contact frequency | 3 per customer per 7 days, **across all their events** |
| Quiet hours | No contact 21:00–08:00 IST |
| Incentive depth | ≤15% of order value **and** ≤₹2,000 absolute |
| Incentive eligibility | Intent-driven failures only |
| Incentive EV hurdle | Must buy ≥2× its cost in expected incremental recovery |
| Daily budget | ₹25,000/day across everything |
| Autonomy limit | >₹25,000 exposure → escalate to a human |
| Minimum EV | <₹50 expected → not worth the channel cost |
| Attempt cap | Per-reason, from the taxonomy |
| Timing floor | Never before the action window opens |
| Never-retry | Fraud declines, always |
| Unknown reason | Fail closed |
| Control arm | Never executed |

Two properties every rule holds to:

**Fail closed.** A rule that cannot evaluate — missing data, unknown code,
arithmetic it can't complete — denies. Recovery is worth money; uncontrolled
action costs more.

**No short-circuiting.** All thirteen run even after the first failure. Stopping
early would be faster and would make the audit trail useless: "denied by
quiet_hours" hides that the action also blew the budget and exceeded the attempt
cap.

Note the contact cap is **per customer, not per event** — the loophole that turns
a reasonable limit into a spam cannon for anyone with several failures. And
silent retries are not counted as contact, because conflating them would make the
system refuse to retry a bank outage on account of an unrelated email last week.

---

## 8. Act — `execute/`

`razorpay_client.py` wraps the SDK behind a single error type so a raw SDK
exception never escapes into decision code. Retries 5xx and gateway errors twice
with backoff; **never retries a 4xx** — a request rejected for a reason will be
rejected again, and retrying a payment call is how you double-charge someone.

`RECOUP_DRY_RUN` returns deterministic, obviously-fake responses without touching
the network, so the pipeline runs fully with no keys configured.

`outbox.py` never messages a real customer in *any* mode. It persists to
`data/outbox.jsonl` with real per-channel costs (email 10p, SMS 25p). Those
numbers are not decorative — the eval subtracts them from gross recovery, so a
zero would become a lie in the report.

`actions.py` refuses to execute unless the policy review allowed it. The caller
already checks, which is exactly why this checks too: the one path that forgets
is the one that matters.

---

## 9. Measure — `eval/`

The subsystem the project's credibility rests on.

**Frozen rolls.** Each event's luck is drawn once at generation time and reused.
Treatment and control therefore face *identical* luck, so measured lift reflects
the decision rather than sampling noise.

**Attribution.** An event is credited to the agent only when it recovered **and**
the counterfactual says it would not have. Recovering *after* an action is not
recovering *because of* one. Three buckets:

- `AGENT` — recovered only under the action taken. Claimable.
- `ORGANIC` — would have recovered anyway. Recoup takes no credit.
- `UNCLEAR` — reported separately, never folded into the headline.

**Cannibalisation** — money spent on customers whose outcome came back `ORGANIC`,
i.e. we paid people to do what they were already going to do. This is the
false-positive cost, and a recovery system that cannot state it does not know
whether it is profitable.

**Unreliable segments** are flagged, not quietly reported. Any segment with <30
events per arm is marked, because a 3-event bucket with 2 recoveries is not a 67%
recovery rate.

**Sensitivity sweep.** Outcomes are re-resolved under pessimistic, default and
optimistic lift assumptions and the *range* is reported. A result that only holds
at one parameter setting is not a result.

---

## 10. What is real and what is simulated

This section is the one to read before believing any number.

**Real — decision quality.** Whether Recoup picks the right strategy for a
failure reason, respects its bounds, refuses to retry risk declines, times
liquidity retries sensibly, defers out of quiet hours, and never exceeds budget.
These are properties of the code and hold regardless of any simulation
parameter. 263 tests cover them, including 27 adversarial bypass attempts.

**Simulated — the rupee figures.** Recoup has no access to a real merchant's
post-failure customer behaviour, so outcomes come from `seed/world.py`. Its lift
parameters are assumptions: plausible, conservative, and still assumptions.

Three things keep that from becoming a fudge:

1. **The pipeline is structurally blind to the simulator.** No module under
   `detect/`, `agent/`, `policy/`, `execute/` or `api/` can import the world
   model or read `data/oracle.json`. `tests/test_no_oracle_leak.py` parses the
   AST of every pipeline file and fails the build on any path to it — including a
   test that the detector itself detects.

2. **Lift is earned by correct actions only.** Retrying a fraud decline,
   discounting an outage, or hammering a liquidity failure five minutes early
   earn zero lift or negative. A simulator that rewarded activity would make
   Recoup look good for doing something stupid.

3. **Results are reported as a range,** not a point estimate.

The scorer's apparent calibration in the sandbox reflects internal consistency,
not predictive validity — its priors and the simulator share a domain model.
Against real traffic the priors would be wrong on day one, which is what
`fit_from_outcomes()` exists to correct.

---

## 11. Results (seed 42, 600 events, ₹34.5L at risk)

| | |
|---|---|
| **Incremental recovery rate** | **+7.2pp** (95% CI −0.0 … +14.5) |
| Incremental recovered | ₹1.3L across 46 events |
| Gross recovery rate | 27.1% — *what a system without a holdout would claim* |
| Control arm recovery rate | 19.9% — *with no help at all* |
| Events harmed | 0 |
| Sensitivity range | +2.5pp (pessimistic) → +10.5pp (optimistic) |
| Decisions | 600 taxonomy, 0 model |

Taxonomy-only, which is what a reviewer reproduces from a clean checkout with no
keys. Model-live figures are in §15 and are not independently checkable.

Reproducible from a clean checkout with no keys configured.

**Caveats, stated rather than buried:**

- **The interval includes zero** (−0.0 … +14.5pp — it clears by less than a
  basis point). At 600 events this sample cannot rule out no effect. The report leads with the interval rather than the
  point estimate, and the dashboard says "behind" when the estimate is behind.
- **This figure has moved several times, never once because a number was
  massaged** (§12, §12.1, §15). An attempt cap that could never fire, a fatigue
  slot reserved after the send rather than before, a model that had been falling
  back to the taxonomy without saying so.

  And, bluntly: until the generator stopped anchoring events to the wall clock,
  the headline moved *between identical runs*. Figures published before that
  commit are not reproducible and should not be compared with these — which is
  why this one is quoted alongside a fixed epoch and a test that enforces it.
- **Cannibalisation is ₹0, and on the model-live run that was measured rather
  than unexercised.** With the model deciding, ₹6,279 of discount was committed
  and ₹1,143 redeemed, and none of it landed on a customer who would have paid
  anyway (§15). Taxonomy-only runs report ₹0 for a weaker reason — the rules
  engine never proposes an incentive at all, so the metric has nothing to
  measure.
- Roughly 70% of at-risk *value* sits in `invoice_overdue`, which is 11.7% of
  events. Results are segmented by event kind so that tail cannot flatter the
  headline.

---

## 12. A bug worth documenting

The first end-to-end run reported **−1.3pp** incremental lift with an interval
straddling zero, and every retry strategy showing exactly zero incremental
recovery while persuasion showed all of it.

That pattern is not a finding, it is a symptom.

**Cause.** In dry run the Razorpay executors record `SKIPPED_DRY_RUN` while the
outbox records `SENT`, because the outbox simulates in every mode and has nothing
to skip. The grader treated `SENT` as "an action happened" and `SKIPPED_DRY_RUN`
as "nothing happened" — so 134 retries and 56 Payment Links were graded as though
Recoup had done nothing, while 172 nudges counted in full. The report was not
measuring recovery strategy. It was measuring which executor writes an outbox
row.

**The subtle part.** The rule it enforced — *"a dry run cannot manufacture
lift"* — sounds correct and violated its own premise. Because the outbox never
messages a real customer in any mode, turning `RECOUP_DRY_RUN` off would have
moved every number in the report. The flag was a dial on the results, which is
precisely what it must not be.

**Fix.** Grade dry-run dispatches identically to sent ones, and put execution
mode in the report header where it cannot be missed. What decides an outcome is
`world.py`, which has no opinion about whether an HTTP request left the machine.

The failing test was rewritten rather than deleted — it now seeds the same event
twice, once per transport, and asserts the outcomes are identical.

### 12.1 Two bounds that could not fire

An adversarial review wrote 27 tests against the policy engine. All 27 passed
against code that should have refused them. Two findings stand out because the
bounds in question were *advertised* and doing nothing:

**`attempt_cap` was structurally inert.** It counted only `ActionRun`s Recoup
had made, and `run()` processes each event exactly once and leaves it non-OPEN —
so the count was always zero and the rule could never deny anything.
`RevenueEvent.attempt_no` was invisible to policy. A `card_expired` order on its
fourth attempt, against a taxonomy ceiling of one, executed. The dashboard would
have shown "zero denials", which reads as *nothing ever hit the cap* rather than
*the cap cannot fire*. It now denies 13 events in a full run.

**`ESCALATE` queued nothing.** Policy escalation set `AWAITING_APPROVAL` and
returned — no `ActionRun`, no queue entry, nothing anywhere listing the event for
a person. A ₹5,00,000 receivable sat unrecovered and unnoticed. These are by
definition the highest-value events in the system.

Also closed: `Bounds` was a caller-supplied argument that could *loosen* every
limit with no trace in the audit row; the control-arm rule compared by identity
so a cohort arriving as a plain string was waved through as treatment;
`review()` raised instead of denying on five malformed inputs, leaving no
`PolicyReview` row at all; a `Review` carried no provenance, so a genuine ALLOW
for one event authorised a fraud retry on another; and `delay_hours` was
validated, persisted, and read by nothing.

**One defect the fixes introduced, caught by the next run.** The executor's new
control-arm check refused `NO_ACTION` and escalations on held-out events. Those
touch nothing outside the database and are exactly how a control event gets its
decision recorded without being acted on — refusing them would leave the holdout
with no audit trail and make the counterfactual unreadable.

---

## 13. What would change for production

- **Priors → fitted rates.** `fit_from_outcomes()` exists; it needs real
  outcomes. Two weeks of live traffic replaces every number in the taxonomy.
- **Holdout 30% → 5–10%.** The wide holdout is a small-sample necessity, not a
  design preference.
- **Webhooks instead of polling.** Recovery attribution currently resolves after
  a fixed window; Razorpay's `payment.captured` and `payment_link.paid` webhooks
  would resolve it on arrival.
- **The outbox becomes real.** It is a deliberate stub — the messaging provider
  is the one component with no interesting design content and real spam risk.
- **Per-merchant bounds.** `Bounds` is a single dataclass by design; in
  production it would be per-merchant configuration with an approval trail.
- **The outbox becomes transactional.** The fatigue slot is reserved before the
  send and released if nothing goes out, which closes the window that mattered;
  a production version would move the outbox into a table so the message and its
  ContactLog row commit together, and render the JSONL after commit.

---

## 14. Running it

```bash
pip install -r requirements.txt
cp .env.example .env              # rzp_test_ keys + a model key (any provider)
python scripts/check_setup.py     # read-only: validates keys, db, mode
python scripts/seed.py            # build the synthetic merchant
python scripts/run_pipeline.py    # assess → decide → review → act
python scripts/run_eval.py        # grade it honestly
python scripts/serve.py           # dashboard on :8000
python scripts/robustness.py      # five datasets
python scripts/verify_live.py     # one real Payment Link
pytest -q                         # 263 tests
```

Recoup refuses to start against a `rzp_live_` key. It sends messages and spends
money; a project that should only ever run in test mode ought to make that
structurally impossible rather than merely intended.

---

## 15. Did the model earn its place?

### The model did not beat the lookup table

With Gemini deciding the 161 events the taxonomy could not settle on its own:

| | Taxonomy only | With the model |
|---|---|---|
| Incremental lift | **+7.2pp** | **+6.8pp** |
| 95% CI | −0.0 … +14.5 | −0.5 … +14.0 |
| Events caused | 46 | 44 |
| Incentive committed | ₹0 | **₹6,279** |
| Incentive redeemed | ₹0 | ₹1,143 |
| Cannibalisation | ₹0 | **₹0** |
| Cost per incremental rupee | 0.02p | **1.20p** |
| Decisions by the model | 0 | 161 of 600 |

The model was handed every event the deterministic path found genuinely
ambiguous, committed ₹6,279 of discount, and returned *slightly less*
incremental lift than the taxonomy managed alone.

Two qualifications belong beside that, not below it. The gap is a fraction of a
±7pp interval, so this is one sample and not a finding. And it is not
like-for-like: incentives are only ever proposed on the model path, so the model
was attempting something the rules engine never tries at all.

But cannibalisation came back at **₹0** against ₹6,279 committed. Every discount
reached a customer who would not otherwise have paid — the eligibility rule and
the EV hurdle held. The model did not waste money; it simply did not beat the
table.

That is the measurement this project exists to be able to make. A recovery
system that cannot tell you whether its most expensive component earns its place
is not measuring anything.

---

## 16. Robustness and live verification

### Does it hold on other data?

Every headline figure comes from seed 42. Reproducible is not the same as robust,
so `scripts/robustness.py` re-runs seed → pipeline → resolve → measure on **28
independent datasets**. It is the companion to the sensitivity sweep: that varies
the *assumptions* with the data fixed, this varies the *data* with the
assumptions fixed.

| | across 28 datasets |
|---|---|
| Lift range | **−4.0pp … +16.2pp** |
| Median | **+7.7pp** |
| Spread (sd) | 4.6pp |
| Positive point estimate | **26 of 28** |
| Interval excludes zero | **15 of 28** |
| At or above +5pp | 21 of 28 |

**Two datasets came out negative** (seeds 5 and 41). That matters more than the
median does, and it is here rather than in a footnote.

It also matters *how* it was found. The first version of this check ran five
seeds, all five were positive, and this section previously said the direction held
on every dataset. Widening to 28 found the counterexamples the smaller sample had
missed — the claim had been true of the sample and false of the system, and
nothing in the five-seed run hinted at the difference. A robustness check small
enough to miss its own counterexamples is a lucky draw with a table around it.

**Read the spread carefully — it is the estimator moving, not the system.**

| | across the same 28 datasets |
|---|---|
| Arm-difference lift | −4.0 … +16.2pp — **enormous** |
| Events Recoup actually caused | 25 … 48 — **1.9× spread** |

The lowest-lift datasets are not the ones where Recoup did least; they are the
ones whose *control arm recovered best*. The arm difference is the only quantity
a real deployment could compute, and at a ~180-event holdout it is dominated by
which way the holdout fell. The per-event causal count is far steadier — and it
is exact only because the frozen roll supplies a counterfactual for every event,
which a live merchant could never have.

So the spread argues for a larger holdout or more events, not for distrusting the
pipeline. `scripts/robustness.py` prints both numbers side by side for exactly
this reason.

The model is disabled during these runs. Letting a non-deterministic component
vary alongside the data would leave no way to say which one moved a lift.


### The Razorpay path is verified against the live API

`scripts/verify_live.py` creates one ₹499 Payment Link in test mode and reads it
back — because every other check verifies that Recoup *would* call correctly,
which is not the same claim as a 200 from the service.

It also asserts the property that actually matters. Seeded customers carry
fabricated emails and phone numbers, and Razorpay will send a Payment Link to
whatever contact details it is handed, on its own schedule — outside quiet hours,
the weekly contact cap and the cost ledger. So every link Recoup creates silences **every channel Razorpay offers** —
`sms`, `email` and `whatsapp` — plus `reminder_enable`, and the script checks the
request carried all of them.

Reading the values back is not enough on its own, and getting that wrong is how
`whatsapp` went unsilenced for several commits. It was absent from every request,
the readback reported it false because that is the *merchant account's* default,
and the script called it verified — it was verifying the account. A merchant with
WhatsApp notifications switched on would have had Razorpay message every
fabricated number in the seed set.

That check failing would not be cosmetic. It would mean a synthetic dataset had
begun emailing real inboxes at addresses that happen to exist.
