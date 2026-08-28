# Recoup

**A bounded revenue-recovery agent for Razorpay merchants.**

Recoup watches every revenue-at-risk event on a merchant account — failed
payments, abandoned checkouts, overdue receivables — decides *one* recovery
action per event, executes it through Razorpay test-mode APIs, and measures what
it actually recovered against a randomised holdout.

Razorpay AI Buildathon 2026 · Track 3, AI Revenue Recovery.

---

## The two claims this project is built to support

Most revenue-recovery tooling makes one claim: *we recovered X% of your failed
payments.* That claim is almost always inflated, for a reason that is easy to
miss — a large share of failed payments recover on their own. The bank outage
ends, the customer tries again, the money arrives. Any system that sent an email
in the meantime will happily take credit for it.

Recoup is built around not doing that.

### 1. It reports incremental recovery, not gross recovery

30% of events are randomly assigned to a **control arm**. Control events are
scored and decided exactly as treatment events are — the full decision is
computed and recorded — and then execution is suppressed at the last step. So
for every held-out event, we know precisely what Recoup *would* have done, and
what happened when it didn't.

The headline number is the difference between the arms. Gross recovery is
reported too, next to it, so the gap between the flattering number and the honest
one is visible rather than hidden.

The holdout costs real money — about ₹11.7L of at-risk value in the reference
dataset goes untouched. At this sample size, a narrower holdout would leave the
lift estimate too noisy to mean anything, and a system that cannot measure itself
is not one worth deploying.

### 2. The model proposes; a policy engine disposes

The agent can suggest whatever it likes. It **cannot**:

- message a customer between 21:00 and 08:00 IST
- contact anyone more than 3 times in 7 days, across all their events
- discount a technical failure (a bank outage doesn't get cheaper with a coupon)
- discount more than 15% of order value, or ₹2,000, whichever binds first
- spend past a ₹25,000/day incentive budget
- act on more than ₹25,000 of exposure without a human
- retry a payment declined for suspected fraud, ever
- touch the control arm

These aren't prompt instructions. They're [`recoup/policy/rules.py`](recoup/policy/rules.py) —
code that runs *after* the model has spoken and that the model has no way to
reach around. Prompt-level guardrails fail silently; the one time the model
ignores one, nothing catches it. A rule that runs afterward catches it every
time, and writes down that it caught it.

Every check is recorded — the passes as well as the failures. *"This action was
permitted because these thirteen bounds were checked and cleared"* is a claim the
audit trail can actually support.

---

## The domain core

A failed payment is not one thing. [`recoup/taxonomy.py`](recoup/taxonomy.py) classifies 16
reason codes against Razorpay's `error_source` / `error_step` schema and derives
what each one implies:

| Failure | Strategy | Incentive? | Why |
|---|---|---|---|
| `issuer_down` | wait 2h, retry untouched | never | Nothing is wrong with the customer |
| `incorrect_otp` | retry now, same rail | never | Fat-finger; highest-yield recovery there is |
| `insufficient_funds` | retry in the next salary window | never | Intent is intact, the balance isn't |
| `card_expired` | switch rails, one attempt | never | Retrying the same card is guaranteed to fail |
| `payment_cancelled` | persuade | **yes** | Nothing broke — the customer chose not to pay |
| `fraud_suspected` | **do not retry** | never | Re-presenting a risk decline earns chargebacks |

Two of sixteen reason codes permit spending money. That constraint comes from the
domain, not from a budget setting: a discount can only move a failure whose cause
was intent. Everywhere else it's pure margin burn — you paid the customer to do
what they were already going to do.

An unrecognised reason code fails closed: no action, surfaced for a human, and
counted as a coverage metric rather than quietly dropped.

---

## Pipeline

```
RevenueEvent      what went wrong, and how much is at stake
  ↓
Assessment        deterministic score — no LLM. ~80% of events resolve here.
  ↓
Decision          proposed action + rationale (rules, or the model when it's genuinely ambiguous)
  ↓
PolicyReview      13 bounds, all of them, no short-circuiting
  ↓
ActionRun         what actually executed against Razorpay, and what it cost
  ↓
Outcome           did the money come back, and may we claim it
```

One row per stage, immutable once written. Replaying any event is a join, not a
reconstruction — which is what makes "trace this recovered rupee back to the
decision that recovered it" an answerable question.

**The LLM is not in the hot path.** The taxonomy plus a handful of customer
signals settles most events deterministically. The model is called for the
ambiguous, high-value minority — where the marginal decision quality is worth the
latency and the cost. Knowing when *not* to call a model is part of the design.

**And the model is swappable.** Any OpenAI-compatible endpoint works — OpenAI,
Groq, Google Gemini, OpenRouter, Together, a local Ollama — alongside the
Anthropic SDK, selected by two lines of `.env`. The policy engine never learns
which one answered: a proposal from a frontier model and one from a free tier
clear the same thirteen bounds, or neither does. That is tested rather than
asserted — [`tests/test_providers.py`](tests/test_providers.py) decides the same
event through two providers and requires identical output.

This matters for a question a reviewer will ask: *what if the model is bad?*
The answer is in code. A bad proposal is refused by the same bounds a good one
passes, and the taxonomy answers instantly when the model is unreachable.

---

## Honesty about what is measured

**Real:** decision quality. Whether Recoup picks the right strategy for a failure
reason, respects its bounds, refuses to retry risk declines, times liquidity
retries sensibly, and never exceeds budget. These are properties of the code and
hold regardless of any simulation parameter.

**Simulated:** the rupee figures. Recoup has no access to a real merchant's
post-failure customer behaviour, so outcomes come from a world model in
[`recoup/seed/world.py`](recoup/seed/world.py). Its lift parameters are assumptions — plausible and
conservative, but assumptions.

Three things keep that from becoming a fudge:

1. **The pipeline is structurally blind to the simulator.** No module under
   `detect/`, `agent/`, `policy/`, `execute/` or `api/` can import the world
   model or read `data/oracle.json`. [`tests/test_no_oracle_leak.py`](tests/test_no_oracle_leak.py)
   parses the AST of every pipeline file and fails the build on any path to it —
   including a test that the detector itself detects.

2. **Lift is earned by correct actions only.** Retrying a fraud decline,
   discounting an outage, or hammering a liquidity failure five minutes later
   earn zero lift or negative. A simulator that rewarded activity would make
   Recoup look good for doing something stupid.

3. **Results are reported as a range.** The eval sweeps the lift assumptions from
   pessimistic to optimistic and reports the range. A result that only holds at
   one parameter setting is not a result.

The scorer's calibration in the sandbox reflects internal consistency, not
predictive validity — its priors and the simulator share a domain model.
`fit_from_outcomes()` is the path from guess to measured: it estimates per-reason
rates from observed outcomes and shrinks them toward the prior in proportion to
how little evidence exists. Every reported score states which of the two produced
it.

---

## Results

Seed 42 · 600 events · ₹34.5L at risk

| | |
|---|---|
| **Incremental recovery rate** | **+6.5pp** (95% CI −0.7 … +13.8) |
| Incremental recovered | **₹91,830** across 43 events |
| Gross recovery rate | 26.4% — *what a system without a holdout would claim* |
| Control arm | 19.9% — *recovered with no help at all* |
| Net after cost | ₹91,018 |
| Cannibalisation | **₹0** — no discount reached a customer who'd have paid anyway |
| Events harmed | 0 |
| Sensitivity range | +2.5pp (pessimistic) → +9.6pp (optimistic) |
| Decisions | 439 by the taxonomy, **161 by the model** |

Figures above are from a run with the model live. The pipeline also runs with
**no keys configured at all** — `seed.py && run_pipeline.py && run_eval.py` on a
clean checkout gives +7.5pp with every decision taken by the taxonomy. Keys
upgrade simulated execution to real Razorpay calls and taxonomy-only decisions
to model-assisted ones; they do not gate the run.

The gap between 26.4% and +6.5pp is the entire point of this project. A recovery
tool without a holdout would have reported the first number.

**Caveats, stated rather than buried:**

- **The interval includes zero** (−0.7pp at the low end). At 600 events this
  sample cannot rule out no effect. The report leads with the interval rather
  than the point estimate for that reason, and the dashboard says "behind" when
  the point estimate is behind.
- **This number has moved three times and never once because a number was
  massaged.** +7.9pp until an attempt cap that could never fire began denying 13
  events it always should have; +7.5pp until the fatigue slot started being
  reserved before the send; +6.5pp once the model was actually deciding rather
  than falling back. A figure that only looks good until you fix your own bugs
  is not worth defending.

### The model did not beat the lookup table

With Gemini deciding the 161 events the taxonomy could not settle on its own:

| | Rules only | With the model |
|---|---|---|
| Incremental lift | **+7.5pp** | **+6.5pp** |
| 95% CI | +0.2 … +14.7 | −0.7 … +13.8 |
| Incentive committed | ₹0 | **₹8,307** |
| Incentive redeemed | ₹0 | ₹784 |
| Cannibalisation | ₹0 | **₹0** |
| Cost per incremental rupee | 0.02p | **0.88p** |

The model was handed every event the deterministic path found genuinely
ambiguous, proposed ₹8,307 of discounts, and produced a *slightly lower*
incremental lift than the taxonomy managed alone.

Two things must be said before that becomes a claim. The difference is well
inside the ±7pp interval, so this is one sample rather than a result. And it is
not like-for-like: incentives are only ever proposed on the model path, so the
model was attempting something the rules engine never tries at all.

But cannibalisation came back at **₹0** — every discount went to a customer who
would not otherwise have paid. The eligibility rule and the EV hurdle did their
job. The model did not waste money; it simply did not beat the table.

That is the measurement this project exists to be able to make. A recovery
system that cannot tell you whether its most expensive component is earning its
place is not measuring anything.

- **Cannibalisation is ₹0 and currently cannot be otherwise** — incentives are
  only proposed on the LLM path, so without an API key the metric is implemented
  and tested but never exercised.
- ~70% of at-risk *value* sits in `invoice_overdue`, which is 11.7% of events.
  Results are segmented by event kind so that tail cannot flatter the headline.


### Does it hold on other data?

Every figure above comes from seed 42. Reproducible is not the same as robust,
so `scripts/robustness.py` re-runs seed → pipeline → resolve → measure on
independent datasets. It is the companion to the sensitivity sweep: that varies
the *assumptions* with the data fixed, this varies the *data* with the
assumptions fixed.

| seed | lift | 95% CI | gross | control |
|---|---|---|---|---|
| 1 | **+10.6pp** | +4.1 … +17.0 | 24.5% | 13.9% |
| 7 | **+8.3pp** | +1.1 … +15.5 | 26.4% | 18.2% |
| 42 | **+7.2pp** | −0.0 … +14.5 | 27.1% | 19.9% |
| 99 | **+7.0pp** | −0.1 … +14.0 | 24.4% | 17.5% |
| 2024 | **+1.2pp** | −6.0 … +8.4 | 22.9% | 21.7% |

Positive on all five, median +7.2pp, spread 3.5pp. **But the magnitude moves
roughly ninefold, and only two of the five intervals exclude zero.** That is
evidence of a consistent *sign*, not of a reliable *effect size*, and the script
prints it that way rather than reporting "5 of 5 positive" — those are different
claims and quoting the flattering one is the failure this project is built to
avoid.

**Read that spread carefully — it is the estimator moving, not the system.**

| | across the five datasets |
|---|---|
| Arm-difference lift | +1.2 … +10.6pp — **9× spread** |
| Events Recoup actually caused | 30 … 46 — **1.5× spread** |

The lowest-lift dataset is not the one where Recoup did least; it is the one
whose *control arm recovered best* (21.7% against 13.9% at the other end). The
arm difference is the only quantity a real deployment could compute, and at a
180-event holdout it is noisy. The per-event causal count is stable — and it is
exact only because the frozen roll gives a counterfactual for every event, which
a live merchant could never have.

So the spread argues for a larger holdout or more events, not for distrusting
the pipeline. `scripts/robustness.py` prints both numbers side by side for
exactly this reason.

The model is disabled during these runs. Letting a non-deterministic component
vary alongside the data would leave no way to say which one moved a lift.


### The Razorpay path is verified against the live API

`scripts/verify_live.py` creates one ₹499 Payment Link in test mode and reads it
back — because every other check verifies that Recoup *would* call correctly,
which is not the same claim as a 200 from the service.

It also asserts the property that actually matters. Seeded customers carry
fabricated emails and phone numbers, and Razorpay will send a Payment Link to
whatever contact details it is handed, on its own schedule — outside quiet hours,
the weekly contact cap and the cost ledger. So every link Recoup creates sets
`notify.email`, `notify.sms` and `reminder_enable` to false, and the script reads
those back off the created object rather than trusting the request carried them.
Confirmed false on all three channels.

That check failing would not be cosmetic. It would mean a synthetic dataset had
begun emailing real inboxes at addresses that happen to exist.

Full reasoning, design decisions and a documented bug: **[ARCHITECTURE.md](ARCHITECTURE.md)**.
The 5-minute pitch script: **[PITCH.md](PITCH.md)**.

---

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env              # rzp_test_ keys + a model key (see below)
python scripts/check_setup.py     # read-only: validates keys, db, mode
python scripts/seed.py            # build the synthetic merchant
python scripts/run_pipeline.py    # assess → decide → review → act
python scripts/run_eval.py        # grade it honestly
python scripts/serve.py           # dashboard on :8000
python scripts/robustness.py      # does it hold on other datasets?
python scripts/verify_live.py     # one real Payment Link, notifications off
pytest -q                         # 252 tests
```

Recoup refuses to start against a `rzp_live_` key. It sends messages and spends
money; a project that should only ever run in test mode ought to make that
structurally impossible rather than merely intended.

---

## Status

| | |
|---|---|
| ✅ Failure taxonomy — 16 reason codes, fail-closed | `recoup/taxonomy.py` |
| ✅ Audit-trail schema — 9 tables, one row per stage | `recoup/db.py` |
| ✅ Synthetic merchant + world model | `recoup/seed/` |
| ✅ Feature extraction, incl. liquidity-window timing | `recoup/detect/features.py` |
| ✅ Recoverability scorer + empirical recalibration | `recoup/detect/scorer.py` |
| ✅ Policy engine — 13 bounds + an input-validation gate, 27 bypass tests | `recoup/policy/rules.py` |
| ✅ Action executors against Razorpay test mode | `recoup/execute/` |
| ✅ Agent decision layer — routes to the model only where it earns its place | `recoup/agent/` |
| ✅ Eval harness — incremental lift, cannibalisation, sensitivity sweep | `recoup/eval/` |
| ✅ Pipeline orchestrator — simulated time, quiet-hours deferral | `recoup/pipeline.py` |
| ✅ Dashboard — replay any event end to end | `recoup/api/` |
| ✅ Provider layer — any OpenAI-compatible endpoint or Anthropic | `recoup/agent/providers.py` |
