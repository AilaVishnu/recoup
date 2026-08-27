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
| **Incremental recovery rate** | **+7.0pp** (95% CI −0.3 … +14.3) |
| Incremental recovered | **₹1.3L** across 45 events |
| Gross recovery rate | 26.9% — *what a system without a holdout would claim* |
| Control arm | 19.9% — *recovered with no help at all* |
| Events harmed | 0 |
| Sensitivity range | +2.3pp (pessimistic) → +9.6pp (optimistic) |

The gap between 26.9% and +7.0pp is the entire point of this project. A recovery
tool without a holdout would have reported the first number.

**Caveats, stated rather than buried:**

- **The 95% interval includes zero.** The point estimate is positive and stays
  positive across the whole pessimistic-to-optimistic sweep, but at 600 events
  this sample cannot rule out no effect. The report says so on its own front
  panel rather than quoting the point estimate alone.
- An earlier version of this table read +7.9pp. Fixing an attempt cap that
  could never fire removed 13 events that should never have been acted on, and
  the headline came down with them. A fix that lowers your own number is still
  a fix.
- **Cannibalisation is ₹0 and currently cannot be otherwise** — incentives are
  only proposed on the LLM path, so without an API key the metric is implemented
  and tested but never exercised.
- ~70% of at-risk *value* sits in `invoice_overdue`, which is 11.7% of events.
  Results are segmented by event kind so that tail cannot flatter the headline.

Full reasoning, design decisions and a documented bug: **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env              # rzp_test_ keys + ANTHROPIC_API_KEY
python scripts/check_setup.py     # read-only: validates keys, db, mode
python scripts/seed.py            # build the synthetic merchant
python scripts/run_pipeline.py    # assess → decide → review → act
python scripts/run_eval.py        # grade it honestly
python scripts/serve.py           # dashboard on :8000
pytest -q                         # 231 tests
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
