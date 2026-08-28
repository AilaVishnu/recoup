# Recoup — 5-minute pitch

A script, not an outline. Timings are for a 5:00 recording; the spoken words are
roughly 700, which is a comfortable pace with pauses.

**Before you record:** `python scripts/check_setup.py`, then a fresh
`python scripts/seed.py && python scripts/run_pipeline.py && python scripts/run_eval.py`,
then start the dashboard. A Payment Link failing on camera because a key rotated
is a bad five minutes, and it is avoidable.

**Have open:** terminal with the eval report already run, browser on
`localhost:8000`, and `recoup/taxonomy.py` in the editor.

---

## 0:00 – 0:35 · The problem, and the trap in it

> Every Razorpay merchant loses revenue three ways that look different and are
> the same shape. A payment is attempted and fails. A checkout is started and
> abandoned. An invoice goes past due. The customer wanted to buy, the merchant
> wanted to sell, and the money didn't move.
>
> The obvious response is to chase all of it. That's a trap, and it's the trap
> most recovery tools fall into — because **a large share of that revenue arrives
> anyway.** The outage ends, the customer retries, the invoice gets paid on the
> 5th. Any system that sent an email in the meantime will happily take credit.
>
> I built Recoup around not doing that.

*On screen: the README's opening, or nothing — just talk.*

---

## 0:35 – 1:20 · A failed payment is not one thing

*Open `recoup/taxonomy.py`. Scroll slowly through the profiles.*

> This is the file the whole project rests on. Sixteen failure reason codes,
> mapped to Razorpay's `error_source` and `error_step` schema, and each one
> carries what it *implies*.
>
> An issuer outage and a fraud decline both surface as "payment failed". One
> should be retried in two hours, untouched, and the other must **never** be
> retried — re-presenting a risk decline is how a merchant earns a higher decline
> rate and a chargeback problem.
>
> Insufficient funds is my favourite. Intent is intact, the balance isn't. So
> Recoup times the retry to the salary window — the 1st to the 5th — because a
> retry on the 22nd isn't wrong strategy, it's an empty account.

*Point at `incentive_eligible`.*

> And only **two of sixteen** reason codes let Recoup spend money. That's not a
> budget setting, it's from the domain: a discount can only move a failure whose
> cause was intent. Discounting a bank outage pays a customer to do what they
> were already going to do.

---

## 1:20 – 2:10 · The two claims

> Everything else follows from two decisions.
>
> **First — 30% of events go into a randomised control arm.** They're scored and
> decided *exactly* like treated events. The full reasoning is computed and
> stored. Then execution is suppressed at the last step. So for every held-out
> event I know precisely what Recoup *would* have done, and what happened when it
> didn't. The headline number is the difference.
>
> **Second — the model proposes, and a policy engine disposes.**

*Open `localhost:8000/policy`.*

> Thirteen bounds. The agent cannot message anyone between 9pm and 8am IST,
> cannot contact a customer more than three times a week across all their events,
> cannot discount a technical failure, cannot exceed 15% or ₹2,000, cannot spend
> past ₹25,000 a day, and cannot act on more than ₹25,000 of exposure without a
> human.
>
> These aren't prompt instructions. Prompt guardrails fail *silently* — the one
> time the model ignores one, nothing catches it. These run **after** the model
> has spoken, and every check is recorded, passes included.

---

## 2:10 – 3:05 · The screen that proves it

*Go to `/events`, filter cohort = control, open one.*

> This is one event, replayed end to end. The failure. The score, and every
> feature it used. The proposed action with its verbatim rationale. All fourteen
> checks — thirteen bounds plus an input-validation gate — with what each one
> said.
>
> And then this.

*Point at the suppressed banner and read it.*

> *"Suppressed for the holdout. The decision above is exactly what Recoup would
> have done — computed, reviewed and recorded, and then not sent."*
>
> That's the counterfactual, and it's why the incremental number means anything.

*Open a treated event beside it if you have time.*

---

## 3:05 – 4:00 · The numbers, honestly

*Switch to the terminal showing the eval report.*

> Gross recovery is 27.1%. The control arm recovered 19.9% **with no help at
> all**. So the honest number is the difference: **seven point two percentage
> points**, about ₹1.3 lakh that would not have come back.
>
> A tool without a holdout would have put 27.1% on the slide.

*Point at the CI.*

> And I have to say this: **the interval includes zero.** It clears by less than
> a basis point. At 600 events this sample can't rule out no effect. The
> report leads with the interval rather than the point estimate for exactly that
> reason, and the dashboard says "behind" when the estimate is behind.
>
> The rupee figures are simulated — I don't have a real merchant's post-failure
> customer behaviour. What's *real* is the decision quality, and the pipeline is
> structurally blind to the simulator: a test parses the AST of every module and
> fails the build if any of them can reach the answers.

---

## 4:00 – 4:40 · The bug

> One thing I want to show you, because it's the most useful thing I did.
>
> My first end-to-end run reported **negative** lift, and every retry strategy
> showing exactly zero. That's not a finding, it's a symptom.
>
> In dry-run mode the Razorpay executors marked actions "skipped" while the
> outbox marked them "sent" — so the grader threw away 190 of 384 actions and
> counted 172. The report wasn't measuring recovery strategy at all. It was
> measuring which executor happened to write an outbox row.
>
> The rule it enforced sounded right — *a dry run can't manufacture lift* — and
> it violated its own premise, because flipping the flag would have moved every
> number in the report.
>
> Fixing it later cost me almost a point of headline lift, when an attempt cap
> that could never fire started denying thirteen events it should always have
> denied. I published the lower number.

---

## 4:40 – 5:10 · The model didn't win, and I can prove it

> Last thing. I gave Gemini the 161 events the taxonomy couldn't settle. It spent
> eight thousand rupees on discounts and came back with **six and a half points
> of lift where the lookup table alone got seven and a half.**
>
> That difference is inside the error bars, so I'm not claiming the model made it
> worse. What I'm claiming is that I can *tell* — and cannibalisation came back at
> zero, so every rupee of discount went to someone who genuinely wasn't coming
> back. The bounds worked. The model just didn't beat the table.
>
> A recovery system that can't tell you whether its most expensive component is
> earning its place isn't measuring anything.

---

## 5:10 – 5:30 · Close

> The model is swappable — any OpenAI-compatible endpoint, or Anthropic, two
> lines of config — and the policy engine never learns which one answered. A
> proposal from a frontier model and one from a free tier clear the same bounds
> or neither does.
>
> Because the model isn't the architecture. The taxonomy, the bounds, and the
> measurement are. That's Recoup.

---

## Notes

**Cut first if you run long:** the treated-event comparison at 3:05, and the
provider paragraph at 4:40. Everything else carries an argument.

**Do not cut the bug.** Anyone can demo a system that works. Showing that you
chased an implausible number instead of shipping it, and that you published a
figure that went *down* after you fixed your own code, is the strongest signal in
five minutes of video.

**Say "the interval includes zero" out loud.** A reviewer who works that out
themselves will assume you hoped they wouldn't.

**The model-didn't-win section is the strongest 30 seconds you have.** Every
other entry will claim their LLM helped. Being the one candidate who measured it,
found it didn't, and said so on camera is worth more than a bigger number would
have been. If you are over time, cut the provider paragraph in the close instead.
