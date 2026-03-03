#set page(margin: 1.5cm)
#set text(size: 11pt)
#set math.equation(numbering: "(1)")

= Question 1

The general function-approximation TD update is:

$
  theta_k <-
  theta_k + epsilon [r_t + gamma accent(V, hat)^pi (s_(t+1)) - accent(V, hat)^pi (s_t)]
  (partial f (s_t, bold(theta)))/ (partial theta_k)
$

To recover tabular TD, let $bold(theta)$ have one entry per state and define $f(s, bold(theta)) = sum_j theta_j bb(1)(s = j)$. Then $(partial f) / (partial theta_k) = bb(1)(s_t = k)$, giving:

$
  theta_k <- theta_k + epsilon [r_t + gamma accent(V, hat)^pi (s_(t+1)) - accent(V, hat)^pi (s_t)] bb(1)(s_t = k)
$

Only the parameter for the current state is updated, which is exactly lookup-table TD(0).

= Question 2

For the linear approximator $f(s, bold(w)) = sum_k w_k phi_k (s)$, the partial derivative is simply $(partial f) / (partial w_k) = phi_k (s_t)$. Substituting into Eq.~(1):

$
  w_k <- w_k + epsilon underbrace([r_t + gamma accent(V, hat)^pi (s_(t+1)) - accent(V, hat)^pi (s_t)], delta_t) phi_k (s_t)
$

Each weight update is the product of two factors: a scalar TD error $delta_t$ (shared across all $k$) and the feature activation $phi_k (s_t)$. Features that are active in the current state absorb the prediction error; inactive features are unchanged. This enables generalisation: states sharing features transfer learned value.

= Question 3

We implement TD with a tapped delay line representation ($phi_(i tau)(bold(S)) = S_(i tau)$) and the linear update from Q2. @q3a shows the task inputs: a spike stimulus at $t = 10$~s and a Gaussian reward centred at $t = 20$~s.

#figure(
  image("./figures/question-3/q3a_stim_reward.png", width: 75%),
  caption: [Stimulus and reward as functions of time.],
) <q3a>

@q3b shows three key observables recorded every 10th trial (colour runs from dark = trial 1 to light = trial 201):

#figure(
  image("./figures/question-3/q3b_td_learning.png", width: 75%),
  caption: [$accent(V, hat)(t)$, $Delta accent(V, hat)(t)$, and $delta(t)$ across 21 sampled trials.],
) <q3b>

*Observations.*
- _Early trials_: $delta$ peaks at reward time because the model has not yet assigned value to earlier states.
- _Mid-learning_: TD bootstrapping propagates value backward in time, building an anticipatory ramp in $accent(V, hat)$ after the stimulus.
- _Late trials_: $delta$ shifts to stimulus onset while the reward-time error vanishes as the reward becomes fully predicted.

*Dopamine correspondence.* In the TD theory, dopamine neuron firing corresponds to $delta(t)$, the prediction error, not the value $accent(V, hat)(t)$. The simulation reproduces the experimentally observed cue-ward shift: the $delta$ bump migrates from reward time to stimulus time over the course of learning, with no intermediate activity between the two events.

#pagebreak()

= Appendix: AI Disclosure

This report was drafted by hand and refined with AI assistance as detailed below. All simulation were verified against the handout; all results were inspected and interpreted by the author.

*GitHub Copilot* — autocomplete during report writing, primarily for equation typesetting.

*Claude Code* — the following prompts were used:
- "read src/coursework/assignment.pdf and give me a tip for question 1"
- "give me a hint for question 2"
- "walk me through example of value functions from reinforcement learning"
  - "how do we go from this to TD learning?"
  - "explain question 3"
- "refine the report so far"
- " src/lib/core.py defines an experiment, add the plots needed by src/pdfs/assignment.pdf question 3 as publication quality matplotlib plots with monospace font, and save the plots to src/report/figures/question-3"
- "read the assignment and report and bugs in the code"
- "in src/lib/core.py properly parameterise run_td_tapped to support both tapped and boxcar representations."
- "add the plots for question-4 in the usual way to src/lib/core.py"
- "now add in question-5, same as question-4 except reward_probability=0.5, n_trials=1000 and seed=11"
- "now for question-6 do a sweep of the reward probabilities of [0.0,0.25,0.5,0.75,1.0] and make the appropriate plots"