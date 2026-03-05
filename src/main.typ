// -- Document setup ----------------------------------------------------------

#set document(
  title: "4G3 Assignment 1",
)

#set page(
  paper: "a4",
  margin: 1.5cm,
  numbering: "1"
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
)

#set par(justify: true)

#set heading(numbering: none)

#set math.equation(numbering: "(1)")

#set figure(gap: 0.8em)

#show figure.caption: set text(size: 10pt)

#show link: underline

// -- Title -------------------------------------------------------------------

#align(center)[
  #text(size: 16pt, weight: "bold")[4G3 Computational Neuroscience --- Assignment 1]

  #v(0.4em)

  Blind Grading Number: #text(fill: red)[*8415L*]
]

#v(0.6em)

// -- AI disclosure (required at the beginning) -------------------------------

#block(
  width: 100%,
  inset: 8pt,
  stroke: 0.5pt + luma(160),
  radius: 2pt,
)[
  *AI Disclosure.* This report was drafted by hand and refined with AI
  assistance. GitHub Copilot was used for autocomplete during writing. Claude
  Code was used for coding assistance and report drafting. Prompts are
  listed in Appendix B. All simulations were verified against the handout; all
  results were inspected and interpreted by the author.
]

#v(0.4em)

// -- Questions ---------------------------------------------------------------

= Question 1

The general function-approximation TD update is:

$
  theta_k <-
  theta_k + epsilon [r_t + gamma accent(V, hat)^pi (s_(t+1)) - accent(V, hat)^pi (s_t)]
  (partial f (s_t, bold(theta))) / (partial theta_k)
$ <td-general>

To recover tabular TD, let $bold(theta)$ have one entry per state and define
$f(s, bold(theta)) = sum_j theta_j bb(1)(s = j)$.
Then $(partial f) / (partial theta_k) = bb(1)(s_t = k)$, giving:

$
  theta_k <- theta_k + epsilon [r_t + gamma accent(V, hat)^pi (s_(t+1)) - accent(V, hat)^pi (s_t)] bb(1)(s_t = k)
$

Only the parameter for the current state is updated, which is lookup-table TD(0).

= Question 2

For the linear approximator $f(s, bold(w)) = sum_k w_k phi_k (s)$, the partial
derivative is simply $(partial f) / (partial w_k) = phi_k (s_t)$.
Substituting into @td-general:

$
  w_k <- w_k + epsilon underbrace(
    [r_t + gamma accent(V, hat)^pi (s_(t+1)) - accent(V, hat)^pi (s_t)],
    delta_t,
  ) phi_k (s_t)
$

Each weight update is the product of two factors: a scalar TD error $delta_t$
(shared across all $k$) and the feature activation $phi_k (s_t)$. Features that
are active in the current state absorb the prediction error; inactive features
are unchanged. This enables generalisation: states sharing features transfer
learned value.

= Question 3

#figure(
  image("./figures/question-3/q3a_stim_reward.png", width: 75%),
  caption: [Stimulus and reward as functions of time.],
) <q3a>

#figure(
  image("./figures/question-3/q3b_td_learning.png", width: 75%),
  caption: [
    $accent(V, hat)(t)$, $Delta accent(V, hat)(t)$, and $delta(t)$ across
    21 sampled trials.
  ],
) <q3b>

*(c)* Early in learning $accent(V, hat)(t)$ is near zero everywhere. The TD
error $delta(t)$ is large and positive at reward time because the reward is
unexpected. As learning progresses, $accent(V, hat)(t)$ develops a ramp that
rises from stimulus onset to reward time: the agent learns that the stimulus
predicts reward. The temporal difference $Delta accent(V, hat)(t)$ shows where
the value function is increasing most steeply, producing a positive bump that
migrates backward from the reward toward the stimulus across trials. The TD
error $delta(t)$ correspondingly shifts: the positive peak at reward time
shrinks (reward becomes expected) while a new peak emerges at stimulus onset
(the stimulus itself becomes a surprising predictor of future value).

*(d)* The TD error $delta(t)$ corresponds to dopamine activity. This
agrees with the lectures: initially dopamine fires at reward
delivery; after conditioning, the dopamine response shifts to stimulus onset
while the reward-time response vanishes. In @q3b, we see exactly this
progression --- the $delta(t)$ peak migrates from $t = 20 "s"$ to $t = 10 "s"$
across trials.

= Question 4

#figure(
  image("./figures/question-4/q4a_boxcar_learning.png", width: 75%),
  caption: [
    Boxcar representation: $accent(V, hat)(t)$, $Delta accent(V, hat)(t)$,
    and $delta(t)$ across 21 sampled trials ($epsilon = 0.01$).
  ],
) <q4a>

*(b)* Compared with the tapped delay line (@q3b), the boxcar representation
produces a smoother value function because each feature integrates stimulus
presence over a range of past time steps, providing temporal generalisation. In
the first trial, $accent(V, hat)(t)$ and $delta(t)$ are identical to the
tapped case (weights start at zero). By the last trial, $accent(V, hat)(t)$
has converged to a smooth ramp rather than a sharp step, and $delta(t)$ shows
a cleaner shift with less residual noise. During intermediate trials the
convergence is faster in terms of the qualitative shape because boxcar features
share information across time lags.

*(c)* The $delta(t)$ panel in @q4a shows the dopamine shift: the reward-time
peak progressively disappears and is replaced by a stimulus-onset peak. This is
smoother than in Q3 because the boxcar features allow each weight update to
influence predictions at multiple time steps, so the shift happens more
gradually and completely within the 201 trials.

*(d)* The learning rate must be reduced ($epsilon = 0.01$ vs $0.2$) because
boxcar features are much larger in magnitude (they sum up to $tau + 1$
stimulus values rather than being 0 or 1). The effective step size is
$epsilon dot phi_k (s_t)$; with boxcar features $phi_k$ can be as large as the
memory span, so a large $epsilon$ causes destabilisation.
Reducing $epsilon$ compensates for the increased feature magnitude and
stabilises learning.

= Question 5

#figure(
  image("./figures/question-5/q5a_last100_by_type.png", width: 75%),
  caption: [
    Average $accent(V, hat)(t)$, $Delta accent(V, hat)(t)$, and $delta(t)$
    over the last 100 trials, split by rewarded, unrewarded, and all trials
    ($p = 0.5$).
  ],
) <q5a>

*(b)* The value $accent(V, hat)(t)$ and the temporal difference
$Delta accent(V, hat)(t)$ are _identical_ across rewarded, unrewarded, and all
trials because they depend only on the weights $bold(w)$ and the stimulus
(which is the same on every trial). The weights are shared across trial types,
so $accent(V, hat)$ does not distinguish whether reward will be delivered. Only
$delta(t)$ differs: at reward time ($t approx 20 "s"$), rewarded trials show a
positive $delta$ (reward exceeds expectation under $p = 0.5$) while unrewarded
trials show a negative $delta$ (omitted reward falls below expectation). At
stimulus onset all three averages coincide because the stimulus is identical.

*(c)* The TD error $delta(t)$ represents dopamine neuron activity. Key features
matched by this variable: (1) a positive response at stimulus onset
(conditioned stimulus predicts value), (2) a positive burst on rewarded trials
at reward time (reward exceeds the $p = 0.5$ expectation), and (3) a negative
dip on unrewarded trials at reward time (omitted reward is worse than
expected). This matches the classic pattern of dopamine responses in partial
reinforcement.

*(d)* Interpreting dopamine as "surprise" or "salience" would predict a
_positive_ dopamine response whenever something unexpected happens, regardless
of whether it is good or bad. However, on unrewarded trials the omission of expected reward is
surprising yet produces a _negative_ $delta$. The TD error is a _signed_
prediction error: it is positive when outcomes exceed expectations and negative
when they fall short. A pure surprise/salience account does not explain negative
dopamine dips.

#figure(
  image("./figures/question-5/q5e_da_nonlinearity.png", width: 75%),
  caption: [
    Average $delta(t)$ compared with the dopamine signal $"DA"(delta(t))$
    after applying the asymmetric nonlinearity.
  ],
) <q5e>

*(e)* The DA nonlinearity compresses both negative values (by $alpha = 6$) and
large positive values (by $beta = 6$ above $x^* = 0.27$). As a result, the
negative dip at reward omission time is much attenuated in $"DA"(delta)$
compared to $delta$ itself, while the positive peaks are also somewhat
compressed. This matches the limited dynamic range of real dopamine neurons:
they cannot fire below zero (floor effect) and saturate at high firing rates.
The average DA signal therefore appears more asymmetric than the raw TD error,
with a larger positive-to-negative ratio.

= Question 6

#figure(
  image("./figures/question-6/q6a_da_sweep.png", width: 75%),
  caption: [
    Average dopamine time course (last 100 trials) for each reward probability
    $p in {0, 0.25, 0.5, 0.75, 1}$.
  ],
) <q6a>

*(a)* At $p = 0$ there is no reward and hence no learning: the DA signal is flat
at zero. As $p$ increases, a stimulus-onset peak develops and grows --- higher
$p$ means the stimulus is a stronger predictor of reward. The reward-time
response is non-monotonic: it is absent at $p = 0$ and $p = 1$ (no surprise),
and largest at intermediate $p$ where reward delivery is most uncertain.
At $p = 1$ the stimulus-onset peak is maximal and the reward-time response
vanishes entirely because reward is fully predicted.

#figure(
  image("./figures/question-6/q6b_da_peaks.png", width: 75%),
  caption: [
    Dopamine level at the stimulus peak and at reward time as a function of
    reward probability $p$.
  ],
) <q6b>

*(b)* The stimulus-time DA peak increases monotonically with $p$: it scales
with the learned value of the stimulus, which is proportional to the expected
reward $p dot r$. The reward-time DA peak is _non-monotonic_ and resembles an
inverted-U. At $p = 0$ and $p = 1$ reward-time DA is near zero (no surprise);
it peaks at intermediate $p$ because the mismatch between actual reward (when
delivered) and expected reward $p dot r$ is largest there. 
Reward-time $delta$ on rewarded trials is $r - p dot r = (1 - p) r$, but these
occur only a fraction $p$ of the time, so the _average_ reward-time DA
reflects a mixture of positive and negative $delta$ values weighted by $p$ and
$1 - p$ respectively, passed through the nonlinearity.

*(c)* If dopamine encoded _uncertainty_ rather than prediction error, we would
expect the stimulus-time DA peak to follow an inverted-U shape (maximal at
$p = 0.5$, where outcome uncertainty is highest). Instead, @q6b shows
that the stimulus peak increases monotonically with $p$, inconsistent with an
uncertainty code.

= Question 7

*(a)* For a $d times d$ visual stimulus, the tapped delay line features are
$phi_(m n tau)(bold(S)) = S_(m n tau) = y_(m n)(t - tau)$. The boxcar
representation sums over the time lag:

$
  phi^square.stroked_(m n tau)(bold(S)) = sum_(u=0)^(tau) S_(m n u)
  = sum_(u=0)^(tau) y_(m n)(t - u)
$

Each boxcar feature detects the total presence of pixel $(m, n)$ over the most
recent $tau + 1$ time steps, providing temporal integration of visual input.

*(b.i)* Each feature detector
$phi^*_(i tau)(bold(S); bold(v)) = sum_(u=0)^(tau) h(sum_(m,n=1)^(d) v_(m n)^((i)) S_(m n u))$
first computes a weighted sum of the stimulus frame at lag $u$, using
spatial weights $v_(m n)^((i))$, then applies a nonlinearity $h(dot)$. The
weights ${v_(m n)^((i))}_(m,n=1)^d$ define a spatial receptive field for
feature $i$ --- they select which pixels
contribute to the feature's activation. The summation over lags $u = 0, dots, tau$ integrates the filtered
response over time, and $h(dot)$ introduces a nonlinear detection threshold.

*(b.ii)* With $a_(i u) = sum_(m,n=1)^d v_(m n)^((i)) S_(m n u)$ and
$phi^*_(i tau) = sum_(u=0)^tau h(a_(i u))$, the value function is
$f(bold(S), bold(theta)) = sum_(i tau) w_(i tau) phi^*_(i tau)$. The update
for $w_(i tau)$ follows @td-general directly:

$
  w_(i tau) <- w_(i tau) + epsilon delta_t phi^*_(i tau)(bold(S); bold(v))
$

For the spatial weights, applying the chain rule:

$
  (partial f) / (partial v_(m n)^((i))) = sum_(tau) w_(i tau) sum_(u=0)^(tau) h'(a_(i u)) S_(m n u)
$

so the update is:

$
  v_(m n)^((i)) <- v_(m n)^((i)) + epsilon delta_t sum_(tau) w_(i tau) sum_(u=0)^(tau) h'(a_(i u)) S_(m n u)
$

The $w$ update is identical in form to Q2 (linear in features). The $bold(v)$
update is different: it involves the derivative $h'$ and couples across time
lags, making it a nonlinear gradient descent step that adapts the receptive
fields.

*(b.iii)* With $h(x) = x$, we have $h'(x) = 1$ and
$phi^*_(i tau) = sum_(u=0)^tau a_(i u) = sum_(m,n) v_(m n)^((i)) phi^square.stroked_(m n tau)$.
The feature detectors become _linear_ combinations of boxcar features weighted
by spatial filters. The $bold(v)$ update simplifies to:

$
  v_(m n)^((i)) <- v_(m n)^((i)) + epsilon delta_t sum_(tau) w_(i tau) phi^square.stroked_(m n tau)
$

Advantage: the model is jointly linear in $bold(w)$ and $bold(v)$, keeping gradients simple. Disadvantage: without a
nonlinearity, different features $phi^*_i$ cannot capture non-overlapping
stimulus patterns --- two features with spatial weights $bold(v)^((i))$ and
$bold(v)^((j))$ that are both active for the same stimulus cannot be
selectively gated, limiting representational power.

*(b.iv)* In the TD framework, the spatial weights $bold(v)^((i))$ adapt to
maximise reward prediction. In the Henschke et al.~experiment, the
reward-associated grating (group iii) is paired with water reward, so TD
learning drives $bold(v)^((i))$ to better encode that orientation, increasing
the corresponding feature response. For groups without reward (i, ii), there is
no prediction error to drive $bold(v)$ updates, so V1 selectivity is
unchanged. The model predicts that reward-paired stimuli sharpen the spatial
filters that detect them, matching the elevated orientation selectivity observed
in the goal-directed VR group.

= Question 8

*(a)* In the standard linear TD model, the TD error
$delta(t) = r(t - Delta t) + Delta accent(V, hat)(t)$ is a _scalar_ shared
across all features. Each weight update
$w_k <- w_k + epsilon delta_t phi_k (s_t)$ depends on $delta$ and the
feature's own activation. A dopamine neuron that receives input from a specific
feature $k$ would modulate its response by $phi_k$, producing
feature-specific apparent dopamine responses even though $delta$ itself is
global. So standard TD can account for feature-specific responses if different
dopamine neurons receive input from different features, without requiring
multiple independent TD errors.

*(b)* Define the feature-specific value component
$accent(V, hat)^pi_k (s) = w_k phi_k (s)$ so that
$accent(V, hat)^pi (s) = sum_k accent(V, hat)^pi_k (s)$. A natural
feature-specific TD error is:

$
  delta_k (t) = r_k (t - Delta t) + gamma accent(V, hat)^pi_k (s_(t+1)) - accent(V, hat)^pi_k (s_t)
$

where $r_k$ is the portion of reward attributed to feature $k$. If we set
$r_k = r dot phi_k (s_t) slash (sum_j phi_j (s_t))$, then summing over $k$ recovers the global
TD error: $sum_k delta_k = delta$. The weight update
$w_k <- w_k + epsilon delta_k phi_k$ produces equivalent learning because
each $delta_k$ drives only its own weight. This model is backward compatible:
globally, $sum_k delta_k$ reproduces all the standard dopamine phenomena (shift
from reward to stimulus, partial reinforcement effects, inverted-U). Locally,
each $delta_k$ is modulated by feature $k$'s activation, producing the
feature-specific specialisation observed by Engelhard et al.~(2019): dopamine
neurons with different feature inputs would show different response profiles
while all contributing to a consistent global prediction error.

// -- Appendices --------------------------------------------------------------

#pagebreak()

= Appendix A: Code

== Scratchpad for TD algorithm:

#text(size: 9pt)[
  #raw(read("new.py"), lang: "python")
]


== Main experiment code:

#text(size: 9pt)[
  #raw(read("assignment.py"), lang: "python")
]

#pagebreak()

= Appendix B: AI Prompts

*GitHub Copilot* --- autocomplete during report writing, primarily for equation
typesetting.

*Claude Code* --- the following prompts were used while coding and drafting the
report:
- `read src/coursework/assignment.pdf and give me a tip for question 1`
- `give a hint for question 2`
- `walk me through example of value functions from reinforcement learning`
- `how do we go from this to TD learning?`
- `give a worked example`
- `give another worked example`
- `explain question 3`
- `explain in more detail`
- `in src/lib/assignment.py check the indexing for the sliding window in state S`
- `src/lib/assignment.py defines an experiment in code, add the plots needed by
  src/pdfs/assignment.pdf question 3 as publication quality matplotlib plots
  with monospace font, and save the plots to src/report/figures/question-3`
- `refine the report so far`
- `read the assignment and report and bugs in the code`
- `in src/lib/assignment.py properly parameterise run_td_tapped to support
  both tapped and boxcar representations.`
- `add the plots for question-4 in the usual way to src/lib/assignment.py`
- `now add in question-5, same as question-4 except reward\_probability=0.5,
  n\_trials=1000 and seed=11`
- `now for question-6 do a sweep of the reward probabilities of
  [0.0,0.25,0.5,0.75,1.0] and make the appropriate plots`
- `check all the code for adherence to src/assignment.pdf and fix any bugs`
- `improve the code quality for readability and concisness`
- `make the code neater`
- `give a hint for question 7`
- `explain in more detail` (ran multiple times)
- `help me with the maths`
- `give a hint for question 8`
- `explain in more detail` (ran multiple times)
- `help me with the maths`
- `review the report for correctness, and suggest improvements`
- `refine the report so far` (ran multiple times)
- `refine the report for coherence and conciseness`
- `make the report slightly more concise`
- `refine the report and code for publication, make it beautiful`