---
title: "Causal Whitespace Patching for Korean Byte-Latent Language Models: Quality-Preserving Latency and the Limits of Scale Amplification"
bibliography: references.bib
status: "anonymous ARR long-paper source"
date: 2026-08-17
abstract: |
  Byte-latent language models amortize global Transformer compute by grouping raw
  bytes into patches, but fewer patches need not preserve quality or reduce
  end-to-end latency. We test causal whitespace-informed patching on Korean in an
  identical byte-latent graph, separating boundary placement, patch rate, learned
  router cost, and cached autoregressive time-to-output. Calibration-only
  selection chose a 72-patch whitespace policy (W72). On a separately sealed
  32MB final stream and five model seeds, W72 was noninferior to an 86-patch
  codepoint baseline at +0.003682 bits per byte and improved on the same-rate
  codepoint policy by 0.010781. Five fresh-process Apple-MPS sessions measured
  2.628% controlled and 2.531% free-running latency reductions, but failed a
  predefined 10% efficiency threshold. A post-result random-weight curve rose
  from 3.572% at 49.8M parameters to 10.217% at 1.618B, exposing systems
  headroom without trained quality evidence. At 188.6M, unchanged W72 failed
  quality; one presealed W80 rescue restored noninferiority and reproduced only
  2.887% controlled and 2.475% free-running reductions. Thus the small effect
  replicates across two trained scales but does not amplify. Quality-constrained
  patch density and an unchanged byte-sequential local path prevent random-weight
  headroom from becoming an automatic trained-model speedup.
---

# Introduction

Tokenizer-free language models avoid a fixed subword vocabulary but must still
control the cost of long byte sequences. Byte Latent Transformer (BLT) applies
small local modules at byte resolution and an expensive global Transformer to
variable-length patches [@pagnoni-etal-2025-byte]. Learned next-byte entropy is
a powerful patching signal, but its predictor and full-byte scoring pass are
part of the real system. Deterministic linguistic boundaries are cheap, yet a
cheap detector need not identify useful information boundaries.

Korean offers a controlled case. Precomposed Hangul syllables typically occupy
three UTF-8 bytes, while observed spaces delimit *eojeol*, units that may contain
several morphemes [@moon-etal-2022-openkorpos]. This structure motivates a
whitespace policy, but it does not by itself establish a Korean-specific
efficiency method. Word-boundary pooling, deterministic sparse hierarchies, and
learned chunking already exist [@thawani-etal-2023-learn;
@videau-etal-2025-aunet; @hwang-etal-2025-hnet]. The unresolved question is
narrower: at an exactly matched patch rate, does moving a causal boundary toward
already observed whitespace improve quality, and does any saved global work
survive as matched-quality end-to-end latency?

We answer this with one shared byte-latent graph and an explicit separation of
four quantities: boundary placement, compression rate, detector cost, and
actual cached generation time. Our main contributions are:

1. a causal same-rate comparison between generic codepoint and
   whitespace-informed boundaries on Korean;
2. calibration-only policy selection and a separately sealed final stream,
   followed by exact physical-checkpoint authorization and independent loss
   replay;
3. detector-inclusive accounting and fresh-process, same-output incremental
   timing rather than inference from patch count or teacher-forced FLOPs; and
4. a trained-versus-random scale analysis showing why larger systems headroom
   does not automatically become a larger quality-qualified speedup.

The result is deliberately not advertised as a 10% efficiency technique. The
compact experiment finds a replicated same-rate quality effect and a stable
2.5--2.6% latency reduction. Random-weight systems headroom later exceeds 10%
at 1.618B parameters, but a one-seed, severely undertrained 188.6M bridge again
produces only 2.5--2.9% after relaxing patch density enough to recover quality.
The negative amplification result identifies the binding constraints: the
quality-feasible number of removed global events and a common byte-sequential
local path.

This distinction also changes how the scale question should be posed. A random
same-weight comparison asks how much runtime the schedule could save if
quality were irrelevant. A trained comparison asks how many of those global
events can actually be removed while staying inside a fixed loss margin. We do
not pool these estimands, fit one scaling curve through them, or use the random
endpoint to authorize a trained-model claim.

# Related Work

BLT uses an entropy model to create dynamically sized byte patches and reports
scaling to billions of parameters [@pagnoni-etal-2025-byte]. SpaceByte invokes
large blocks after a deterministic class of spacelike bytes
[@slagle2024spacebyte]. For UTF-8 Korean, its predicate also fires on multibyte
lead bytes, so authentic SpaceByte cadence is not whitespace-only. H-Net learns
hierarchical chunks end to end [@hwang-etal-2025-hnet], while AU-Net uses
regex-defined word boundaries in a deterministic hierarchy
[@videau-etal-2025-aunet]. Scratchpad Patching further shows that transient
within-patch computation can matter as much as the outer boundary rule
[@zheng-etal-2026-scratchpad]. These studies rule out novelty from merely using
a rule or activating a large trunk sparsely.

Learned and linguistically motivated sequence pooling have been compared before
[@nawrot-etal-2023-efficient]. Korean morphology-aware tokenization can improve
downstream encoders [@park-etal-2020-empirical; @lee-etal-2026-morpheme], but
that does not determine a causal byte-patch schedule. We therefore make no
claim that whitespace is a morphological oracle or that it generally beats
learned tokenizers. Our contribution is the controlled boundary-placement
contrast and its full systems accounting.

Recent byte and latent-token systems optimize other parts of this design space.
ByteFlow learns importance scores for static-graph compression
[@deng-etal-2026-byteflow], Bolmo converts pretrained subword models into
byte-latent models [@minixhofer-etal-2025-bolmo], and FLEXITOKENS adapts
compression rates across data [@owodunni-etal-2026-flexitokens]. These are
strong system references, but they do not supply the same prefix-causal,
same-graph whitespace-versus-codepoint intervention measured here.

# Causal Patch Policies

All main policies share the same local encoder, global Transformer, local
decoder, initialization, training order, and byte targets. They differ only in
the boundary schedule. A boundary decision at position $t$ may depend only on
the observed prefix. The implementation uses the Hugging Face BLT lag
convention: a dummy patch precedes data patches, and a data boundary changes the
decoder/global-state schedule before it changes the following local-encoder
group. Prefix-invariance tests verify that changing suffix bytes or suffix
boundaries cannot alter earlier logits.

We compare six policy families:

- **F**, fixed-length byte patches;
- **C**, a generic causal UTF-8 codepoint grid;
- **W**, the same grid after relocating eligible boundaries toward already
  observed ASCII whitespace;
- **S**, the authentic SpaceByte spacelike-byte cadence;
- **E**, a learned next-byte entropy threshold over all positions; and
- **EC**, the same entropy threshold restricted to codepoint boundaries.

C and W are constructed at 64, 72, or 86 data patches per 512-byte window. W
relocates a scheduled boundary to the nearest eligible observed whitespace
without changing the count, so W72 versus C72 isolates placement rather than
compression. It does not inspect a completed future word or invoke a
morphological analyzer. S retains its realized cadence rather than being
downsampled. E and EC use a 2.02M-parameter causal byte router, a 24-byte maximum
patch length, and one calibration-fitted threshold per seed. Router parameters,
training, scoring, memory, and incremental runtime remain in the corresponding
cost account.

# Experimental Design

## Data, models, and selection

We use the Korean Hangul-script shard of HPLT 3.0
[@oepen-etal-2025-hplt3]. The compact experiment trains on a deterministic
128M-byte stream and selects policies from an 8M-byte calibration stream. A
historical 16M-byte test stream was repeatedly exposed during development and
is never used for the final claim.

Before selection, a separate deterministic scan builds a 32,000,000-byte final
stream. It excludes all 6,911 historical documents under exact UTF-8 digest and
fixed NFKC/case/whitespace-normalized matching, then admits only unseen records
from a pre-existing stable test bucket. The stream contains 1,482 source
documents and 62,500 512-byte windows. An independent full-shard verifier
reconstructs its aggregate commitments. This proves exact and specified
format-normalized disjointness, not absence of partial or semantic
contamination.

Each compact model has 19,596,096 parameters: width 192 with two local encoder
and decoder layers, and width 384 with eight global layers. Float32 AdamW uses
batch size 32, 500 warmup updates, cosine decay from $3\times10^{-4}$ to
$3\times10^{-5}$, and one pass over 128M bytes. Three initial seeds (1,729,
2,718, 31,415) and two confirmation seeds (57,721, 65,537) share initialization
and shuffled-order hashes across policies.

Selection is reconstructed from checkpoints using calibration NLL only. The
first of 64 then 72 patches whose W candidate is within +0.010 bits per byte
(BPB) of C86 in the mean and at least two seeds is fixed. The broad reference is
the lowest calibration BPB among F/C/W/S/E/EC and selected-rate C under a fixed
tie order. A second process independently reloads all selected checkpoints and
reproduces every float32 sequence loss before the lock is issued. Test loss and
latency are unavailable to this decision.

The unit of quality is a predicted byte, not a patch. We sum causal next-byte
negative log likelihood over each 512-byte sequence and divide by the number of
predicted bytes and $\log 2$. Patch schedules therefore change internal global
positions without changing the quality denominator. For paired-seed inference,
each effect is formed within the same initialization and training order before
the five seed effects are summarized. For document inference, all eligible
windows from a source document stay in one bootstrap cluster. Windows that
cross document boundaries remain in the full-stream BPB point estimate but do
not become falsely independent bootstrap units.

Final W-minus-C86 noninferiority requires paired-seed and source-document
bootstrap 95% upper bounds below +0.010 BPB, at least four of five seeds within
the margin, and at least 95% document-cluster coverage. The same-rate mechanism
contrast requires W-minus-C mean at most -0.002 BPB, both upper bounds below
zero, and four of five negative seeds. A calibration futility rule may exclude
the locked broad comparator but may not replace it with a weaker model.

## Actual incremental inference

Only an exact checkpoint pair that passes final quality may be timed. Each of
five fresh-process Apple M4 Pro sessions measures all five model seeds, 64
distinct-document prompts, and two modes. Controlled replay uses a 128-byte
prompt followed by the same 128 held-out bytes. Strict-valid free running uses
the same prompt and greedy decoding until the first UTF-8 accept state after at
least 128 output bytes. Both roles must emit the specified controlled bytes or
their own independently replayed masked-greedy free bytes exactly.

Parallel prefill, cached consumes, selector/router work, cache updates, UTF-8
masking, argmax, DFA transitions, stop checks, and device synchronization are
inside end-to-end timing. Static DFA-mask compilation is outside. Five
repetitions are collapsed to a median within each
session--seed--prompt--role cell; a crossed session-by-seed-by-prompt bootstrap
never treats repetitions as independent. CPU sequential/full oracles and MPS
parallel/cache checks validate logits, probability-distribution distance,
boundaries, cache length, and every emitted byte before timing is eligible.

The original confirmatory threshold requires at least 10% aggregate reduction
in both controlled and free modes, a positive 95% lower bound, all five sessions
positive, at least three sessions at 10%, four positive model seeds, and median
seed reduction at 10%. We retain this deliberately demanding gate while also
reporting the measured effect and interval.

The primary latency statistic is a ratio of medians on paired cells, not the
mean of per-repetition speedups. Within each cell, the candidate and reference
share the prompt, model seed, session, mode, and repetition count. Session,
model-seed, and prompt indices are resampled independently and then applied to
both roles. This preserves the pairing while reflecting the three sources of
variation we can measure. Repetitions stabilize a cell median but never increase
the inferential sample size.

## Post-result scale diagnostics

After the compact result, a separate systems-only plan compares W72 and C86 on
the *same random weights* from 49.8M through 1.618B parameters. Three fresh
sessions at each target measure 16 controlled cases and three repetitions per
cell. This removes training quality from the contrast and estimates only
runtime headroom.

A trained bridge then fixes one 188,639,808-parameter initialization, identical
127,991,808-byte training order, and the compact optimizer for C86 and W72.
When W72 fails the unchanged +0.010 margin, the protocol permits exactly one
presealed density relaxation, W80; W82/W84 fallback is forbidden. W80 must pass
mean and contiguous-block-bootstrap noninferiority and a bitwise checkpoint
replay before five fresh controlled/free sessions open. Amplification requires
the larger point, and more strongly its lower bound, to exceed the corresponding
compact point. Because W72 and W80 differ, this is a density-adjusted
replication rather than a pure scale intervention.

## Provenance and result-use boundaries

The experimental record separates historical development from the later
analytic path. The exposed 16MB screen, its three-seed and five-seed summaries,
and early policy debugging remain public as development evidence. They do not
enter the selection function or the final intervals. Before any new final loss,
the following identities are committed in order: the final-stream seal, the
calibration-only selection plan, the initial physical-model identity, the first
checkpoint-forward calibration replay, the independently replayed selection
lock, confirmation completions, and the post-confirmation five-seed model
authorization. The final evaluator then opens every authorized physical bundle
on the one sealed stream, and the quality-lock process independently repeats
the full checkpoint forward before timing can begin.

Timing has its own later chain: an exact candidate--reference pair, cases,
output semantics, counters, statistical reduction, and 10% gate are fixed
before latency is read. Each accepted timing session runs in a fresh process and
publishes a tracked receipt whose hashes bind the heavier raw arrays and output
commitments. A subsequent session is not eligible until the preceding receipt
is committed. Correctness, power, thermal, process-isolation, or namespace
failure invalidates the whole session rather than a prompt subset.

Several pre-timing implementation failures are retained rather than erased. An
initial MPS near-tie caused a correctness stop, a later attempt began on battery,
one backend tolerance check failed while CPU semantics and greedy output
remained stable, and a dry run exposed an `mps`/`mps:0` identity bug. Each
revision changed only the corresponding pre-timing correctness guard; the
physical pair, cases, output horizon, statistic, and efficiency threshold did
not change. This history motivates our narrower claim of prospective local Git
sealing, not public preregistration or cryptographic one-shot execution.

# Results

## Compact quality and timing

The fixed 64-then-72 rule rejects W64 and selects W72. Authentic SpaceByte has
the best calibration BPB, but W72 is +0.103950 BPB worse and 0/3 seeds fall
within the broad margin. The broad branch therefore ends under the fixed
futility rule rather than substituting an easier comparator.

| Sealed-final contrast | Mean BPB | Paired 95% upper | Document 95% upper | Seed criterion | Result |
|---|---:|---:|---:|---:|---:|
| W72 $-$ C86 | +0.003682 | +0.004780 | +0.004612 | 5/5 within +0.010 | pass |
| W72 $-$ C72 | -0.010781 | -0.009868 | -0.010010 | 5/5 negative | pass |

: Sealed-final compact quality contrasts. Lower BPB is better. {#tbl:compact-quality}

The document bootstrap covers 61,019 of 62,500 windows (97.630%) from all 1,482
documents. W72 therefore preserves C86 quality and improves over C72 at the
same patch rate. The latter identifies a boundary-placement effect within this
graph; it does not identify Korean morphology as the cause.

The direction is not driven by one model seed. W72-minus-C86 is +0.004362,
+0.002436, +0.003124, +0.004533, and +0.003955 BPB over the five paired seeds.
W72-minus-C72 is -0.009659, -0.010605, -0.011478, -0.010761, and -0.011401.
Thus all seeds satisfy the noninferiority margin and all favor W72 in the
same-rate mechanism comparison. The paired 95% intervals are
[+0.002585, +0.004780] and [-0.011693, -0.009868], respectively.

W72 uses 72 rather than 86 data patches per window, a 16.279% reduction. Dense
matmul accounting falls from 6.153B to 5.640B FLOPs per window (8.332%), while
both policies retain the same 19.596M parameters and every byte-local consume.
These are workload counts, not latency estimates.

| Mode | C86 median | W72 median | Reduction | Crossed 95% interval | Positive sessions/seeds | 10% sessions |
|---|---:|---:|---:|---:|---:|---:|
| controlled | 370.816 ms | 361.070 ms | 2.628% | [2.026%, 3.526%] | 5/5; 5/5 | 0/5 |
| free running | 387.783 ms | 377.970 ms | 2.531% | [1.687%, 3.127%] | 5/5; 5/5 | 0/5 |

: Quality-authorized compact incremental inference. {#tbl:compact-actual}

All 16,000 free-running outputs pass independent strict-UTF-8, masked-greedy,
cache, and first-eligible-stop replay. Both intervals exclude zero, and every
session and seed favors W72, but both co-primary 10% gates fail. We therefore
describe a small reproducible effect, not a positive 10% inference technique.
At the controlled horizon W72 creates 36 patches versus C86's 43 while both
retain 127 byte-consume steps. An exploratory same-checkpoint crossover later
finds 2.84--2.85% schedule reductions under either model's weights, localizing
the effect to removed patch events rather than favorable candidate weights.

The effect also survives the temporal controls fixed for the systems test.
Controlled session-level reductions are 2.521%, 2.418%, 2.736%, 2.806%, and
2.675%; free-running reductions are 2.256%, 2.548%, 2.078%, 2.708%, and 2.191%.
Candidate-first and reference-first cells retain the same direction. TTFT
changes are only +0.157% controlled and -0.090% free with intervals crossing
zero, whereas decode-only changes are +2.789% and +2.540%. The result is a
decode-schedule effect rather than a prefill win.

Role-isolated parameter bytes and maximum MPS current/driver increments are
identical. Process-RSS differences are small and mixed across seeds, so memory
is descriptive and supports no reduction claim. The latency denominator is raw
output bytes and cases, never a policy-dependent token or patch count.

## Broader policy and robustness context

The initial three-seed development comparison prevents the same-rate result
from being misread as a universal policy ranking.

| Policy | Calibration BPB | Mean data patches | Learned auxiliary |
|---|---:|---:|---:|
| authentic SpaceByte S | **1.530750** | 153.3 | none |
| whitespace W86 | 1.621408 | 86.0 | none |
| codepoint C86 | 1.631042 | 86.0 | none |
| fixed-byte F86 | 1.636231 | 86.0 | none |
| entropy E | 1.638470 | near 86 | 2.02M router |
| constrained entropy EC | 1.643627 | near 86 | 2.02M router |

: Historical calibration context. It is not combined with sealed-final inference. {#tbl:policy-context}

S is substantially better in BPB but uses about 78.3% more global positions
than W86. It is the strongest quality reference, not a rate-matched efficiency
winner. The futility decision therefore excludes the broad final branch rather
than pretending C86 is the strongest raw-byte model. E and EC are worse than
W86 in this compact setup and add a 2,016,960-parameter router; this rejects the
tested router configuration, not learned routing in general. Because a complete
router-inclusive Phase 3 wall-time table was not authoritatively reconstructed,
we also do not claim a full six-policy Pareto frontier.

Two historical causal placebos have the same direction: W86 improves over a
whitespace-free delayed grid by 0.010308 BPB and over a rate-matched causal
rolling-hash grid by 0.020700, with all three seeds negative. These controls
make a simple phase shift or equally frequent arbitrary event insufficient
explanations, but they are not five-seed final comparisons. On a pinned Korean
Wikipedia stream, five-seed W86--C86 is -0.013711 BPB and stays below the fixed
+0.020 regression ceiling. This is a domain guard, not a contamination-free
benchmark or a population-level Korean result.

## Random systems headroom versus trained scale

The random-weight schedule contrast grows overall with graph size, although it
is not monotone. The 1.618B endpoint passes its separately fixed systems-only
10% point and 8% lower-bound gate. It supplies no trained quality evidence.

| Parameters | Random W72--C86 controlled reduction | 95% interval |
|---:|---:|---:|
| 49.8M | 3.572% | [2.771%, 4.502%] |
| 98.4M | 4.460% | [3.846%, 4.893%] |
| 188.6M | 7.218% | [3.868%, 8.934%] |
| 378.1M | 7.060% | [6.788%, 7.500%] |
| 790.4M | 8.714% | [8.284%, 8.948%] |
| 1.618B | **10.217%** | [9.104%, 10.987%] |

: Same-weight random-graph systems headroom. These rows do not establish trained quality. {#tbl:random-scale}

![Random-weight W72--C86 controlled systems headroom grows with graph size, while the two trained quality-qualified points remain near 2.5--2.9%. Random and trained curves answer different questions and are not fit as one scaling law.](figures/scale-headroom-versus-trained.png){#fig:scale-headroom width=100%}

The 188.6M trained result shows why the random curve cannot be promoted.
Unchanged W72 is +0.024200 BPB worse than C86 and fails before timing. The sole
W80 rescue reduces the event difference from 16.279% to 6.977% and passes:

| Policy | Calibration BPB | Difference from C86 | Block-bootstrap 95% interval | Result |
|---|---:|---:|---:|---:|
| C86 | 1.441126 | -- | -- | reference |
| W72 | 1.465327 | +0.024200 | -- | fail |
| W80 | 1.445184 | +0.004058 | [+0.003070, +0.005114] | pass |

: Trained 188.6M quality screen and the single presealed density rescue. {#tbl:trained-quality}

An independent process reproduces all 15,625 W80 losses bitwise. Five
fresh-process sessions then find 2.887% controlled reduction ([2.119%, 3.209%])
and 2.475% free-running reduction ([1.948%, 3.052%]); all 16 prompts and all
five sessions favor W80 in both modes. Controlled is only 0.259 percentage
points above the compact point and its lower bound is below that point. Free
running is 0.056 points below compact. Neither lower bound exceeds the compact
point, so the fixed amplification hypothesis fails.

Two additional scale diagnostics narrow the interpretation. A real float32
AdamW worker shows that the 1.618B graph fits below the fixed 75% MPS-memory
ceiling, but a 64MB training pair supplies only 0.04 source bytes per parameter;
we therefore stop before treating resource feasibility as language quality.
Separately concentrating 91.8% of 46.6M parameters in the global trunk produces
only 3.923% random-weight reduction ([3.247%, 4.310%]). Parameter share alone
does not reproduce the absolute cost of a large global event.

## Bottleneck localization

The whole-trial crossover attributes the compact effect to schedule rather than
weights. Candidate weights decode 2.852% faster under W72 than C86; reference
weights decode 2.842% faster under the same schedule swap, and every seed has
the same sign. During decode C86 creates 22 new patches and W72 creates 18. A
non-boundary byte step costs approximately 2.35--2.36ms in the synchronized
diagnostic, and boundary work adds approximately 2.51--2.56ms. Four removed
decode boundaries therefore predict about 10.2ms, matching the observed
same-checkpoint gaps of 10.1ms.

These component timings are not promoted to production shares because extra
synchronization changes kernel overlap. Their value is diagnostic: selector
construction is not hiding the gain, and all 127 cached byte consumes remain.
The compact model removes seven patch events over the full observed path but
only four during the decode portion, while the larger W80 intervention removes
three over the full path. The trained scale therefore increases global-event
cost while simultaneously reducing how many events quality allows us to
remove.

# Discussion

The results separate three quantities that a patch-count argument conflates.
First, a larger graph can make each removed global event more expensive: the
random curve is evidence for this systems headroom. Second, a trained model may
not tolerate the same event-removal rate: W72 fails at 188.6M, forcing W80 and
shrinking the available intervention. Third, both schedules retain the
byte-sequential local encoder/decoder, byte head, cache, UTF-8 state, and host
dispatch on every generated byte. These common costs impose an Amdahl bound.

A useful decomposition is therefore

$$
G_{\mathrm{E2E}} \approx
f_{\mathrm{removable}}\,
s_{\mathrm{global}}\,
q_{\mathrm{feasible}}.
$$

Random weights principally expose the middle factor. Matched-quality trained
inference requires all three. This explains why systems headroom crosses 10%
while two trained points remain near 2.5--2.9%.

The same-rate W72--C72 result is nevertheless scientifically useful. Because
the graph, initialization, training order, data, and number of global positions
match, it identifies the effect of relocating this implementation's global
schedule toward observed whitespace. It does not show that linguistic rules
beat learned routing in general, that whitespace is optimal Korean
segmentation, or that the result transfers to H-Net, CUDA, batched serving, or
chat workloads. Authentic SpaceByte remains the strongest calibration model,
and its failure to meet the futility margin explicitly prevents a strongest-raw
replacement claim.

The engineering implication is also narrower than “use spaces.” Further minor
boundary tuning cannot remove the common 127-step byte path. A materially larger
gain needs a new block/local-generation mechanism, evaluated against generic
UTF-8/scalar and generic multi-token controls at the same accepted-output and
quality contract. Frozen Hangul drafts, local thinning, vocabulary transfer,
and retrieval explored after the primary result did not satisfy those combined
requirements; we preserve them as supplemental negative diagnostics rather than
folding them into the method.

# Conclusion

Causal whitespace-informed relocation improves Korean byte-model quality over
a same-rate codepoint schedule and preserves quality against a denser C86
baseline. It also yields a small, repeatable 2.5--2.9% end-to-end reduction at
two trained scales. The predeclared 10% compact hypothesis and the later scale-
amplification hypothesis both fail. Random graphs reveal that global-event
headroom can exceed 10% at 1.618B parameters, but the larger trained model must
retain more patches and still executes the same per-byte local path. The
defensible contribution is therefore a controlled boundary-placement result,
detector-inclusive actual-inference evidence, and an empirical separation of
random systems headroom from trained quality-constrained speedup---not a new
scaling law or a production-ready efficiency technique.

# Limitations

The compact experiment has five training seeds, whereas the 188.6M extension
has one. Its five timing sessions estimate systems variability for one physical
checkpoint pair, not training variability. The larger graph sees only
127,991,808 source bytes (0.6785 bytes per parameter) and is severely
undertrained. Compact and larger candidates also use W72 and W80, respectively,
so they cannot identify a pure parameter-scale effect.

All primary timings use one Apple M4 Pro and batch-1-style incremental
generation. CUDA kernels, server batching, other sequence lengths, chat
templates, and production schedulers may change the bottleneck. The random
curve uses deterministic untrained weights and controlled replay only. It
isolates runtime geometry but says nothing about language quality or free-
running behavior.

HPLT is web-derived and may contain harmful, duplicated, copyrighted, or
personal material. Its catalogue labels the distributed Korean package CC0 and
identifies Common Crawl and the Internet Archive as sources, but places legal
compliance on users and maintains a takedown process
[@hplt-project-2026-catalogue]. We treat CC0 as package metadata rather than a
guarantee of rights in every underlying page. The repository redistributes no
corpus text and tracks no URLs, record IDs, raw outputs, or per-sequence losses.
We did not perform a content-level PII or offensive-content audit; hashing and
aggregate commitments are data-minimization measures, not anonymization of the
source corpus. The final stream's exact and fixed-normalized disjointness does
not rule out partial, near-duplicate, or semantic overlap. Korean Wikipedia is
used only as a domain diagnostic.

The historical 16MB test and some initial model outcomes were known before the
later provenance hardening. We disclose that chronology and exclude those
metrics from calibration-only selection and final confidence intervals. The
official final path is one prospectively Git-sealed analytic evaluation plus a
deterministic checkpoint-forward verification replay. Local Git ancestry and
no-clobber artifacts are not public preregistration, trusted execution, or proof
that an author could not delete an uncommitted run. The scale extension and
W80 rescue are explicitly post-result diagnostics whose rules were fixed before
their own candidate outcomes or timings.

We report model sizes, byte budgets, optimizer settings, final hardware, and
stage-specific elapsed times where authoritative receipts exist. We cannot
reconstruct a complete project-wide accelerator-hour total without false
precision: the historical record includes heterogeneous exploratory and
abandoned diagnostics, not all of which share one wall-time schema. This limits
compute-accounting completeness even though the final analytic stages remain
individually reproducible.

# Ethical Considerations

This work studies computational efficiency and does not evaluate people or make
decisions about individuals. Its main risk is overclaiming a hardware- and
language-specific diagnostic as a general efficient language model. We mitigate
that risk by publishing failed thresholds, comparator exclusions, exact claim
boundaries, aggregate evidence, and code paths, and by withholding a positive
efficient-model release because no such model was established. Corpus handling
is content-minimizing: tracked artifacts contain only aggregate commitments and
statistics. The intended use is research and reproducibility, not deployment or
commercial corpus redistribution.

AI assistants were used for code drafting, adversarial protocol review,
repository navigation, and language editing. They do not meet the authorship
criteria. Empirical claims were checked against tracked evidence or executable
tests, and the human authors retain responsibility for the design, analysis,
writing, and submission.
