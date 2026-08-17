# JamoFlow

JamoFlow studies where byte-latent Korean language models should place global-compute boundaries, and whether a cheap boundary policy remains useful after its full systems cost is counted.

The current experiment does **not** implement a Jamo-level representation, Korean morphology engine, multi-Jamo decoding, or production generation speedup. It compares prefix-causal patch schedules in one Hugging Face BLT graph and separates boundary placement, patch rate, learned-router cost, Unicode robustness, and domain transfer.

## Current status

- Phase 0 corpus/routing audits and compact Phase 1–2 mechanism studies are complete.
- Phase 3 is a 19.6M-parameter compact study on deterministic HPLT 3.0 Korean streams. Its original 16MB `test` split is now explicitly development/screening evidence because early three- and five-seed results exposed it.
- A disjoint 32,000,000-byte Korean final stream has been constructed and independently verified without model-loss access. Rate/reference selection is calibration-only and must be reconstructed twice from exact physical checkpoints before the new final loss can be opened.
- Calibration-only selection fixed W72. On the sealed final stream, W72 was noninferior to C86 (+0.003682 BPB; both 95% upper bounds below +0.010) and improved on same-rate C72 by 0.010781 BPB in all five seeds. Relative to C86 it uses 16.28% fewer data patches and 8.33% fewer counted dense-matmul FLOPs; those analytical reductions are not a speed or memory result. Authentic SpaceByte was the strongest calibration model, but W72 failed the fixed broad-reference futility screen, so no best-raw-model replacement claim is made.
- Five-session actual-inference v5r3 is complete. W72 reduced controlled-replay end-to-end latency by 2.628% (crossed-bootstrap 95% CI [2.026%, 3.526%]) and strict-valid free-running latency by 2.531% ([1.687%, 3.127%]). All 5/5 sessions and 5/5 model seeds favored W72 in both modes, and all 16,000 generated outputs passed strict UTF-8, stopping, determinism, and Jamo-transition checks. However, 0/5 sessions reached the prospectively fixed 10% target, so the primary matched-quality actual-efficiency gate failed. This is a stable small speed effect and a primary-negative efficiency result, not a positive efficiency technique.
- Role-isolated memory was effectively equal in MPS current/driver increments and parameter bytes; process-RSS differences were small and mixed across seeds. Memory remains descriptive, with no improvement claim.
- The research pivoted from patch-boundary cadence alone to the byte-sequential local path. A post-result 2×2 checkpoint-by-schedule profile confirmed that W72's roughly 2.8% decode effect is a schedule effect: removing four decode boundary updates saves about 10.2ms, matching the measured 10.1ms gap. The shared 127-step local-byte path is the remaining bottleneck. Larger-scale/BPE/CUDA studies remain gated on a new compact candidate demonstrating meaningful matched-quality actual inference improvement.
- A calibration-only opportunity oracle found that 86.39% of complete bytes belong to precomposed Hangul syllables. Perfect Hangul-scalar blocks could remove 57.59% of sequential target calls and explain 98.68% of all scalar-grouping savings, so a learned-draft preflight is warranted. The suffix is not deterministic: continuation bytes retain 6.30 bits of joint uncertainty given the lead byte and 2.41 bits of conditional mutual information. The next model must learn a cheap joint/conditional draft; rules alone cannot emit the remainder.
- The frozen-W72 learned-draft preflight was negative. None of four approximately 40K-parameter heads passed the fixed acceptance/cost gate. Generic independent UTF-8 was strongest (24.38% complete-pair and 42.37% first-continuation acceptance); Hangul conditional reached 17.70%, and all three initializations trailed the independent control. Jamo/composition factorization is therefore not promoted. The only remaining multi-byte check is a target-side perfect-draft block-kernel upper bound: exact speculative verification can still commit a correction byte on mismatch, so its expected committed bytes are `2 + P(first accepted) + P(pair accepted)`, not merely the accepted suffix count. No new speed claim is made unless an exact measured block runtime passes.
- Static local/global reallocation reduced trained-model E2E latency by 22.8--24.3% but worsened calibration BPB by 0.0956, so it failed matched quality. A subsequent frozen-W72 conditional-local 2×2×2 screen also rejected every operator/component pair; even decoder-only second-MLP skipping at Hangul-prefix positions worsened BPB by 0.1988. Orthographic validity therefore did not identify computation-free local positions. At that stage the next step was a model-free audit of reversible Unicode-scalar/Hangul-factorized representations against generic scalar and BPE controls, not post-hoc skip-rate tuning.
- The later 8K dense/compositional quality study rejected the pure Hangul-codebook route: its document BPB was 0.20854 worse than dense BPE-2K, although true Hangul assignment beat a matched shuffle by 0.01149 BPB. The bottleneck was vocabulary cold-start plus codebook interference, not a missing hash variant.
- A sealed 2K→8K vocabulary-transfer probe then passed its development gate. Canonical merge-tree transfer with untied uniform-input/byte-weighted-output initialization reached 1.46572 BPB after 512 updates, 0.09099 better than its architecture-matched random control and 0.03605 above the dense-2K anchor. All 28 checkpoint NLL arrays were independently replayed bitwise. This is generic vocabulary-transfer evidence, not a Korean method or speed result.
- The sealed strong-baseline closure is complete. The untied EEVE initializer analogue was best at 1.45453 BPB (0.02487 above the dense-2K anchor); four untied roles passed, no tied role passed, Hangul-specific norm selection was negligible, and the compact 307+205 two-stage schedule hurt. All 54 checkpoint NLL arrays were independently replayed bitwise.
- The follow-up foldable-Jamo screen is also complete and primary-negative. True Jamo improved over no residual and over an exposure-matched shuffle, but its shuffle advantage was only 0.00031 BPB untied and 0.00083 tied, below the fixed 0.002 minimum; the same-cost generic multi-hash residual was about 0.0013 BPB better in both architectures. The Korean-specific residual branch is therefore stopped without threshold tuning.
- The separately sealed multi-hash mechanism screen also stopped the generic hash branch. The historical generic residual still beat its ordinary dense base by 0.01556 BPB, but it was 0.00799 BPB worse than an ordinary dense 8K control whose new input/output row updates were amplified by the audit-fixed factors. Balanced-random and exposure-stratified hash controls were also worse than that dense control, and surface assignment was unsupported. No fresh multi-hash, Korean-surface, or actual-inference stage is authorized. The strongest observation is instead a non-novelty-assumed optimizer control: the ordinary dense 8K role came within about 0.00132 BPB of the dense-2K anchor while retaining the previously measured roughly 20% random-weight latency headroom. A separate fresh-data protocol must compare it with standard joint AdamW and modern new-row-only/high-LR baselines before any trained speed claim.
- A disjoint fresh Korean vocabulary-adaptation stream was sealed and independently full-rescan verified: 128MB train and 8MB calibration, excluding all historical Phase-3 and sealed-final documents by exact and normalized identity. The one-seed screen was fixed before training: dense-2K continuation versus three untied dense-8K roles (ordinary joint, literature-aligned new-row-only→full CPT, and audit-fixed asymmetric update amplification), all on the same ordered 128MB stream. Model-free inventory gave 2,213 versus 1,677 optimizer steps; this 24.22% token-step opportunity was not treated as quality or actual-speed evidence.
- That fresh one-seed screen is complete and independently replayed bitwise. The audit-fixed dense-8K update geometry reached 1.38401 document BPB, 0.01022 below the continued dense-2K model, and beat ordinary dense-8K by 0.01835 BPB; ordinary dense-8K narrowly passed the fixed +0.010 noninferiority margin, while the compact new-row-only→full-CPT analogue failed at +0.03066 BPB. Geometry also reduced measured optimizer time by 35.36% versus 2K on the same raw stream, with a disclosed 27.99% parameter increase. This is a one-seed calibration result, not yet an inference or publication success; the selected trained checkpoints now require controlled and free-running actual E2E measurement.
- The exact trained-checkpoint actual preflight is now complete and primary-negative. Dense-8K update geometry reduced same-output controlled E2E by 20.131% (95% paired-prompt interval [16.524%, 22.723%], 64/64 prompts faster), but strict-valid free-running improved only 8.843% ([-0.168%, 16.685%], 44/64 prompts faster), below the prospectively fixed joint 10%/stability gate. All cache/full logits, greedy traces, UTF-8 outputs, and repetition determinism passed independent replay. The free-path effect was almost entirely explained by model-dependent generated-token counts (prompt-level correlation 0.9996), so the 8K multi-seed path is stopped without lowering the threshold. The next fail-fast moves once, to the pre-existing 16K systems Pareto point, and must repeat quality on a new disjoint Korean stream before any new actual timing.
- The sealed fresh-v2 16K quality fail-fast passed. Dense-16K update geometry reached 1.39347 document BPB, beating the fresh-v2 dense-2K anchor by 0.01486 BPB, the fixed 8K geometry anchor by 0.00441, ordinary 16K joint training by 0.02896, and the 16K two-stage control by 0.04564; all paired-document intervals passed their fixed gates. Five full checkpoint replays were bitwise identical. Relative to 2K it used 32.00% fewer optimizer steps and 28.53% less measured optimizer time, but 58.48% more parameters. This authorizes only a trained 16K actual-inference preflight: 2K is the primary expansion baseline, 8K is a mandatory secondary frontier comparator, and no inference or publication success is claimed yet.
- The trained fresh-v2 16K actual preflight is primary-negative despite a promising aggregate. Against 2K, controlled same-output E2E fell 24.925% and free-running fell 10.312% with a positive paired-prompt interval, but only 43/64 free prompts were faster versus the sealed 48/64 requirement. Against the mandatory 8K frontier, free-running regressed 8.292%. Independent 64-case checkpoint replay passed. Prompt-level latency and token-count effects correlated 0.99984, while canonical retokenization gaps were zero at the median, so further dense-vocabulary scaling is stopped; the next fail-fast tests an exact-output-preserving target-block upper bound on the trained 16K model.
- The trained 16K perfect-draft target-block upper bound passed its sealed target-only gate. Fixed-primary block 4 reduced controlled E2E by 63.927% (95% prompt-bootstrap 62.677--64.959%) and free E2E by 65.683% (63.061--66.521%), with 64/64 prompts faster in both modes and exact checkpoint replay/output across 1,280 free traces. Draft compute and imperfect acceptance/rollback remain excluded, so this is not a speculative-efficiency claim; it authorizes only a same-tokenizer draft fail-fast with a cheap generic n-gram/copy control.
- A 2026 literature refresh makes that control stricter: SSSD, UniSpec, DictSpec, Cacheback, and SAM-Decoding already cover prompt/self-output lookup, corpus n-grams, non-Latin dictionaries, hardware-aware draft sizing, and cache/suffix retrieval. The next primary is therefore a measured same-tokenizer hybrid of compact train-only token n-grams plus prompt/self-output fallback. It is a prior-work baseline, not a JamoFlow contribution; only a later cost-matched Korean-aware extension that beats it in actual E2E could support novelty.
- The train-only 16K retrieval baseline failed its prospectively sealed joint gate but exposed an important estimand split. Fixed-primary hybrid retrieval improved exact target-greedy free E2E by 26.244% (95% prompt-bootstrap 13.877--32.095%, 61/64 faster), yet controlled same-output improved only 5.310% (0.683--10.189%, 45/64), below the joint 10%/48-prompt gate. All full/cache/output checks passed. The failure was not relabeled and the cases were closed to tuning; it authorized only the mechanism audit recorded in the following bullets.
- That mechanism audit fixes one hypothesis before replay: free-path prompt-copy proposals made within a Hangul eojeol must accept at least 0.25 more tokens per cycle than proposals immediately after whitespace, with at least 32 cycles per stratum, 16 paired cases, and a positive paired-case bootstrap lower bound. It measures no latency and permits no secondary-feature fallback.
- The first mechanism-plan execution reached result serialization only after the checkpoint/event replay and counter checks, then failed because an insufficient-coverage branch left NumPy/non-finite scalars in canonical JSON. It published no result or aggregate. V1 is retained as invalidated history; V2 changes only fail-closed scalar normalization and keeps the hypothesis, cases, thresholds, and no-fallback rule exact.
- The corrected mechanism screen rejected the Hangul-boundary router hypothesis. Free hybrid prompt proposals accepted 1.406 tokens/cycle within a Hangul eojeol versus 1.752 immediately after whitespace; the paired-case contrast was -0.246, opposite the preregistered +0.25 direction, and only 13 cases contained both strata versus the required 16. The boundary router is closed without secondary fallback. A descriptive 508 Hangul-inside no-proposal cycles can motivate only a fresh novelty audit of eojeol completion, not a method or efficiency claim.
- The completed mechanism audit and a refreshed review including LinguaSpec, DictSpec, TokenTiming, OmniDraft, and LogitSpec closed broad linguistic-routing, Korean word-dictionary, and character-to-target-retokenization novelty. The resulting public-model replication on EXAONE 3.5 7.8B was negative: exact generic retrieval reduced target calls but increased free E2E latency by 14.938%, so retrieval and morphology extensions are closed.
- The first same-weight schedule sensitivity tested W72 against C86 on deterministic random 49.8M, 76.5M, and 98.4M BLT graphs. Controlled E2E fell by 3.572%, 3.758%, and 4.460%; every measured prompt and fresh-process session favored W72, but that stage's sealed 100M 10%/8%-lower/stability gate failed. Full model-state, cache, boundary, and MPS correctness replay passed.
- A later extension broadened the random-weight schedule curve to 188.6M--1.62B and then tested a trained 188.6M graph. Unchanged W72 failed quality at +0.02420 BPB versus C86, so it was never timed. The single presealed density-relaxed W80 rescue passed quality at +0.00406 BPB (block-bootstrap 95% upper +0.00511) and passed a full bitwise checkpoint replay.
- On five fresh-process trained-model sessions, W80 reduced controlled E2E by 2.887% (95% crossed interval [2.119%, 3.209%]) and strict-valid free-running E2E by 2.475% ([1.948%, 3.052%]); all 16/16 prompts and 5/5 sessions favored W80 in both modes. Controlled was descriptively 0.259 percentage points above the compact result, but its lower bound did not exceed the compact point; free running was 0.056 points below compact. The fixed scale-amplification hypothesis therefore failed.
- The revised campaign conclusion is a two-trained-scale boundary-placement and systems-limit result: quality-qualified whitespace schedules produce a small, consistent roughly 2.5--2.9% actual latency effect at 19.6M and 188.6M, while model size does not automatically amplify it. Random-weight headroom, patch/event reduction, or target-call reduction alone does not establish a useful trained inference technique. Code, aggregate evidence, corrections, and the audit trail are the release artifacts.
- No Korean-wide, production, general-hardware, or publication-scale efficiency claim is currently made.

Start with:

- [source-conversation and decision audit](docs/31-source-conversation-and-decision-audit.md)
- [Phase 3 confirmatory protocol](docs/22-phase3-confirmatory-protocol.md)
- [novelty and causal-identification audit](docs/28-novelty-and-identification-audit.md)
- [latest boundary-model literature amendment](docs/36-latest-boundary-literature-amendment.md)
- [publication comparator and Korean downstream protocol](docs/48-publication-comparator-and-downstream-protocol.md)
- [Korean downstream rendering and sealed-split addendum](docs/49-downstream-rendering-addendum.md)
- [Phase 3 initial Gate I result](docs/50-phase3-initial-results.md)
- [Phase 3 initial mechanism Gate M result](docs/51-phase3-initial-mechanism-results.md)
- [document-cluster inference correction](docs/52-document-cluster-inference-integrity-addendum.md)
- [selection and time-to-output correction](docs/53-selection-and-time-to-output-correction.md)
- [BPE prompt-boundary runtime addendum](docs/54-bpe-prompt-boundary-runtime-addendum.md)
- [ByteFlow causal/systems literature amendment](docs/55-byteflow-causal-literature-amendment.md)
- [execution concurrency and integrity policy](docs/56-execution-concurrency-and-integrity-policy.md)
- [actual-inference protocol integrity audit](docs/57-actual-inference-protocol-integrity-audit.md)
- [mechanism reanalysis authorization correction](docs/58-mechanism-reanalysis-authorization-correction.md)
- [locked comparator artifact-lineage correction](docs/59-locked-comparator-artifact-lineage-correction.md)
- [online tokenization and Korean inference-efficiency literature amendment](docs/60-online-tokenization-and-korean-efficiency-amendment.md)
- [dual-BPE sealed-test correction](docs/61-dual-bpe-sealed-test-correction.md)
- [learning-curve noninferiority correction](docs/62-learning-curve-noninferiority-correction.md)
- [BPE body-match correction](docs/63-bpe-body-match-correction.md)
- [family-aware scale-feasibility correction](docs/64-family-aware-scale-feasibility-correction.md)
- [publication evidence-identity correction](docs/68-publication-evidence-identity-correction.md)
- [publication comparator-role lock](docs/69-publication-comparator-role-lock.md)
- [downstream label-boundary correction](docs/70-downstream-label-boundary-correction.md)
- [contamination indexed-retrieval correction](docs/71-contamination-index-correction.md)
- [BPE token UTF-8 transition correction](docs/72-bpe-token-utf8-transition-correction.md)
- [raw-context matched-BPB correction](docs/73-raw-context-matched-bpb-correction.md)
- [publication runtime evidence correction](docs/74-publication-runtime-evidence-correction.md)
- [latest related-work revalidation](docs/75-latest-related-work-revalidation.md)
- [publication model-lock graph](docs/76-publication-model-lock-graph.md)
- [auxiliary-router and execution audit](docs/77-publication-auxiliary-router-and-execution-audit.md)
- [corrected initial evidence](docs/78-phase3-corrected-initial-evidence.md)
- [confirmation authorization lock](docs/79-phase3-confirmation-authorization-lock.md)
- [sealed-final and selection-v2 correction](docs/80-sealed-final-test-and-selection-v2-correction.md)
- [selected-reference confirmation v3](docs/81-selected-reference-confirmation-v2.md)
- [five-seed summary path correction](docs/82-five-seed-summary-path-correction.md)
- [historical five-seed confirmation results](docs/83-phase3-five-seed-confirmation-results.md)
- [sealed Korean final-test construction](docs/84-sealed-final-test-result.md)
- [inference selection-plan v2 seal](docs/85-inference-selection-plan-v2-seal.md)
- [publication actual-inference v5r3 protocol](docs/86-publication-actual-inference-v5-protocol.md)
- [actual-inference v5 correctness investigation](docs/87-actual-v5-free-path-correctness-debug.md)
- [paper claim–evidence matrix](docs/88-paper-claim-evidence-matrix.md)
- [Fable 5 interim review response and revised research direction](docs/89-fable5-interim-review-and-research-direction.md)
- [v5r3 summary counter-shape correction](docs/90-v5r3-summary-counter-shape-debug.md)
- [v5r3 actual-inference result and research pivot](docs/91-v5r3-actual-inference-result-and-research-pivot.md)
- [exploratory component-profile result and architecture decision](docs/93-exploratory-component-profile-result-and-architecture-decision.md)
- [Hangul block opportunity preflight](docs/94-hangul-block-opportunity-preflight.md)
- [Hangul block opportunity result](docs/95-hangul-block-opportunity-result.md)
- [Hangul draft acceptance preflight](docs/96-hangul-draft-acceptance-preflight.md)
- [Hangul draft result and systems cost-model correction](docs/97-hangul-draft-acceptance-result-and-cost-model-correction.md)
- [conditional-local frozen sensitivity result and representation pivot](docs/109-conditional-local-frozen-sensitivity-result-and-pivot.md)
- [8K compositional-head quality rejection and vocabulary-transfer pivot](docs/135-compositional-head-quality-result-and-transfer-pivot.md)
- [sealed vocabulary-transfer probe protocol](docs/136-vocabulary-transfer-probe-protocol.md)
- [vocabulary-transfer result and strong-baseline closure](docs/137-vocabulary-transfer-probe-result-and-baseline-closure.md)
- [strong vocabulary-transfer baseline protocol](docs/138-strong-vocabulary-transfer-baseline-protocol.md)
- [strong vocabulary-transfer baseline result and foldable-Jamo decision](docs/139-strong-vocabulary-transfer-baseline-result-and-foldable-jamo-decision.md)
- [sealed foldable-Jamo residual development protocol](docs/140-foldable-jamo-residual-protocol.md)
- [foldable-Jamo residual result and multi-hash vocabulary-adaptation pivot](docs/141-foldable-jamo-residual-result-and-multihash-pivot.md)
- [Fable 5 final retrospective and current research direction](docs/142-fable5-final-retrospective-and-current-direction.md)
- [foldable vocabulary reparameterization literature and novelty audit](docs/143-foldable-vocabulary-reparameterization-literature-audit.md)
- [foldable multi-hash AdamW first-update audit protocol](docs/144-foldable-multihash-update-audit-protocol.md)
- [foldable multi-hash update-audit v1 invalidation and v2 correction](docs/145-foldable-multihash-update-audit-v1-invalidation-and-v2-correction.md)
- [foldable multi-hash update-audit v2 invalidation and v3 correction](docs/146-foldable-multihash-update-audit-v2-invalidation-and-v3-correction.md)
- [foldable multi-hash update-audit v3 invalidation and v4 correction](docs/147-foldable-multihash-update-audit-v3-invalidation-and-v4-correction.md)
- [foldable multi-hash AdamW update-audit result](docs/148-foldable-multihash-update-audit-result-and-mechanism-decision.md)
- [foldable multi-hash mechanism-control protocol](docs/149-foldable-multihash-mechanism-control-protocol.md)
- [foldable multi-hash mechanism result and optimizer-only pivot](docs/150-foldable-multihash-mechanism-result-and-optimizer-pivot.md)
- [fresh Korean vocabulary-adaptation data protocol](docs/151-fresh-vocabulary-adaptation-data-protocol.md)
- [fresh Korean vocabulary-adaptation data result](docs/152-fresh-vocabulary-adaptation-data-result.md)
- [fresh Korean vocabulary-adaptation one-seed protocol](docs/153-fresh-vocabulary-adaptation-one-seed-protocol.md)
- [fresh Korean vocabulary-adaptation one-seed result](docs/154-fresh-vocabulary-adaptation-one-seed-result.md)
- [fresh vocabulary trained actual-inference preflight protocol](docs/155-fresh-vocabulary-trained-actual-preflight-protocol.md)
- [fresh vocabulary trained actual result and 16K pivot](docs/156-fresh-vocabulary-actual-result-and-16k-pivot.md)
- [fresh-v2 Korean data protocol for the 16K test](docs/157-fresh-v2-korean-data-protocol.md)
- [fresh-v2 Korean data result](docs/158-fresh-v2-korean-data-result.md)
- [fresh-v2 16K vocabulary quality fail-fast protocol](docs/159-fresh-vocabulary-16k-quality-protocol.md)
- [fresh-v2 16K vocabulary quality result](docs/160-fresh-v2-16k-quality-result.md)
- [fresh-v2 16K trained actual-inference protocol](docs/161-fresh-vocabulary-16k-trained-actual-protocol.md)
- [fresh-v2 16K trained actual result and target-block pivot](docs/162-fresh-v2-16k-trained-actual-result-and-block-pivot.md)
- [fresh-v2 16K target-block upper-bound protocol](docs/163-fresh-v2-16k-target-block-upper-bound-protocol.md)
- [fresh-v2 16K target-block upper-bound result](docs/164-fresh-v2-16k-target-block-upper-bound-result.md)
- [trained 16K retrieval-draft literature audit and fail-fast direction](docs/165-retrieval-draft-literature-audit-and-fail-fast-direction.md)
- [fresh-v2 16K retrieval-draft actual protocol](docs/166-fresh-v2-16k-retrieval-draft-actual-protocol.md)
- [fresh-v2 16K retrieval actual result and free-path correction](docs/167-fresh-v2-16k-retrieval-actual-result-and-free-path-correction.md)
- [fresh-v2 16K retrieval mechanism audit protocol](docs/168-fresh-v2-16k-retrieval-mechanism-audit-protocol.md)
- [retrieval mechanism v1 invalidation and v2 correction](docs/169-retrieval-mechanism-v1-invalidation-and-v2-correction.md)
- [retrieval mechanism result and boundary-router closure](docs/170-retrieval-mechanism-result-and-boundary-router-closure.md)
- [retrieval novelty closure and large-model replication direction](docs/171-retrieval-novelty-closure-and-large-model-replication-direction.md)
- [large-model retrieval compatibility preflight protocol](docs/172-large-model-retrieval-compatibility-preflight-protocol.md)
- [large-model preflight V1 invalidation and V2 correction](docs/173-large-model-preflight-v1-invalidation-and-v2-correction.md)
- [large-model preflight V2 invalidation and V3 oracle correction](docs/174-large-model-preflight-v2-invalidation-and-v3-oracle-correction.md)
- [large-model preflight V3 invalidation and V4 decision contract](docs/175-large-model-preflight-v3-invalidation-and-v4-decision-contract.md)
- [EXAONE 8B compatibility result and actual-stage decision](docs/176-exaone-8b-compatibility-result-and-actual-stage-decision.md)
- [EXAONE retrieval data and case protocol](docs/177-exaone-retrieval-data-and-case-protocol.md)
- [EXAONE retrieval data result and resource-calibration decision](docs/178-exaone-retrieval-data-result-and-resource-calibration-decision.md)
- [EXAONE baseline-only resource calibration protocol](docs/179-exaone-baseline-resource-calibration-protocol.md)
- [EXAONE resource calibration V1 invalidation and V2 correction](docs/180-exaone-resource-calibration-v1-invalidation-and-v2-correction.md)
- [EXAONE resource calibration V2 invalidation and V3 correction](docs/181-exaone-resource-calibration-v2-invalidation-and-v3-correction.md)
- [EXAONE resource calibration result and actual decision](docs/182-exaone-resource-calibration-result-and-actual-decision.md)
- [EXAONE 7.8B retrieval actual-inference protocol](docs/183-exaone-retrieval-actual-protocol.md)
- [EXAONE case-selection provenance correction](docs/184-exaone-case-selection-provenance-correction.md)
- [EXAONE retrieval actual result and core-scale decision](docs/185-exaone-retrieval-actual-result-and-core-scale-decision.md)
- [publication-scale W72/C86 schedule preflight](docs/186-scale-schedule-preflight-protocol.md)
- [schedule-scale result and terminal research decision](docs/187-scale-schedule-preflight-result-and-terminal-research-decision.md)
- [publication and release readiness](docs/188-publication-and-release-readiness.md)
- [post-100M schedule extrapolation protocol](docs/189-scale-schedule-extrapolation-protocol.md)
- [post-100M schedule extrapolation result](docs/190-scale-schedule-extrapolation-result-and-research-pivot.md)
- [large-scale training feasibility protocol](docs/191-large-scale-training-feasibility-protocol.md)
- [large-scale training feasibility result](docs/192-large-scale-training-feasibility-result-and-architecture-pivot.md)
- [global-heavy schedule bridge protocol](docs/193-global-heavy-schedule-bridge-protocol.md)
- [global-heavy result and trained-scale pivot](docs/194-global-heavy-result-and-trained-scale-pivot.md)
- [balanced 200M trained screen protocol](docs/195-balanced-200m-trained-screen-protocol.md)
- [balanced 200M quality-failure analysis protocol](docs/196-balanced-200m-quality-failure-analysis-protocol.md)
- [balanced 200M W72 quality failure and W80 pivot](docs/197-balanced-200m-quality-failure-result-and-w80-pivot.md)
- [balanced 200M W80 rescue protocol](docs/198-balanced-200m-w80-rescue-protocol.md)
- [balanced 200M W80 quality and actual-inference result](docs/199-balanced-200m-w80-quality-and-actual-result.md)
- [revised scale research direction](docs/200-revised-scale-research-direction.md)
- [ARR submission package and release decision](docs/201-arr-submission-package-and-release-decision.md)
- [quality-constrained scale frontier follow-up program](docs/202-quality-constrained-scale-frontier-program.md)
- [scalar representation and BPE opportunity protocol](docs/110-scalar-representation-and-bpe-opportunity-protocol.md)
- [scalar representation opportunity result and BPE constraint](docs/111-scalar-representation-opportunity-result.md)
- [scalar/Hangul-hybrid actual-runtime preflight protocol](docs/112-scalar-runtime-preflight-protocol.md)
- [scalar runtime result and token-frontier pivot](docs/113-scalar-runtime-preflight-result-and-token-frontier-pivot.md)
- [valid-output actual-inference correction](docs/65-valid-output-actual-inference-correction.md)
- [family parameter-identity correction](docs/66-family-parameter-identity-correction.md)
- [family time-projection correction](docs/67-family-time-projection-correction.md)
- [anonymous ARR long-paper source](paper/arr-submission.md)
- [paper build and submission instructions](paper/README.md)
- [extended audit manuscript](paper/draft.md)

`docs/00-topic-selection.md` and `docs/01-verification-report.md` are historical decision records. Their later errata and the audits above take precedence.

## Environment and tests

Phase 3 is pinned to Python 3.13, PyTorch 2.13.0, and Transformers 5.14.1.

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e '.[research]'
PYTHONPATH=src .venv/bin/pytest -q tests
```

The full-data command sequence, byte limits, seeds, gates, and conditional follow-up runs live in the Phase 3 protocol and its timestamped addenda. Smoke outputs are explicitly marked non-evidentiary and cannot be promoted by the full summarizers.

## Local audit

The Phase 0 reference implementation uses only the Python standard library.

```bash
PYTHONPATH=src python3 -m jamoflow audit docs/*.md \
  --format plain \
  --corpus-label "JamoFlow repository documents" \
  --interpretation-note "Repository research notes are not a representative Korean corpus." \
  --output-dir results/stage1-local
```

The generated report is a tooling smoke test when run on repository documents. It is not evidence about natural Korean corpora.

Directories can be restricted to Markdown without copying or changing source files:

```bash
PYTHONPATH=src python3 -m jamoflow audit /path/to/read-only-vault \
  --format plain \
  --plain-record-unit file \
  --include-suffix .md \
  --corpus-label "private Markdown convenience sample" \
  --output-dir results/private/vault-stage1
```

`results/private/` is ignored by Git. Only aggregate, non-content-bearing findings may be promoted into a tracked research note.

## Artifact and privacy boundary

Downloaded/processed corpora, checkpoints, patch matrices, raw timing samples, and per-sequence losses are written only to ignored `data/`, `artifacts/`, and `runs/` paths. Tracked `results/` contain validated aggregates and provenance, never corpus text.

The optional private Markdown ecology check is read-only and diagnostic. It may promote only aggregate counts and contrasts; paths, filenames, raw text, record hashes, prompts, and generated samples are forbidden.
