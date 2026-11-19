# Qwen3.6-35B-A3B — origin of the Shift/Hand logit reversal

1023 prompt tokens, fixed 24 GiB on M4 Max, 40 resident layers, BF16 DFlash2. The measured target row follows the common token `**`. All numbers below are in raw logits unless labeled RMS.

## Changing the prefill geometry for selected operations

| Wide prefill operations | Hand | Shift | Shift − Hand |
|---|---:|---:|---:|
| None | 24.469723 | 24.436199 | -0.033524 |
| Full attention | 24.604971 | 24.597473 | -0.007498 |
| GDN | 24.436504 | 24.342674 | -0.093830 |
| MoE | 24.364460 | 24.567110 | +0.202650 |
| Full attention and GDN | 24.460363 | 24.586779 | +0.126415 |
| All | 24.418312 | 24.581171 | +0.162859 |

Widening only the MoE blocks is sufficient to change the second output token to Shift. Widening only full attention or only GDN keeps Hand first. Widening these two together also selects Shift with every MoE block narrow. Both routes can flip this prompt. The comparison isolates the whole MoE block, including activation normalization, routing and both expert branches.

Each segment runs the real kernels at the requested geometry on all prompt tokens. Its input comes from the preceding segment at the residual-stream boundary. Each segment supplies its output and the complete KV or recurrent buffers of the mixer layers it computes. In the final request assemble all 40 post-prefill states before the ordinary 7-row verifier. Pure narrow and pure wide segmented controls exactly reproduce all 248320 second-position logits. The MoE segment includes post-attention RMSNorm, routing, routed experts, shared expert and residual. A mixer segment includes its input RMSNorm, ordinary attention or GDN, and the residual. Wide/narrow names denote prefill geometry only.

## Replacing the post-prefill state before verification

| Narrow state inserted into the wide run | Shift − Hand |
|---|---:|
| None | +0.162859 |
| Attention KV | +0.253609 |
| GDN recurrent and convolution | -0.138680 |
| All | -0.033524 |

## Layer measurements

Switch the prefill in model order, starting with the narrow path. Each row gives the measured change in the final Shift − Hand margin when switching just that operation after its predecessors have become wide. All later operations retain the narrow geometry. These are target logits measured after DFlash2 verification, including downstream effects.

| Layer | Mixer | Prefill mixer Δ | Prefill MoE Δ | Margin after layer | Winner |
|---:|---|---:|---:|---:|---|
| 0 | GDN | -0.101160 | +0.205048 | +0.070364 | Shift |
| 1 | GDN | +0.056759 | +0.029631 | +0.156754 | Shift |
| 2 | GDN | -0.042755 | -0.016422 | +0.097576 | Shift |
| 3 | attention | +0.059256 | -0.081980 | +0.074852 | Shift |
| 4 | GDN | +0.028479 | -0.198545 | -0.095215 | Hand |
| 5 | GDN | +0.077581 | +0.063046 | +0.045412 | Shift |
| 6 | GDN | +0.019213 | +0.062157 | +0.126781 | Shift |
| 7 | attention | +0.025232 | +0.069674 | +0.221687 | Shift |
| 8 | GDN | -0.107069 | +0.093943 | +0.208561 | Shift |
| 9 | GDN | +0.023222 | +0.052443 | +0.284225 | Shift |
| 10 | GDN | +0.004789 | -0.025444 | +0.263571 | Shift |
| 11 | attention | -0.059156 | -0.006723 | +0.197691 | Shift |
| 12 | GDN | -0.005383 | +0.022846 | +0.215155 | Shift |
| 13 | GDN | -0.038561 | +0.116276 | +0.292870 | Shift |
| 14 | GDN | +0.021297 | -0.057608 | +0.256559 | Shift |
| 15 | attention | +0.050037 | -0.072344 | +0.234253 | Shift |
| 16 | GDN | +0.008171 | -0.000126 | +0.242298 | Shift |
| 17 | GDN | -0.008791 | -0.038363 | +0.195145 | Shift |
| 18 | GDN | +0.009647 | -0.002251 | +0.202541 | Shift |
| 19 | attention | -0.016428 | -0.003490 | +0.182623 | Shift |
| 20 | GDN | +0.000622 | +0.020931 | +0.204176 | Shift |
| 21 | GDN | -0.004372 | -0.000561 | +0.199244 | Shift |
| 22 | GDN | +0.008337 | -0.011705 | +0.195875 | Shift |
| 23 | attention | +0.004242 | -0.031979 | +0.168139 | Shift |
| 24 | GDN | -0.006973 | +0.014952 | +0.176117 | Shift |
| 25 | GDN | -0.000114 | +0.004856 | +0.180859 | Shift |
| 26 | GDN | -0.010847 | +0.012592 | +0.182604 | Shift |
| 27 | attention | -0.004305 | -0.000654 | +0.177645 | Shift |
| 28 | GDN | -0.003742 | +0.003664 | +0.177567 | Shift |
| 29 | GDN | +0.002802 | -0.003483 | +0.176886 | Shift |
| 30 | GDN | -0.004269 | -0.006794 | +0.165823 | Shift |
| 31 | attention | +0.003111 | +0.007298 | +0.176231 | Shift |
| 32 | GDN | -0.000317 | -0.005793 | +0.170122 | Shift |
| 33 | GDN | -0.000774 | +0.001003 | +0.170351 | Shift |
| 34 | GDN | +0.005684 | -0.013674 | +0.162361 | Shift |
| 35 | attention | +0.001217 | +0.000202 | +0.163780 | Shift |
| 36 | GDN | -0.000076 | -0.000462 | +0.163242 | Shift |
| 37 | GDN | +0.000212 | -0.000483 | +0.162971 | Shift |
| 38 | GDN | +0.000242 | -0.000980 | +0.162233 | Shift |
| 39 | attention | +0.000626 | +0.000000 | +0.162859 | Shift |

In this switching order the signed changes sum to +0.200697 for MoE, +0.063831 for full attention and -0.068146 for GDN. The winner changes at the MoE boundaries of layers 0, 4 and 5.

For each boundary, the true wide prefill supplies all 1023 residual vectors at that boundary. Use the narrow kernels for all following operations, letting them compute their own state, and restore only the prefill caches for already-wide mixers before the ordinary 7-row verifier. Each margin uses a fresh uncached prompt and the same initial output token. Consecutive differences telescope to the natural margin change; this decomposition depends on the input-to-output order of switching. The final two boundaries return the same full logit vector as the pure wide pass.

## Local numerical differences and readout at verification

Layers are zero-based. Prefill RMS columns compare narrow and wide on matching inputs at each operation. The signed verifier columns decompose the actual change in preference for Shift after the complete natural prefills. A positive value increases its margin.

| Layer | Mixer | Prefill mixer RMS | Prefill MoE RMS | Verifier mixer Δ | Verifier MoE Δ | Cumulative Δ |
|---:|---|---:|---:|---:|---:|---:|
| 0 | GDN | 0.0000036 | 0.0000471 | +0.000007 | +0.000007 | +0.000014 |
| 1 | GDN | 0.0000025 | 0.0000290 | -0.000019 | -0.000013 | -0.000018 |
| 2 | GDN | 0.0000039 | 0.0000389 | +0.000112 | -0.000315 | -0.000221 |
| 3 | attention | 0.0000205 | 0.0000346 | +0.000240 | -0.000076 | -0.000058 |
| 4 | GDN | 0.0000039 | 0.0000481 | -0.000311 | +0.000162 | -0.000207 |
| 5 | GDN | 0.0000044 | 0.0000735 | -0.000414 | -0.000086 | -0.000708 |
| 6 | GDN | 0.0000050 | 0.0000613 | -0.000029 | +0.000238 | -0.000499 |
| 7 | attention | 0.0000349 | 0.0001118 | +0.000228 | -0.000071 | -0.000342 |
| 8 | GDN | 0.0000053 | 0.0001070 | -0.000591 | -0.000968 | -0.001900 |
| 9 | GDN | 0.0000055 | 0.0001078 | +0.000408 | +0.000117 | -0.001375 |
| 10 | GDN | 0.0000109 | 0.0000657 | +0.000263 | +0.000807 | -0.000305 |
| 11 | attention | 0.0000531 | 0.0001403 | -0.001050 | +0.001223 | -0.000132 |
| 12 | GDN | 0.0000070 | 0.0001094 | +0.000882 | +0.000410 | +0.001160 |
| 13 | GDN | 0.0000066 | 0.0000939 | +0.000750 | -0.000401 | +0.001510 |
| 14 | GDN | 0.0000079 | 0.0001412 | +0.000609 | -0.000936 | +0.001182 |
| 15 | attention | 0.0000725 | 0.0001458 | +0.003409 | -0.002913 | +0.001678 |
| 16 | GDN | 0.0000084 | 0.0001959 | +0.001041 | -0.002117 | +0.000602 |
| 17 | GDN | 0.0000086 | 0.0002249 | -0.001295 | -0.000086 | -0.000779 |
| 18 | GDN | 0.0000083 | 0.0001805 | -0.002719 | +0.001442 | -0.002056 |
| 19 | attention | 0.0000776 | 0.0001487 | -0.020642 | +0.002681 | -0.020018 |
| 20 | GDN | 0.0000096 | 0.0002034 | +0.003047 | +0.000587 | -0.016383 |
| 21 | GDN | 0.0000093 | 0.0001878 | +0.001077 | +0.004998 | -0.010308 |
| 22 | GDN | 0.0000100 | 0.0001199 | +0.000372 | -0.004072 | -0.014007 |
| 23 | attention | 0.0000905 | 0.0001868 | -0.007284 | +0.003668 | -0.017623 |
| 24 | GDN | 0.0000091 | 0.0001853 | +0.005722 | +0.004154 | -0.007748 |
| 25 | GDN | 0.0000091 | 0.0001452 | +0.009933 | +0.004296 | +0.006481 |
| 26 | GDN | 0.0000105 | 0.0002317 | -0.006206 | +0.001948 | +0.002223 |
| 27 | attention | 0.0000916 | 0.0002521 | +0.007466 | +0.002879 | +0.012569 |
| 28 | GDN | 0.0000117 | 0.0002812 | -0.015097 | +0.043044 | +0.040516 |
| 29 | GDN | 0.0000124 | 0.0003792 | -0.016285 | -0.006210 | +0.018021 |
| 30 | GDN | 0.0000130 | 0.0004074 | -0.028498 | +0.000992 | -0.009485 |
| 31 | attention | 0.0001026 | 0.0002117 | -0.004934 | -0.005953 | -0.020373 |
| 32 | GDN | 0.0000154 | 0.0003601 | +0.021360 | +0.002352 | +0.003340 |
| 33 | GDN | 0.0000148 | 0.0004954 | +0.002637 | -0.000117 | +0.005860 |
| 34 | GDN | 0.0000179 | 0.0002839 | +0.004653 | +0.047483 | +0.057996 |
| 35 | attention | 0.0001660 | 0.0004365 | +0.192542 | -0.016319 | +0.234219 |
| 36 | GDN | 0.0000230 | 0.0004802 | -0.011737 | -0.045154 | +0.177328 |
| 37 | GDN | 0.0000228 | 0.0003073 | +0.149896 | -0.218653 | +0.108570 |
| 38 | GDN | 0.0000340 | 0.0003919 | +0.228710 | -0.010003 | +0.327278 |
| 39 | attention | 0.0003245 | 0.0000000 | -0.157391 | +0.026403 | +0.196290 |

Final RMS denominator correction: +0.0000934.

For target verification row zero, project the difference in each residual update onto (head_Shift - head_Hand) * final RMSNorm weight divided by the wide final RMS denominator. The identical embedding cancels. Add the reported narrow-to-wide final RMS denominator correction. Contributions telescope to the observed margin change within 0.000005 logits. These direct readout terms include propagated differences; they are not isolated prefill interventions.

Wide prefill probed once more, recording each residual boundary then replacing it with its natural narrow counterpart for all valid input positions. A local RMS includes operator arithmetic and, for recurrent/attention blocks, the effect of their own state across the prompt. The last layer is scored only at the final prompt position because the prefill paths can crop their output rows. RMS is over hidden features and reported prompt positions.

[Raw measurements and segment specifications](atomic-engine-qwen36-shift-layer-attribution-24gib.json).
