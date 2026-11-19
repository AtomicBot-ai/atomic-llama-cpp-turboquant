# Qwen3.6 fast prefill can change greedy answers

**Status** — Open. Fast prefill is the default for the qualified 8–24 GiB
fixed-budget Metal profile. Use the previous profile when reproducing an
answer generated with that configuration.

## Observed behavior

With a 1023-token handover input at 24 GiB, both paths emit the same first
token. At the next position the narrow path prefers `Hand` and the wide path
prefers `Shift`. With all 40 expert layers resident and the same BF16 DFlash2
verifier, the target logit difference (Shift minus Hand) moves from −0.033524
to +0.162859. Temperature zero selects the highest target logit in each run.
Widening only the MoE operations is sufficient to reverse the ranking on
this input. Widening attention and GDN together can also reverse it.
[Full-vocabulary comparison](atomic-engine-qwen36-shift-logits-24gib.json)
and [layer attribution](atomic-engine-qwen36-shift-layer-attribution-24gib.md).

In a separate comparison of six prompts run to EOS, both production profiles
identified the three exceptional handover batches, their actions and owners.
Both 912-token answers mishandled the outstanding donor consent in their
summary. In the invoice task both computed EUR 195 payable, but the fast
answer attributed a EUR 60 discrepancy to a cancelled EUR 65 visit without
support in the input. Total elapsed time across the six prompts fell from
48.56 s to 33.02 s. These cases establish individual factual errors. They
do not estimate their rate on other prompts.
[Inputs, complete answers and quality criteria](atomic-engine-qwen36-complete-quality-24gib.json).

## Reproduce the user-visible difference

Start the rebuilt server with the BF16 draft alongside the standard target:

```sh
python3 tools/expert-source/run-qwen35-metal-profile.py 24
```

From another terminal, send the frozen input without retokenizing it:

```sh
jq '{prompt: .inputs.ids.edge, temperature: 0, seed: 42, n_predict: 64, cache_prompt: false, return_tokens: true}' \
  docs/atomic-engine-qwen36-complete-quality-24gib.json | \
  curl -sS -H 'Content-Type: application/json' --data-binary @- http://127.0.0.1:8080/completion
```

Restart with `--no-fast-prefill` and repeat the request. These two shipping
profiles use 40 and 36 resident layers respectively. The linked attribution
experiment holds residency and the verifier fixed to isolate prefill.
For the native server with the same model and launch arguments, set
`LLAMA_QWEN35_FAST_PREFILL=0` before startup to select the previous profile.
