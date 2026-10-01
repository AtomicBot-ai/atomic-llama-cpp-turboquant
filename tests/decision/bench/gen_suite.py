#!/usr/bin/env python3
"""Write the llama-decision-bench suite (suite.jsonl next to this file).

Usage:
    python3 tests/decision/bench/gen_suite.py [--check BIN_DIR MODEL]

Groups (the "group" field; K1 is reported per group):
    systemone 1q        one question over a ~150-token support ticket (noul, choice)
    systemone 5q        five questions over the same ticket (noul, choice, score)
    router N=1/4/8      executor cards (~100 tokens each) + criterion + ~256-token task
    router N=1 t128     task length variants of N=1 (~128 and ~512 task tokens)
    router N=1 t512

Every text is fixed in this file, so the suite is byte-stable. With --check, the
task texts are tokenized with llama-laya-cli --tokenize and their token counts printed.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

TICKET = (
    "Subject: Charged twice for the annual plan\n\n"
    "Hi, I upgraded my team workspace to the annual Pro plan on September 3rd and my card was charged "
    "$1,188 twice within two minutes (receipts INV-20931 and INV-20932). The second charge is still pending "
    "on my bank statement. I only have one workspace and 12 seats. Please refund the duplicate charge as soon "
    "as possible - our finance team closes the month on Friday and this will show up as an unexplained expense. "
    "Also, the billing page now says my plan renews on September 3rd next year AND on September 4th, which looks "
    "wrong. I have been a customer for three years and this is the first time something like this has happened. "
    "Thanks, Dana (workspace: northwind-design)"
)

Q_NOUL = {"type": "noul", "instructions": "The customer asks for money back.",
          "criteria": {"true": "A refund or chargeback is requested.", "false": "No refund is requested."}}
Q_CHOICE = {"type": "choice", "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": "Payments, invoices, refunds and plan changes.",
                         "technical": "Bugs, outages and integration problems.",
                         "account": "Login, seats, permissions and workspace settings.",
                         "sales": "New purchases, quotes and upgrades before payment.",
                         "other": "Anything else."}}
Q_SCORE = {"type": "score", "instructions": "How urgent is this ticket?",
           "criteria": ["Not urgent: no deadline and no impact.", "Low: minor impact, can wait a week.",
                        "Medium: real impact, should be handled within two days.",
                        "High: money or work is blocked, handle today.", "Critical: outage or legal risk, handle now."]}
Q_ANGRY = {"type": "noul", "instructions": "The customer is angry or threatens to leave."}
Q_LANG = {"type": "choice", "instructions": "Which language should the reply use?",
          "criteria": ["English", "German", "French", "Spanish", "Other"]}

TASKS = {
    "invoice": (
        "Extract the invoice number, the invoice date and the total amount due from the document below. Return JSON "
        "with the keys invoice_number, invoice_date (ISO 8601) and total (a number, no currency sign).\n\n"
        "NORTHWIND TRADING - Hafenstrasse 12, Hamburg\n"
        "Invoice no. NW-2026-004817, dated 14 September 2026, due 14 October 2026.\n"
        "Bill to: Contoso Retail Ltd, Leeds, United Kingdom.\n"
        "Oak shelving units, twelve pieces, and thirty packs of steel wall brackets, with delivery and assembly "
        "at the Leeds store, less a five percent discount on the goods.\n"
        "Total due: EUR 2,474.85 (reverse charge, no VAT). Please pay by bank transfer and quote the invoice number. "
        "The previous balance of EUR 312.40 was settled on 2 September and is not part of this total. Questions go "
        "to the accounts team by email or phone during office hours. Goods remain our property until paid in full; "
        "late payments may be charged interest at the statutory rate."
    ),
    "meeting": (
        "Summarise the meeting notes below in at most five bullet points for people who missed the call. Keep every "
        "decision and every owner with a date; leave out small talk.\n\n"
        "Notes, product sync, Tuesday. Present: Priya (PM), Tom (backend), Aiko (design), Luis (support). Priya opened "
        "with the churn numbers: 4.1% monthly for small teams, up from 3.2% in July; most exits mention the confusing "
        "seat billing. Luis said support gets about thirty tickets a week about double charges when a team adds seats "
        "in the middle of a cycle. Tom explained that proration runs twice when the webhook retries; a fix is ready "
        "but needs a migration for old invoices. Decision: ship the webhook fix behind a flag on Monday, Tom owns it, "
        "and run the migration on the 28th after finance signs off. Aiko showed two mockups for the new billing page; "
        "the team picked the second one with the timeline of charges. Aiko will hand over final specs by Friday. Luis "
        "asked for a macro that explains proration; Priya will write it with him this week. Next sync moves to Thursday "
        "because of the offsite. Open question: whether to refund all affected teams automatically - Priya to check "
        "the cost with finance before the next sync."
    ),
    "code": (
        "Fix the bug in the Python function below and explain the fix in two sentences. The function must return the "
        "median of a list of numbers without changing the list the caller passed in, and must raise ValueError on an "
        "empty list.\n\n"
        "def median(values):\n"
        "    values.sort()\n"
        "    n = len(values)\n"
        "    if n == 0:\n"
        "        return None\n"
        "    mid = n / 2\n"
        "    if n % 2 == 1:\n"
        "        return values[mid]\n"
        "    return (values[mid - 1] + values[mid]) / 2\n\n"
        "Our test suite calls median([3, 1, 2]) and expects 2, median([4, 1, 3, 2]) and expects 2.5, and checks that "
        "the input list is unchanged afterwards. It also calls median([]) inside pytest.raises(ValueError). Right now "
        "the first call fails with 'TypeError: list indices must be integers or slices, not float', the input list "
        "comes back sorted, and the empty case returns None instead of raising. Keep the function signature, do not "
        "import numpy or statistics, and keep it readable for a junior developer who will maintain it. The code runs "
        "on Python 3.9, so do not use syntax that needs a newer version."
    ),
    "sql": (
        "Write one PostgreSQL query for the question below and say which index would help it most.\n\n"
        "Tables: orders(id bigint primary key, customer_id bigint, created_at timestamptz, status text, total_cents "
        "integer), customers(id bigint primary key, country text, signup_at timestamptz, plan text), refunds(id bigint, "
        "order_id bigint, amount_cents integer, created_at timestamptz). Status is one of 'paid', 'cancelled', "
        "'pending'. A refund can be partial and an order can have several refunds.\n\n"
        "Question: for each country, what was the net revenue in euros (paid order totals minus all refunds of those "
        "orders) in the third quarter of 2026, counting only customers who signed up before 2026 and are on the 'team' "
        "or 'business' plan? Order the result by net revenue, highest first, and show the number of distinct paying "
        "customers per country next to it. Countries with no paid orders in the quarter should not appear. The orders "
        "table has about 40 million rows and the refunds table about 900 thousand, so avoid joins that multiply order "
        "rows by their refunds before summing, and do not use temporary tables."
    ),
}

CARDS = [
    ("local/qwen3.5-4b-q4_k_m", {
        "name": "Qwen3.5-4B local", "kind": "local", "description": "4B general model on this laptop (Q4_K_M)",
        "checks": [
            {"skill": "field extraction", "status": "measured", "passed": 188, "total": 200, "criterion": "exact match", "source": "atomic-evals/extract", "version": "2026-09"},
            {"skill": "meeting summary", "status": "measured", "passed": 141, "total": 180, "criterion": "all decisions kept", "source": "atomic-evals/summary", "version": "2026-09"},
            {"skill": "python bug fix", "status": "measured", "passed": 97, "total": 164, "criterion": "hidden tests pass", "source": "atomic-evals/pyfix", "version": "2026-08"},
            {"skill": "long-document QA", "status": "missing", "source": "atomic-evals/longqa", "version": "2026-09"}]}),
    ("local/qwen3.5-9b-q4_k_m", {
        "name": "Qwen3.5-9B local", "kind": "local", "description": "9B general model on this laptop (Q4_K_M), slower",
        "checks": [
            {"skill": "field extraction", "status": "measured", "passed": 194, "total": 200, "criterion": "exact match", "source": "atomic-evals/extract", "version": "2026-09"},
            {"skill": "meeting summary", "status": "measured", "passed": 158, "total": 180, "criterion": "all decisions kept", "source": "atomic-evals/summary", "version": "2026-09"},
            {"skill": "SQL writing", "status": "measured", "passed": 121, "total": 150, "criterion": "same result set", "source": "atomic-evals/sql", "version": "2026-09"}]}),
    ("local/gemma-3-4b-q4_0", {
        "name": "Gemma 3 4B local", "kind": "local", "description": "4B multilingual model, QAT Q4_0",
        "checks": [
            {"skill": "field extraction", "status": "measured", "passed": 171, "total": 200, "criterion": "exact match", "source": "atomic-evals/extract", "version": "2026-09"},
            {"skill": "translation", "status": "measured", "passed": 176, "total": 200, "criterion": "judge score at least 4 of 5", "source": "atomic-evals/translate", "version": "2026-08"},
            {"skill": "python bug fix", "status": "missing", "source": "atomic-evals/pyfix", "version": "2026-08"}]}),
    ("local/phi-4-mini-q8_0", {
        "name": "Phi-4-mini local", "kind": "local", "description": "3.8B reasoning-tuned model (Q8_0)",
        "checks": [
            {"skill": "python bug fix", "status": "measured", "passed": 118, "total": 164, "criterion": "hidden tests pass", "source": "atomic-evals/pyfix", "version": "2026-08"},
            {"skill": "SQL writing", "status": "measured", "passed": 102, "total": 150, "criterion": "same result set", "source": "atomic-evals/sql", "version": "2026-09"},
            {"skill": "meeting summary", "status": "measured", "passed": 119, "total": 180, "criterion": "all decisions kept", "source": "atomic-evals/summary", "version": "2026-09"}]}),
    ("cloud/large-reasoning", {
        "name": "Large reasoning model (API)", "kind": "cloud", "description": "frontier model behind a paid API, 2-10 s latency",
        "checks": [
            {"skill": "field extraction", "status": "measured", "passed": 198, "total": 200, "criterion": "exact match", "source": "atomic-evals/extract", "version": "2026-09"},
            {"skill": "python bug fix", "status": "measured", "passed": 157, "total": 164, "criterion": "hidden tests pass", "source": "atomic-evals/pyfix", "version": "2026-08"},
            {"skill": "SQL writing", "status": "measured", "passed": 144, "total": 150, "criterion": "same result set", "source": "atomic-evals/sql", "version": "2026-09"}]}),
    ("cloud/fast-general", {
        "name": "Fast general model (API)", "kind": "cloud", "description": "small hosted model, cheap, under 1 s",
        "checks": [
            {"skill": "field extraction", "status": "measured", "passed": 183, "total": 200, "criterion": "exact match", "source": "atomic-evals/extract", "version": "2026-09"},
            {"skill": "meeting summary", "status": "measured", "passed": 150, "total": 180, "criterion": "all decisions kept", "source": "atomic-evals/summary", "version": "2026-09"},
            {"skill": "long-document QA", "status": "missing", "source": "atomic-evals/longqa", "version": "2026-09"}]}),
    ("cloud/code-specialist", {
        "name": "Code model (API)", "kind": "cloud", "description": "hosted coding model",
        "checks": [
            {"skill": "python bug fix", "status": "measured", "passed": 151, "total": 164, "criterion": "hidden tests pass", "source": "atomic-evals/pyfix", "version": "2026-08"},
            {"skill": "SQL writing", "status": "measured", "passed": 139, "total": 150, "criterion": "same result set", "source": "atomic-evals/sql", "version": "2026-09"},
            {"skill": "meeting summary", "status": "missing", "source": "atomic-evals/summary", "version": "2026-09"}]}),
    ("local/llama-3.2-3b-q4_k_m", {
        "name": "Llama 3.2 3B local", "kind": "local", "description": "3B instruct model (Q4_K_M), fastest local option",
        "checks": [
            {"skill": "field extraction", "status": "measured", "passed": 159, "total": 200, "criterion": "exact match", "source": "atomic-evals/extract", "version": "2026-09"},
            {"skill": "meeting summary", "status": "measured", "passed": 104, "total": 180, "criterion": "all decisions kept", "source": "atomic-evals/summary", "version": "2026-09"},
            {"skill": "python bug fix", "status": "measured", "passed": 61, "total": 164, "criterion": "hidden tests pass", "source": "atomic-evals/pyfix", "version": "2026-08"}]}),
]

CRITERIA = {
    "invoice": "All three fields exact; nothing invented.",
    "meeting": "Every decision and owner with its date is kept; at most five bullets.",
    "code": "The hidden tests pass and the input list is not changed.",
    "sql": "The query returns the same rows as the reference query.",
}


def card(i):
    cid, c = CARDS[i % len(CARDS)]
    return {"id": cid, "card": {"schema": "atomic.executor-card/1", **c}}


def router(task_key, n, task=None, first=0):
    return {"task": task if task is not None else TASKS[task_key], "criterion": CRITERIA[task_key],
            "candidates": [card(first + i) for i in range(n)]}


def half(text):
    # about half of the words, cut at a sentence end
    cut = text.rfind(". ", 0, len(text) // 2)
    return text[: cut + 1] if cut > 0 else text[: len(text) // 2]


def suite():
    rows = [
        ("so-1q-noul", "systemone", "systemone 1q", {"state": TICKET, "questions": {"refund": Q_NOUL}}),
        ("so-1q-choice", "systemone", "systemone 1q", {"state": TICKET, "questions": {"team": Q_CHOICE}}),
        ("so-5q-a", "systemone", "systemone 5q",
         {"state": TICKET, "questions": {"refund": Q_NOUL, "team": Q_CHOICE, "urgency": Q_SCORE, "angry": Q_ANGRY, "language": Q_LANG}}),
        ("so-5q-b", "systemone", "systemone 5q",
         {"state": {"channel": "email", "customer": {"plan": "pro", "seats": 12, "tenure_years": 3}, "message": TICKET},
          "questions": {"urgency": Q_SCORE, "team": Q_CHOICE, "refund": Q_NOUL, "language": Q_LANG, "angry": Q_ANGRY}}),
        ("rt-n1-invoice", "router", "router N=1", router("invoice", 1)),
        ("rt-n1-code", "router", "router N=1", router("code", 1, first=3)),
        ("rt-n4-meeting", "router", "router N=4", router("meeting", 4)),
        ("rt-n4-sql", "router", "router N=4", router("sql", 4, first=4)),
        ("rt-n8-invoice", "router", "router N=8", router("invoice", 8)),
        ("rt-n8-code", "router", "router N=8", router("code", 8, first=2)),
        ("rt-n1-t128", "router", "router N=1 t128", router("meeting", 1, task=half(TASKS["meeting"]))),
        ("rt-n1-t512", "router", "router N=1 t512", router("meeting", 1, task=TASKS["meeting"] + "\n\n" + TASKS["sql"])),
    ]
    return [{"id": i, "endpoint": e, "group": g, "body": b} for i, e, g, b in rows]


def check(bin_dir, model):
    texts = [TICKET] + list(TASKS.values()) + [half(TASKS["meeting"]), TASKS["meeting"] + "\n\n" + TASKS["sql"]]
    names = ["ticket"] + list(TASKS) + ["meeting/2", "meeting+sql"]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    out = subprocess.run([os.path.join(bin_dir, "llama-laya-cli"), "-m", model, "--tokenize", f.name],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, env=dict(os.environ, LC_ALL="C"))
    os.unlink(f.name)
    for name, line in zip(names, out.stdout.splitlines()):
        print("%-12s %4d tokens" % (name, len(json.loads(line))), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", nargs=2, metavar=("BIN_DIR", "MODEL"), help="print token counts of the texts")
    args = ap.parse_args()
    rows = suite()
    path = os.path.join(HERE, "suite.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %s (%d requests)" % (os.path.relpath(path), len(rows)), flush=True)
    if args.check:
        check(*args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
