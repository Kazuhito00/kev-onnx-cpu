// Produce reference answers with open-jev itself (the implementation this
// repository's protocol was ported from), so the Python port can be checked
// against it rather than against my reading of the TypeScript.
import { writeFileSync } from "node:fs";
import { OpenJev, choice, score, noul } from "open-jev";

const CASES = [
  {
    id: "support_routing",
    state:
      "Hi, I was charged twice for my Pro subscription this month. Please refund the duplicate payment.",
    questions: [
      choice(
        "Which team should handle this request?",
        ["billing", "technical", "sales"],
        {
          billing: "charges, refunds, invoices",
          technical: "bugs and outages",
          sales: "pricing and upgrades",
        },
      ),
      noul("The customer is angry."),
      score("How urgent is this?", ["low", "medium", "high", "critical"]),
    ],
  },
  {
    id: "bug_triage",
    state:
      "The export button crashes in Safari but works in Chrome. Not blocking, we told users to switch browsers.",
    questions: [
      choice("How severe is this bug?", ["cosmetic", "degraded", "blocking"]),
      noul("This should be escalated immediately."),
    ],
  },
  {
    id: "incident",
    state:
      "URGENT: production database is down, all customers affected, CEO asking for updates every 5 minutes.",
    questions: [
      score("How urgent is this?", ["low", "medium", "high", "critical"]),
      noul("This is a security incident."),
      noul("The on-call engineer should be paged."),
    ],
  },
  {
    id: "japanese_state",
    state: "先週の納品が遅れた件、まだ連絡がありません。至急対応してください。",
    questions: [
      noul("The customer is angry."),
      score("How urgent is this?", ["low", "medium", "high"]),
    ],
  },
  {
    id: "many_options",
    state:
      "Can you send me the invoice for March? Also my login stopped working on the mobile app.",
    questions: [
      choice("What is the primary intent?", [
        "order_status",
        "cancel_subscription",
        "login_problem",
        "bug_report",
        "invoice_request",
        "other",
      ]),
      score("How many separate requests are in this message?", [
        "one",
        "two",
        "three or more",
      ]),
    ],
  },
];

const jev = await OpenJev.load({ model: "kev-0.6b", device: "cpu", dtype: "q4f16" });
console.log("loaded:", jev.runtime ?? "cpu");

const records = [];
for (const testCase of CASES) {
  const answers = await jev.decide(testCase.state, testCase.questions);
  records.push({
    id: testCase.id,
    state: testCase.state,
    questions: testCase.questions.map((q) => ({
      type: q.type,
      instructions: q.instructions,
      options: q.options ?? null,
      descriptions: q.descriptions ?? null,
    })),
    answers,
  });
  console.log(
    " ",
    testCase.id.padEnd(16),
    answers
      .map((a) => a.choice ?? a.level ?? (a.answer ? "yes" : "no"))
      .join(", "),
  );
}

writeFileSync("golden_kev.json", JSON.stringify({ model: "kev-0.6b", dtype: "q4f16", cases: records }, null, 1));
console.log("wrote golden_kev.json");
await jev.dispose?.();
