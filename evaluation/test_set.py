"""
evaluation/test_set.py
-----------------------
Hand-written test cases for RAGAS evaluation.

Each case has:
  question      — the natural language query to run through the pipeline
  ground_truth  — the correct, complete answer based on the sample documents
  category      — what type of retrieval challenge this case represents
  note          — why this case is in the set (what failure mode it catches)

Design principles:
  - Every test case is directly verifiable against the 4 sample documents
  - Categories span all document types: PDF, CSV (invoices, employees, sales)
  - Includes one negative-evidence case (answer IS in context but is "none")
  - No out-of-scope queries here — the scope guard is tested separately in
    ragas_eval.py and those rows are excluded from RAGAS scoring

Category guide:
  pdf_direct    — specific fact from service_contract_nexus_solutions.pdf
  csv_keyword   — CSV query where a proper name gives BM25 a keyword boost
  csv_abstract  — CSV query with no specific names/IDs (harder retrieval)
  cross_doc     — requires combining information from 2+ documents
  neg_evidence  — answer IS findable but the conclusion is "none / not found"
"""

TEST_CASES = [
    # ── PDF direct (expect high scores — narrative text, specific answers) ─────

    {
        "question": "What is the total contract value with Nexus Solutions?",
        "ground_truth": (
            "The total contract value with Nexus Solutions is USD 96,000, "
            "payable in three equal milestones of USD 32,000 each via wire transfer."
        ),
        "category": "pdf_direct",
        "note": "Single-fact lookup from the PDF contract. Baseline — should score well on all 4 metrics.",
    },
    {
        "question": "Who signed the service contract with Nexus Solutions and what is their role?",
        "ground_truth": (
            "The service contract with Nexus Solutions was signed by James Okafor, "
            "whose role is Engineering Manager."
        ),
        "category": "pdf_direct",
        "note": "Cross-verifiable: James Okafor also appears in employees.csv. "
                "Tests whether the answer cites the PDF rather than the employee record.",
    },
    {
        "question": "What are the payment terms and late payment penalties in the service contract?",
        "ground_truth": (
            "The contract is payable in three milestones of USD 32,000 each: "
            "Milestone 1 due February 16 2024 (design approval), "
            "Milestone 2 due April 16 2024 (backend and mobile app), "
            "Milestone 3 due July 16 2024 (QA, deployment, and handover). "
            "Late payments beyond 15 days incur a penalty of 1.5% per month "
            "on the outstanding amount."
        ),
        "category": "pdf_direct",
        "note": "Multi-part answer spanning both chunks of the PDF. "
                "Tests context recall — both chunks need to be retrieved.",
    },

    # ── CSV keyword (name gives BM25 a strong keyword hit) ────────────────────

    {
        "question": "What did Michael Torres sell in Q1 2024?",
        "ground_truth": (
            "In Q1 2024, Michael Torres sold: "
            "January — 42 units of Analytics Pro (North) and 5 units of Implementation Pack (North); "
            "February — 55 units of Analytics Pro (North) and 28 units of CRM Suite (East); "
            "March — 40 units of CRM Suite (North) and 15 units of DataNode Server (East)."
        ),
        "category": "csv_keyword",
        "note": "Proper name gives BM25 a strong keyword boost. "
                "Expect high scores — this is the easy CSV case.",
    },
    {
        "question": "Which invoices are overdue for Orion Retail?",
        "ground_truth": (
            "Orion Retail has one overdue invoice: INV-2024-003 for an SEO Audit Report "
            "worth USD 2,200, due on February 2 2024, paid via bank transfer."
        ),
        "category": "csv_keyword",
        "note": "Client name is a strong BM25 keyword. "
                "Tests that the system can find a specific overdue invoice when one actually exists, "
                "as a contrast to the Nexus Solutions negative-evidence case.",
    },

    # ── CSV abstract (harder — no specific names in the query) ────────────────

    {
        "question": "What did the top sales representatives achieve in Q1 2024?",
        "ground_truth": (
            "In Q1 2024, the two sales representatives were Michael Torres and Fatima Al-Hassan. "
            "Michael Torres sold Analytics Pro and Implementation Pack in January (North region), "
            "Analytics Pro and CRM Suite in February, and CRM Suite and DataNode Server in March. "
            "Fatima Al-Hassan sold DataNode Server and Analytics Pro in January (South and West), "
            "DataNode Server and Implementation Pack in February, "
            "and Analytics Pro and CRM Suite in March."
        ),
        "category": "csv_abstract",
        "note": "No specific names in the query — relies on semantic retrieval against CSV rows. "
                "This is the case that failed the first scope calibration pass (score -7.51). "
                "Included specifically to measure how well the pipeline handles it end to end.",
    },

    # ── Cross-document (requires combining 2+ sources) ────────────────────────

    {
        "question": "What is the total value of all invoices raised to Nexus Solutions?",
        "ground_truth": (
            "The total value of all invoices raised to Nexus Solutions is USD 96,000 "
            "across three invoices: INV-2024-004 (USD 32,000), INV-2024-010 (USD 28,500), "
            "and INV-2024-018 (USD 35,500)."
        ),
        "category": "cross_doc",
        "note": "Requires aggregating 3 invoice rows from invoices.csv. "
                "Cross-verifiable against the contract total value ($96,000). "
                "Tests context precision — invoice chunk must be ranked above the contract chunk "
                "for this specific question.",
    },

    # ── Negative evidence (answer IS findable; conclusion is 'none') ──────────

    {
        "question": "Which invoices from Nexus Solutions are overdue?",
        "ground_truth": (
            "None of the Nexus Solutions invoices are overdue. "
            "INV-2024-004 (USD 32,000) has a status of Paid, "
            "INV-2024-010 (USD 28,500) has a status of Invoiced, "
            "and INV-2024-018 (USD 35,500) has a status of Invoiced."
        ),
        "category": "neg_evidence",
        "note": "The invoice data IS in the retrieved context — the correct answer is 'none are overdue'. "
                "The pipeline's grounding prompt causes it to say 'I could not find this information' "
                "rather than reasoning from negative evidence. "
                "RAGAS faithfulness will be high (no hallucination), but context recall may expose "
                "that the answer doesn't actually use the retrieved context to state the correct conclusion. "
                "Included deliberately to surface this limitation.",
    },
]
