"""
Flaky Test Investigator – CrewAI main entry-point.

Three-agent sequential crew:
  1. Flaky Test Detector   → identifies flaky tests from history
  2. Root Cause Analyst    → inspects source code for flaky patterns
  3. Fix Recommender       → produces a structured JSON report
"""

import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from tools.custom_tools import TestHistoryTool, TestSourceCodeTool
from models import FlakyTestReportList

# ── Load environment variables ──────────────────────────────────────
load_dotenv()

# ── LLM configuration ──────────────────────────────────────────────
LLM_MODEL = "groq/llama-3.3-70b-versatile"

# ── Instantiate tools ──────────────────────────────────────────────
test_history_tool = TestHistoryTool()
test_source_code_tool = TestSourceCodeTool()


# =====================================================================
# AGENTS
# =====================================================================

flaky_test_detector = Agent(
    role="Flaky Test Detector",
    goal=(
        "Analyse the pass/fail history of every test in the suite and "
        "identify which tests are flaky (non-deterministic). "
        "Calculate the flakiness rate for each test."
    ),
    backstory=(
        "You are a senior QA engineer with years of experience in CI/CD pipelines. "
        "You have a keen eye for spotting tests that flip between passing and failing "
        "without any code change. You rely on historical run data to flag unreliable tests."
    ),
    tools=[test_history_tool],
    llm=LLM_MODEL,
    verbose=True,
)

root_cause_analyst = Agent(
    role="Root Cause Analyst",
    goal=(
        "For every flaky test identified, retrieve the test source code and "
        "determine the most probable root cause of flakiness (e.g. race conditions, "
        "hard-coded sleeps, non-deterministic ordering, shared mutable state)."
    ),
    backstory=(
        "You are an expert test automation architect who has debugged thousands of flaky "
        "tests across microservice and monolith codebases. You can read a test and "
        "immediately spot anti-patterns such as time.sleep(), unguarded threads, "
        "reliance on external services, and non-deterministic assertions."
    ),
    tools=[test_source_code_tool],
    llm=LLM_MODEL,
    verbose=True,
)

fix_recommender = Agent(
    role="Fix Recommender",
    goal=(
        "Produce a final structured JSON report (FlakyTestReportList) that includes "
        "the test name, flakiness rate, probable cause, recommended fix, and whether "
        "the test should be quarantined."
    ),
    backstory=(
        "You are a principal SDET who writes quality-gate policies for large engineering "
        "organisations. You combine the data from the detector and root-cause analyst to "
        "create actionable remediation plans. You always output valid JSON matching the "
        "provided schema."
    ),
    llm=LLM_MODEL,
    verbose=True,
)


# =====================================================================
# TASKS  (sequential – context flows from task 1 → 2 → 3)
# =====================================================================

detect_flaky_tests_task = Task(
    description=(
        "Use the Test History Tool to retrieve the pass/fail history of all tests. "
        "For each test, compute the flakiness rate (percentage of failures out of "
        "total runs). List only the tests that have a flakiness rate > 0%. "
        "Return the test names and their flakiness rates."
    ),
    expected_output=(
        "A list of flaky test names together with their flakiness rates "
        "(e.g. 'test_payment_processing: 40% flaky')."
    ),
    agent=flaky_test_detector,
)

root_cause_analysis_task = Task(
    description=(
        "For every flaky test identified in the previous task, use the Test Source "
        "Code Tool to fetch its source code. Analyse the code and determine the "
        "probable root cause of flakiness for each test. Common causes include: "
        "hard-coded time.sleep() calls, threading / race conditions, reliance on "
        "non-deterministic data ordering, shared mutable state, and external service "
        "dependencies."
    ),
    expected_output=(
        "For each flaky test: the test name, a brief description of the identified "
        "flaky pattern(s), and the probable root cause."
    ),
    agent=root_cause_analyst,
    context=[detect_flaky_tests_task],
)

fix_recommendation_task = Task(
    description=(
        "Using the flakiness rates from the Flaky Test Detector and the root-cause "
        "analysis from the Root Cause Analyst, produce a comprehensive JSON report. "
        "For each flaky test, provide:\n"
        "  • test_name\n"
        "  • flakiness_rate (float, 0-100)\n"
        "  • probable_cause (string)\n"
        "  • recommended_action (string – concrete fix)\n"
        "  • quarantine (bool – True if flakiness_rate >= 50%)\n\n"
        "Output MUST conform to the FlakyTestReportList schema."
    ),
    expected_output=(
        "A valid JSON object with a 'reports' key containing a list of "
        "FlakyTestReport objects."
    ),
    agent=fix_recommender,
    context=[detect_flaky_tests_task, root_cause_analysis_task],
    output_json=FlakyTestReportList,
    output_file="flaky_test_report.json",
)


# =====================================================================
# CREW
# =====================================================================

flaky_test_crew = Crew(
    agents=[flaky_test_detector, root_cause_analyst, fix_recommender],
    tasks=[detect_flaky_tests_task, root_cause_analysis_task, fix_recommendation_task],
    process=Process.sequential,
    verbose=True,
)


# =====================================================================
# ENTRY-POINT
# =====================================================================

if __name__ == "__main__":
    print("\n🔍  Flaky Test Investigator – Starting crew run …\n")
    result = flaky_test_crew.kickoff()
    print("\n✅  Crew run complete!")
    print("─" * 60)
    print(result)
