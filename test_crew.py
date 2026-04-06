"""
Pytest test-suite for the Flaky Test Investigator project.

Covers:
  • Custom tools  (TestHistoryTool, TestSourceCodeTool)
  • Pydantic models (FlakyTestReport, FlakyTestReportList)
  • Agents         (existence & configuration)
  • Crew           (assembly & process type)
"""

import pytest
from tools.custom_tools import TestHistoryTool, TestSourceCodeTool
from models import FlakyTestReport, FlakyTestReportList
from main import (
    flaky_test_detector,
    root_cause_analyst,
    fix_recommender,
    flaky_test_crew,
    detect_flaky_tests_task,
    root_cause_analysis_task,
    fix_recommendation_task,
)
from crewai import Process


# ─────────────────────────────────────────────────────────────────────
# TOOL TESTS
# ─────────────────────────────────────────────────────────────────────

class TestTestHistoryTool:
    """Tests for TestHistoryTool."""

    def setup_method(self):
        self.tool = TestHistoryTool()

    def test_returns_dict(self):
        result = self.tool._run()
        assert isinstance(result, dict)

    def test_contains_ten_tests(self):
        result = self.tool._run()
        assert len(result) == 10

    def test_each_history_has_ten_entries(self):
        result = self.tool._run()
        for test_name, history in result.items():
            assert len(history) == 10, f"{test_name} has {len(history)} entries"

    def test_history_values_are_booleans(self):
        result = self.tool._run()
        for test_name, history in result.items():
            assert all(isinstance(v, bool) for v in history), (
                f"{test_name} contains non-boolean values"
            )

    def test_has_stable_tests(self):
        """At least one test should have all True (100 % pass rate)."""
        result = self.tool._run()
        stable = [name for name, hist in result.items() if all(hist)]
        assert len(stable) > 0, "Expected at least one stable test"

    def test_has_flaky_tests(self):
        """At least one test should have mixed True/False."""
        result = self.tool._run()
        flaky = [
            name for name, hist in result.items()
            if not all(hist) and any(hist)
        ]
        assert len(flaky) > 0, "Expected at least one flaky test"


class TestTestSourceCodeTool:
    """Tests for TestSourceCodeTool."""

    def setup_method(self):
        self.tool = TestSourceCodeTool()

    def test_known_test_returns_source(self):
        code = self.tool._run(test_name="test_payment_processing")
        assert "def test_payment_processing" in code

    def test_stable_test_returns_source(self):
        code = self.tool._run(test_name="test_login_valid_credentials")
        assert "def test_login_valid_credentials" in code

    def test_unknown_test_returns_error(self):
        code = self.tool._run(test_name="test_does_not_exist")
        assert "ERROR" in code

    def test_flaky_test_contains_sleep(self):
        code = self.tool._run(test_name="test_payment_processing")
        assert "time.sleep" in code

    def test_flaky_test_contains_threading(self):
        code = self.tool._run(test_name="test_dashboard_widget_render")
        assert "threading" in code

    def test_all_flaky_tests_have_source(self):
        history_tool = TestHistoryTool()
        history = history_tool._run()
        flaky_tests = [
            name for name, hist in history.items()
            if not all(hist)
        ]
        for test_name in flaky_tests:
            code = self.tool._run(test_name=test_name)
            assert "ERROR" not in code, f"Missing source for {test_name}"


# ─────────────────────────────────────────────────────────────────────
# MODEL TESTS
# ─────────────────────────────────────────────────────────────────────

class TestFlakyTestReport:
    """Tests for the FlakyTestReport Pydantic model."""

    def test_valid_report(self):
        report = FlakyTestReport(
            test_name="test_payment_processing",
            flakiness_rate=40.0,
            probable_cause="Hard-coded time.sleep() waiting for async callback",
            recommended_action="Replace sleep with polling/retry mechanism",
            quarantine=False,
        )
        assert report.test_name == "test_payment_processing"
        assert report.flakiness_rate == 40.0
        assert report.quarantine is False

    def test_quarantine_flag(self):
        report = FlakyTestReport(
            test_name="test_concurrent_file_upload",
            flakiness_rate=50.0,
            probable_cause="Race condition in concurrent uploads",
            recommended_action="Use thread-safe collection or mutex",
            quarantine=True,
        )
        assert report.quarantine is True

    def test_missing_required_field_raises(self):
        with pytest.raises(Exception):
            FlakyTestReport(
                test_name="incomplete_test",
                # missing other required fields
            )


class TestFlakyTestReportList:
    """Tests for the FlakyTestReportList Pydantic model."""

    def test_empty_list(self):
        report_list = FlakyTestReportList(reports=[])
        assert len(report_list.reports) == 0

    def test_list_with_reports(self):
        r1 = FlakyTestReport(
            test_name="test_a",
            flakiness_rate=30.0,
            probable_cause="Timing issue",
            recommended_action="Add retry",
            quarantine=False,
        )
        r2 = FlakyTestReport(
            test_name="test_b",
            flakiness_rate=60.0,
            probable_cause="Race condition",
            recommended_action="Add mutex",
            quarantine=True,
        )
        report_list = FlakyTestReportList(reports=[r1, r2])
        assert len(report_list.reports) == 2

    def test_serialization_round_trip(self):
        r = FlakyTestReport(
            test_name="test_x",
            flakiness_rate=25.0,
            probable_cause="External dependency",
            recommended_action="Mock external service",
            quarantine=False,
        )
        report_list = FlakyTestReportList(reports=[r])
        json_str = report_list.model_dump_json()
        restored = FlakyTestReportList.model_validate_json(json_str)
        assert restored.reports[0].test_name == "test_x"


# ─────────────────────────────────────────────────────────────────────
# AGENT TESTS
# ─────────────────────────────────────────────────────────────────────

class TestAgents:
    """Tests for CrewAI agent configuration."""

    def test_flaky_test_detector_role(self):
        assert flaky_test_detector.role == "Flaky Test Detector"

    def test_flaky_test_detector_has_tool(self):
        tool_names = [t.name for t in flaky_test_detector.tools]
        assert "Test History Tool" in tool_names

    def test_root_cause_analyst_role(self):
        assert root_cause_analyst.role == "Root Cause Analyst"

    def test_root_cause_analyst_has_tool(self):
        tool_names = [t.name for t in root_cause_analyst.tools]
        assert "Test Source Code Tool" in tool_names

    def test_fix_recommender_role(self):
        assert fix_recommender.role == "Fix Recommender"

    def test_fix_recommender_has_no_tools(self):
        assert len(fix_recommender.tools) == 0


# ─────────────────────────────────────────────────────────────────────
# CREW TESTS
# ─────────────────────────────────────────────────────────────────────

class TestCrew:
    """Tests for the assembled crew."""

    def test_crew_has_three_agents(self):
        assert len(flaky_test_crew.agents) == 3

    def test_crew_has_three_tasks(self):
        assert len(flaky_test_crew.tasks) == 3

    def test_crew_process_is_sequential(self):
        assert flaky_test_crew.process == Process.sequential

    def test_task_context_chain(self):
        """Root-cause task depends on detection; fix task depends on both."""
        assert detect_flaky_tests_task in root_cause_analysis_task.context
        assert detect_flaky_tests_task in fix_recommendation_task.context
        assert root_cause_analysis_task in fix_recommendation_task.context

    def test_last_task_has_output_file(self):
        assert fix_recommendation_task.output_file == "flaky_test_report.json"

    def test_last_task_has_output_json(self):
        assert fix_recommendation_task.output_json == FlakyTestReportList
