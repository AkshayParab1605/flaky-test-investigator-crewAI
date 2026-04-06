"""Pydantic models for structured output of the Flaky Test Investigator crew."""

from pydantic import BaseModel, Field
from typing import List


class FlakyTestReport(BaseModel):
    """Report for a single flaky test."""

    test_name: str = Field(
        ...,
        description="Fully-qualified name of the flaky test function.",
    )
    flakiness_rate: float = Field(
        ...,
        description="Percentage of runs that failed (0.0 – 100.0).",
    )
    probable_cause: str = Field(
        ...,
        description="Root-cause analysis explaining why the test is flaky.",
    )
    recommended_action: str = Field(
        ...,
        description="Concrete recommendation to fix or stabilise the test.",
    )
    quarantine: bool = Field(
        ...,
        description="Whether the test should be quarantined immediately (True/False).",
    )


class FlakyTestReportList(BaseModel):
    """Collection of flaky-test reports produced by the crew."""

    reports: List[FlakyTestReport] = Field(
        default_factory=list,
        description="List of individual flaky-test reports.",
    )
