"""Limits and storage names for the hourly fleet-wide CI and security fix sweep."""

CI_FLEET_REPORT_BUILDER = "github_ci_fleet"
#: Hourly, off the top of the hour so the sweep does not queue behind other loops.
CI_FLEET_CRON = "17 * * * *"
CI_FLEET_LOOP_NAME = "CI fix sweep"
CI_FLEET_DIRECTORY = "ci-fleet"
#: Without primary repositories, scan what the connection reaches pushed to this recently.
CI_FLEET_ACTIVE_DAYS = 14
CI_FLEET_MAX_REPOSITORIES = 25
#: Coding-agent runs at once; each holds a clone and a coding-agent process.
CI_FLEET_MAX_PARALLEL = 4
#: Fixes one sweep starts at most; the next sweep picks up the rest.
CI_FLEET_MAX_FIXES = 12
#: The sweep stops starting fixes after this, so an hourly tick ends before the next one.
CI_FLEET_SWEEP_SECONDS = 45 * 60
#: A target whose fix failed is retried on the same head commit only after this long.
CI_FLEET_RETRY_HOURS = 24
#: Security alerts below this severity are reported, not fixed.
CI_FLEET_MIN_ALERT_SEVERITY = "high"

__all__ = [
    "CI_FLEET_ACTIVE_DAYS",
    "CI_FLEET_CRON",
    "CI_FLEET_DIRECTORY",
    "CI_FLEET_LOOP_NAME",
    "CI_FLEET_MAX_FIXES",
    "CI_FLEET_MAX_PARALLEL",
    "CI_FLEET_MAX_REPOSITORIES",
    "CI_FLEET_MIN_ALERT_SEVERITY",
    "CI_FLEET_REPORT_BUILDER",
    "CI_FLEET_RETRY_HOURS",
    "CI_FLEET_SWEEP_SECONDS",
]
