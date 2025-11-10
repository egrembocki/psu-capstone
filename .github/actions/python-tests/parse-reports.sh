#!/usr/bin/env bash
set -euo pipefail

# Parse pytest and coverage reports for GHA summary
# This script parses JUnit XML and coverage XM fiels to generate detailed summaries

parse_junit_xml() {
  local junit_file="$1"

  if [[ ! -f "${junit_file}" ]]; then
    echo "JUnit XML file not found: ${junit_file}"
    return 1
  fi

  # Extract test metrics using grep/sed for compatibility
  local total_tests failures errors skipped test_time
  total_tests=$(grep -o 'tests="[0-9]*"' "${junit_file}" | head -1 | grep -o '[0-9]*' 2>/dev/null || echo "0")
  failures=$(grep -o 'failures="[0-9]*"' "${junit_file}" | head -1 | grep -o '[0-9]*' 2>/dev/null || echo "0")
  errors=$(grep -o 'errors="[0-9]*"' "${junit_file}" | head -1 | grep -o '[0-9]*' 2>/dev/null || echo "0")
  skipped=$(grep -o 'skipped="[0-9]*"' "${junit_file}" | head -1 | grep -o '[0-9]*' 2>/dev/null || echo "0")
  test_time=$(grep -o 'time="[0-9.]*"' "${junit_file}" | head -1 | grep -o '[0-9.]*' 2>/dev/null || echo "0")

  local passed=$((total_tests - failures - errors - skipped))

  # Output test results section
  echo ""
  echo "### Test Results"
  echo ""
  echo "| Total | Passed | Failures | Errors | Skipped | Time (s) |"
  echo "|-------|--------|----------|--------|---------|----------|"
  echo "| ${total_tests} | ${passed} | ${failures} | ${errors} | ${skipped} | ${test_time}s |"
  
  # Show failed tests if any
  if [[ ${failures} -gt 0 || ${errors} -gt 0 ]]; then
    echo ""
    echo "#### Failed Tests Details"
    local failed_tests
    failed_tests=$(grep -E '<testcase.*name="[^"]*".*(<failure|<error)' "${junit_file}" | sed -E 's/.*name="([^"]*)".*$/- \1/' | head -5 2>/dev/null || echo "- Unable to parse failed tests")
    echo "${failed_tests}"
  fi
}

parse_coverage_xml() {
  local coverage_file="$1"

  if [[ ! -f "${coverage_file}" ]]; then
    echo "Coverage XML file not found: ${coverage_file}"
    return 1
  fi

  # Extract coverage metrics using grep/sed for compatibility
  local line_rate line_calc line_pct branch_rate branch_pct branch_calc lines_covered lines_valid
  line_rate=$(grep -o 'line-rate="[0-9.]*"' "${coverage_file}" | head -1 | grep -o '[0-9.]*' 2>/dev/null || echo "0")
  line_calc=$(echo "${line_rate} * 100" | bc 2>/dev/null || echo "0")
  line_pct=$(printf "%.1f" "${line_calc}")
  branch_rate=$(grep -o 'branch-rate="[0-9.]*"' "${coverage_file}" | head -1 | grep -o '[0-9.]*' 2>/dev/null || echo "0")
  branch_calc=$(echo "${branch_rate} * 100" | bc 2>/dev/null || branch_calc="0")
  branch_pct=$(printf "%.1f" "${branch_calc}")

  lines_covered=$(grep -o 'lines-covered="[0-9]*"' "${coverage_file}" | head -1 | grep -o '[0-9]*' 2>/dev/null || echo "0")
  lines_valid=$(grep -o 'lines-valid="[0-9]*"' "${coverage_file}" | head -1 | grep -o '[0-9]*' 2>/dev/null || echo "0")

  # Output coverage results section
  echo ""
  echo "### Code Coverage"
  echo "- **Line Coverage:** ${line_pct}% (${lines_covered}/${lines_valid} lines)"
  [[ "${branch_rate}" != "0" ]] && echo "- **Branch Coverage:** ${branch_pct}%"

  # Show files with low coverage
  local low_coverage
  low_coverage=$(grep -E 'class.*line-rate="0\.[0-7[0-9]*"' "${coverage_file}" | sed -E 's/.*filename="([^"]*)".*line-rate="([^"]*)".*$/- \1 (\2)/' | head -3 2>/dev/null
  if [[ -n "${low_coverage}" ]]; then
    echo ""
    echo "***Files with Low Coverage (<80%):***"
    echo "${low_coverage}"
  fi
}

main() {
  local test_results_summary=""
  local coverage_summary=""

  # Parse JUnit XML if available
  if [[ -f "junit.xml" ]]; then
    test_results_summary=$(parse_junit_xml "junit.xml")
  elif [[ -n "${FALLBACK_TESTS_COUNT:-}" && "${FALLBACK_TESTS_COUNT}" != "0" ]]; then
    test_results_summary="
  ## Test Results
  | Total |
  | ----- |
  | ${FALLBACK_TESTS_COUNT} |"
      fi

      # Parse coverage XML if available
      if [[ "${COVERAGE_ENABLED:-false}" == "true" ]]; then
        if [[ -f "coverage.xml" ]]; then
          coverage_summary=$(parse_coverage_xml "coverage.xml")
        elif [[ -n "${FALLBACK_COVERAGE:-}" ]]; then
          coverage_summary="
  ## Code Coverage
  - **Coverage:** ${FALLBACK_COVERAGE}%"
        fi
      fi

      # Output the combined summary
      echo "${test_results_summary}"
      echo "${coverage_summary}"
}

  # Execute main if script is run directly
  if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "@"
  fi
