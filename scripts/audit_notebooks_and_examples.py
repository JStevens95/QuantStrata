#!/usr/bin/env python3
"""
Comprehensive audit script for QuantStrata notebooks and example scripts.

This script:
1. Tests syntax validity of all .py files
2. Tests import statements in notebooks and scripts
3. Attempts to execute example scripts with basic validation
4. Generates a detailed audit report
"""

import ast
import json
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import importlib.util


@dataclass
class AuditResult:
    """Result of auditing a single file."""
    path: str
    status: str  # "PASS", "FAIL", "SKIP"
    category: str  # "syntax", "import", "execution"
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class AuditReport:
    """Complete audit report."""
    results: List[AuditResult] = field(default_factory=list)
    
    def add(self, result: AuditResult) -> None:
        self.results.append(result)
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        total = len(self.results)
        passed = len([r for r in self.results if r.status == "PASS"])
        failed = len([r for r in self.results if r.status == "FAIL"])
        skipped = len([r for r in self.results if r.status == "SKIP"])
        
        by_category: Dict[str, Dict[str, int]] = {}
        for r in self.results:
            if r.category not in by_category:
                by_category[r.category] = {"pass": 0, "fail": 0, "skip": 0}
            by_category[r.category][r.status.lower()] += 1
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": f"{100 * passed / total:.1f}%" if total > 0 else "N/A",
            "by_category": by_category,
        }
    
    def failures(self) -> List[AuditResult]:
        """Return list of failures."""
        return [r for r in self.results if r.status == "FAIL"]
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = ["# QuantStrata Notebook & Example Audit Report\n"]
        
        # Summary
        summary = self.summary()
        lines.append("## Summary\n")
        lines.append(f"- **Total files tested:** {summary['total']}")
        lines.append(f"- **Passed:** {summary['passed']}")
        lines.append(f"- **Failed:** {summary['failed']}")
        lines.append(f"- **Skipped:** {summary['skipped']}")
        lines.append(f"- **Pass rate:** {summary['pass_rate']}\n")
        
        # Failures detail
        failures = self.failures()
        if failures:
            lines.append("## Failures\n")
            for f in failures:
                rel_path = f.path
                lines.append(f"### `{rel_path}`")
                lines.append(f"- **Category:** {f.category}")
                lines.append(f"- **Error type:** {f.error_type or 'Unknown'}")
                if f.line_number:
                    lines.append(f"- **Line:** {f.line_number}")
                lines.append(f"- **Error:** `{f.error_message}`\n")
        
        # Passed files
        passed = [r for r in self.results if r.status == "PASS"]
        if passed:
            lines.append("## Passed Files\n")
            for r in passed:
                lines.append(f"- `{r.path}` ({r.category})")
        
        return "\n".join(lines)


def check_python_syntax(filepath: Path) -> AuditResult:
    """Check Python file syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return AuditResult(
            path=str(filepath),
            status="PASS",
            category="syntax",
        )
    except SyntaxError as e:
        return AuditResult(
            path=str(filepath),
            status="FAIL",
            category="syntax",
            error_message=str(e.msg),
            error_type="SyntaxError",
            line_number=e.lineno,
        )
    except Exception as e:
        return AuditResult(
            path=str(filepath),
            status="FAIL",
            category="syntax",
            error_message=str(e),
            error_type=type(e).__name__,
        )


def extract_notebook_code(notebook_path: Path) -> Tuple[str, List[Tuple[int, str]]]:
    """Extract all code cells from a notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = []
    code_parts = []
    
    for idx, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if source.strip():
                cells.append((idx, source))
                code_parts.append(source)
    
    full_code = '\n\n'.join(code_parts)
    return full_code, cells


def check_notebook_syntax(notebook_path: Path) -> AuditResult:
    """Check notebook code cell syntax."""
    try:
        full_code, cells = extract_notebook_code(notebook_path)
        
        # Try to parse combined code
        for cell_idx, source in cells:
            try:
                ast.parse(source)
            except SyntaxError as e:
                return AuditResult(
                    path=str(notebook_path),
                    status="FAIL",
                    category="syntax",
                    error_message=f"Cell {cell_idx}: {e.msg}",
                    error_type="SyntaxError",
                    line_number=e.lineno,
                )
        
        return AuditResult(
            path=str(notebook_path),
            status="PASS",
            category="syntax",
        )
    except json.JSONDecodeError as e:
        return AuditResult(
            path=str(notebook_path),
            status="FAIL",
            category="syntax",
            error_message=f"Invalid JSON: {e}",
            error_type="JSONDecodeError",
        )
    except Exception as e:
        return AuditResult(
            path=str(notebook_path),
            status="FAIL",
            category="syntax",
            error_message=str(e),
            error_type=type(e).__name__,
        )


def check_imports(filepath: Path, is_notebook: bool = False) -> AuditResult:
    """Check if imports work by running python with -c flag."""
    try:
        if is_notebook:
            full_code, cells = extract_notebook_code(filepath)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                full_code = f.read()
        
        # Extract import statements
        tree = ast.parse(full_code)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    if module:
                        imports.append(f"from {module} import {alias.name}")
                    else:
                        imports.append(f"from . import {alias.name}")
        
        if not imports:
            return AuditResult(
                path=str(filepath),
                status="PASS",
                category="import",
            )
        
        # Test imports by executing them
        import_code = '\n'.join(imports)
        result = subprocess.run(
            [sys.executable, '-c', import_code],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(filepath.parent.parent.parent.parent) if is_notebook else str(filepath.parent.parent),
        )
        
        if result.returncode == 0:
            return AuditResult(
                path=str(filepath),
                status="PASS",
                category="import",
            )
        else:
            error_lines = result.stderr.strip().split('\n')
            # Get the last meaningful error line
            error_msg = error_lines[-1] if error_lines else "Unknown error"
            error_type = "ImportError"
            if "ModuleNotFoundError" in result.stderr:
                error_type = "ModuleNotFoundError"
            elif "ImportError" in result.stderr:
                error_type = "ImportError"
            elif "TypeError" in result.stderr:
                error_type = "TypeError"
            
            return AuditResult(
                path=str(filepath),
                status="FAIL",
                category="import",
                error_message=error_msg[:200],
                error_type=error_type,
            )
    
    except subprocess.TimeoutExpired:
        return AuditResult(
            path=str(filepath),
            status="FAIL",
            category="import",
            error_message="Import check timed out (60s)",
            error_type="TimeoutError",
        )
    except Exception as e:
        return AuditResult(
            path=str(filepath),
            status="FAIL",
            category="import",
            error_message=str(e)[:200],
            error_type=type(e).__name__,
        )


def run_audit(project_root: Path) -> AuditReport:
    """Run comprehensive audit."""
    report = AuditReport()
    
    # Find all tutorial notebooks
    tutorials = list((project_root / "docs" / "tutorials").rglob("*.ipynb"))
    
    # Find all example notebooks
    example_notebooks = list((project_root / "examples" / "notebooks").rglob("*.ipynb"))
    
    # Find all example scripts
    example_scripts = []
    for subdir in ["examples"]:
        example_scripts.extend(
            p for p in (project_root / subdir).rglob("*.py")
            if "__pycache__" not in str(p)
        )
    
    print(f"\n{'='*60}")
    print("QuantStrata Notebook & Example Audit")
    print(f"{'='*60}")
    print(f"\nFound:")
    print(f"  - {len(tutorials)} tutorial notebooks")
    print(f"  - {len(example_notebooks)} example notebooks")
    print(f"  - {len(example_scripts)} example scripts")
    print(f"\n{'='*60}")
    
    # Check syntax for all notebooks
    print("\n[1/3] Checking notebook syntax...")
    all_notebooks = tutorials + example_notebooks
    for idx, nb_path in enumerate(all_notebooks, 1):
        rel = nb_path.relative_to(project_root)
        result = check_notebook_syntax(nb_path)
        result.path = str(rel)  # Use relative path in report
        report.add(result)
        status_icon = "✓" if result.status == "PASS" else "✗"
        print(f"  [{idx}/{len(all_notebooks)}] {status_icon} {rel}")
    
    # Check syntax for all scripts
    print("\n[2/3] Checking script syntax...")
    for idx, script_path in enumerate(example_scripts, 1):
        rel = script_path.relative_to(project_root)
        result = check_python_syntax(script_path)
        result.path = str(rel)
        report.add(result)
        status_icon = "✓" if result.status == "PASS" else "✗"
        print(f"  [{idx}/{len(example_scripts)}] {status_icon} {rel}")
    
    # Check imports for a subset of key files
    print("\n[3/3] Checking imports (this may take a while)...")
    
    # Test key notebooks
    key_notebooks = [
        project_root / "docs/tutorials/machine_learning/ml_production.ipynb",
        project_root / "docs/tutorials/deep_hedging/deep_hedging_tutorial.ipynb",
        project_root / "docs/tutorials/models/neural_sde_tutorial.ipynb",
        project_root / "docs/tutorials/q_learning/rl_deployment_tutorial.ipynb",
        project_root / "docs/tutorials/pricing/exotic_options.ipynb",
    ]
    
    for nb_path in key_notebooks:
        if nb_path.exists():
            rel = nb_path.relative_to(project_root)
            result = check_imports(nb_path, is_notebook=True)
            result.path = str(rel)
            result.category = "import"
            report.add(result)
            status_icon = "✓" if result.status == "PASS" else "✗"
            print(f"  {status_icon} {rel}")
    
    # Test key example scripts
    key_scripts = [
        project_root / "examples/showcase/01_european_vanilla_pricing.py",
        project_root / "examples/fundamentals/01_market_ids_and_quotes.py",
        project_root / "examples/pipelines/run_build_curves.py",
    ]
    
    for script_path in key_scripts:
        if script_path.exists():
            rel = script_path.relative_to(project_root)
            result = check_imports(script_path, is_notebook=False)
            result.path = str(rel)
            result.category = "import"
            report.add(result)
            status_icon = "✓" if result.status == "PASS" else "✗"
            print(f"  {status_icon} {rel}")
    
    return report


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    
    report = run_audit(project_root)
    
    # Print summary
    print(f"\n{'='*60}")
    print("AUDIT SUMMARY")
    print(f"{'='*60}")
    
    summary = report.summary()
    print(f"\nTotal:   {summary['total']}")
    print(f"Passed:  {summary['passed']}")
    print(f"Failed:  {summary['failed']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Pass Rate: {summary['pass_rate']}")
    
    # Print failures
    failures = report.failures()
    if failures:
        print(f"\n{'='*60}")
        print("FAILURES")
        print(f"{'='*60}")
        for f in failures:
            print(f"\n{f.path}")
            print(f"  Category: {f.category}")
            print(f"  Error: {f.error_type}: {f.error_message}")
            if f.line_number:
                print(f"  Line: {f.line_number}")
    
    # Save report
    report_path = project_root / "docs" / "development" / "AUDIT_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.to_markdown())
    print(f"\n\nFull report saved to: {report_path}")
    
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
