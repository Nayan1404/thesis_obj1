import argparse
import json
import os
import ast
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from testing_utils import run_test

def check_syntax_valid(code):
    """Check if code is syntactically valid Python"""
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def evaluate_single(g):
    """Evaluate a single generation (supports APPS + HumanEval)."""
    # Prefer explicit task_id if present
    task_id = g.get("task_id", g.get("id", "unknown"))

    # Get code from preferred field(s)
    code = g.get("deal_response", g.get("solutions", ""))
    if isinstance(code, list) and len(code) > 0:
        code = code[0]
    elif not isinstance(code, str):
        code = str(code) if code else ""

    # Detect HumanEval vs APPS
    is_humaneval = bool(g.get("test"))

    # Prepare sample for run_test
    if is_humaneval:
        sample = {
            "test": g.get("test"),
            "entry_point": g.get("entry_point")  # not used directly here but kept for completeness
        }
    else:
        io = g.get("input_output", "{}")
        if not isinstance(io, str):
            io = json.dumps(io)
        sample = {"input_output": io}

    syntax_valid = check_syntax_valid(code)

    try:
        res, err = run_test(sample, test=code)
    except Exception as e:
        return {
            "task_id": task_id,
            "passed": False,
            "syntax_valid": syntax_valid,
            "error": [{"name": "EvaluationError", "value": str(e)}],
            "error_name": "EvaluationError",
            "num_tests": 0,
            "num_passed": 0,
            "hallucinated": False,
        }

    ok = all(r is True for r in res)

    error_name = None
    if not ok:
        for e in err:
            if e:
                error_name = e.get("name", "Unknown")
                break

    # Hallucination: syntactically valid code that fails all test cases
    hallucinated = syntax_valid and len(res) > 0 and sum(1 for r in res if r is True) == 0

    return {
        "task_id": task_id,
        "passed": ok,
        "syntax_valid": syntax_valid,
        "error": err,
        "error_name": error_name,
        "num_tests": len(res),
        "num_passed": sum(1 for r in res if r is True),
        "hallucinated": hallucinated,
    }

def evaluate(generations, save_name, workers=None):
    os.makedirs("evaluated_results", exist_ok=True)

    # Multiprocessing with user-configurable cap
    cpu_cap = max(1, cpu_count() - 1)
    if workers is None:
        workers = min(4, cpu_cap)  # sensible default
    num_processes = max(1, min(workers, cpu_cap))
    print(f"Using {num_processes} processes...")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(evaluate_single, generations),
            total=len(generations),
            desc="Evaluating"
        ))

    # Metrics
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    pass_at_1 = (passed / total) if total > 0 else 0.0

    total_tests = sum(r.get("num_tests", 0) for r in results)
    total_tests_passed = sum(r.get("num_passed", 0) for r in results)
    test_case_accuracy = (total_tests_passed / total_tests) if total_tests > 0 else 0.0

    syntax_valid_count = sum(1 for r in results if r.get("syntax_valid", False))
    syntax_validity_rate = (syntax_valid_count / total) if total > 0 else 0.0

    # Hallucination rate: syntactically valid but fails all tests
    hallucinated_count = sum(1 for r in results if r.get("hallucinated", False))
    hallucination_rate = (hallucinated_count / total) if total > 0 else 0.0

    errors = {}
    for r in results:
        if r["error_name"]:
            errors[r["error_name"]] = errors.get(r["error_name"], 0) + 1

    # Write detailed results
    with open(f"evaluated_results/{save_name}_data.json", "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps({
                "task_id": r["task_id"],
                "passed": r["passed"],
                "syntax_valid": r.get("syntax_valid", False),
                "hallucinated": r.get("hallucinated", False),
                "error": r["error"],
                "num_tests": r.get("num_tests", 0),
                "num_passed": r.get("num_passed", 0)
            }) + "\n")

    # Print summary
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS - {save_name}")
    print(f"{'='*70}")
    print(f"Total Problems: {total}")
    print(f"")
    print(f"📊 KEY METRICS:")
    print(f"{'─'*70}")
    print(f"  1. Pass@1:                {pass_at_1:.4f} ({pass_at_1*100:.2f}%)")
    print(f"     → {passed}/{total} problems solved correctly")
    print(f"")
    print(f"  2. Test Case Accuracy:    {test_case_accuracy:.4f} ({test_case_accuracy*100:.2f}%)")
    print(f"     → {total_tests_passed}/{total_tests} individual test cases passed")
    print(f"")
    print(f"  3. Syntax Validity Rate:  {syntax_validity_rate:.4f} ({syntax_validity_rate*100:.2f}%)")
    print(f"     → {syntax_valid_count}/{total} syntactically valid solutions")
    print(f"")
    print(f"  4. Hallucination Rate:    {hallucination_rate:.4f} ({hallucination_rate*100:.2f}%)")
    print(f"     → {hallucinated_count}/{total} valid code that fails all tests")
    print(f"{'─'*70}")
    print(f"")
    print(f"❌ Failures: {total - passed} ({(1-pass_at_1)*100:.2f}%)")
    print(f"{'='*70}\n")

    # Save error breakdown
    with open(f"evaluated_results/{save_name}_errors.json", "w") as f:
        json.dump(errors, f, indent=2)

    # Save summary metrics
    summary = {
        "model": save_name,
        "total_problems": total,
        "passed_problems": passed,
        "failed_problems": total - passed,
        "pass_at_1": round(pass_at_1, 4),
        "test_case_accuracy": round(test_case_accuracy, 4),
        "syntax_validity_rate": round(syntax_validity_rate, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "total_test_cases": total_tests,
        "passed_test_cases": total_tests_passed,
        "syntax_valid_count": syntax_valid_count,
        "hallucinated_count": hallucinated_count,
        "error_breakdown": errors
    }

    with open(f"evaluated_results/{save_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if errors:
        print("Top 5 Error Types:")
        for error_name, count in sorted(errors.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {error_name}: {count} ({count/total*100:.2f}%)")
        print()

def main(args):
    print(f"Loading from {args.generation_file}...")

    gens = []
    with open(args.generation_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                gens.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}")

    print(f"Loaded {len(gens)} items")

    if gens:
        print(f"First item keys: {list(gens[0].keys())}")
        print(f"Sample task_id: {gens[0].get('task_id', 'NOT FOUND')}")
        code_field = gens[0].get('deal_response', gens[0].get('solutions', 'NOT FOUND'))
        print(f"Code field found: {'Yes' if code_field and code_field != 'NOT FOUND' else 'NO - THIS IS THE ISSUE!'}\n")

    base = os.path.basename(args.generation_file).replace(".jsonl", "").replace(".json", "")
    evaluate(gens, base, workers=args.workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_file", required=True, help="Path to generation results JSONL")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes (default: 4)")
    args = parser.parse_args()
    main(args)
