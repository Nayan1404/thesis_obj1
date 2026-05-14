# +
# testing_utils.py
import json
import traceback
import signal
import importlib.util
import platform
import sys
from io import StringIO
from unittest.mock import patch  # may not be used, but keeping in case other code imports it

TIMEOUT = 10


class Timeout(Exception):
    pass


# Cross-platform signal handling
if platform.system() != "Windows":
    def _handler(sig, frame):
        raise Timeout

    signal.signal(signal.SIGALRM, _handler)
    SIGNAL_AVAILABLE = True
else:
    SIGNAL_AVAILABLE = False
    print("Warning: Timeout not available on Windows")


def create_module_from_string(name, code):
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(code, mod.__dict__)
    return mod


def _run_script_and_capture_output(code_str, mod):
    """
    Execute a script (no fn_name) and capture anything printed to stdout.
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code_str, mod.__dict__)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    return output


def run_test(sample, test):
    """
    Run tests for either:
      - APPS-style: sample["input_output"] (JSON with inputs/outputs)
      - HumanEval-style: sample["test"] (Python test code) + optional entry_point
    """
    # --- HumanEval path ---
    if sample.get("test"):
        test_code = sample["test"]
        try:
            mod = create_module_from_string("tmp", test)
        except Exception as e:
            # Syntax error or import error in the generated solution
            return [-2], [{"name": type(e).__name__, "value": str(e)}]

        try:
            if SIGNAL_AVAILABLE:
                signal.alarm(TIMEOUT)
            # This executes the HumanEval unit tests, which will raise on failure
            exec(test_code, mod.__dict__)
            if SIGNAL_AVAILABLE:
                signal.alarm(0)
            return [True], [None]
        except Exception as e:
            if SIGNAL_AVAILABLE:
                signal.alarm(0)
            return [False], [{"name": type(e).__name__, "value": traceback.format_exc()}]

    # --- APPS path (your old behavior) ---
    io = json.loads(sample["input_output"])
    ins, outs = io.get("inputs", []), io.get("outputs", [])
    fn_name = io.get("fn_name", None)
    res, err = [], []
    try:
        mod = create_module_from_string("tmp", test)
    except Exception as e:
        return [-2], [{"name": type(e).__name__, "value": str(e)}]

    for i, inp in enumerate(ins):
        try:
            if SIGNAL_AVAILABLE:
                signal.alarm(TIMEOUT)
            if fn_name:
                fn = getattr(mod, fn_name)
                got = fn(*inp) if isinstance(inp, list) else fn(inp)
                ok = got == outs[i]
                res.append(ok)
                err.append(None if ok else {"name": "WrongOutput", "value": str(got)})
            else:
                # Execute as a script and compare stdout
                output = _run_script_and_capture_output(test, mod)
                ok = output.strip() == str(outs[i]).strip()
                res.append(ok)
                err.append(None if ok else {"name": "WrongStdout", "value": output})
            if SIGNAL_AVAILABLE:
                signal.alarm(0)
        except Exception as e:
            if SIGNAL_AVAILABLE:
                signal.alarm(0)
            res.append(False)
            err.append({"name": type(e).__name__, "value": traceback.format_exc()})
    return res, err
# -













