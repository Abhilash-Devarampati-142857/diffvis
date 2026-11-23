import re
import numpy as np

SAFE_GLOBALS = {
    "__builtins__": {},
    "np": np,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "pi": np.pi,
    "e": np.e,
}


def normalize_partials(expr_str: str) -> str:
    """
    Convert '>' partial derivative shorthand into readable canonical tokens.
    Examples:
      '>>u/>>x' -> 'd2u_dx2'
      '>u/>t'   -> 'du_dt'
      '>x/>u'   -> 'dx_du'
    """
    s = expr_str.strip()
    # Second partials like >>u/>>x or >>u/>>x2
    s = re.sub(r">>\s*([a-zA-Z]\w*)\s*/\s*>>\s*([a-zA-Z]\w*)", r"d2\1_d\2_2", s)
    # First partials like >u/>t or >x/>u
    s = re.sub(r">\s*([a-zA-Z]\w*)\s*/\s*>\s*([a-zA-Z]\w*)", r"d\1_d\2", s)
    return s


def clean_math_expr(expr_str: str) -> str:
    """Normalize common math syntax like ^ to **."""
    s = normalize_partials(expr_str)
    s = s.replace("^", "**")
    return s


def make_evaluator(expr_or_func, var_names):
    """
    Return a callable f(*args) accepting either a Python function or a string expression.
    """
    if callable(expr_or_func):
        return expr_or_func

    code = compile(clean_math_expr(str(expr_or_func)), "<diffvis_expr>", "eval")

    def eval_fn(*values):
        local_ns = dict(zip(var_names, values))
        result = eval(code, SAFE_GLOBALS, local_ns)
        # If expression is a constant like '5', broadcast to match array shape
        if np.ndim(result) == 0 and len(values) > 0:
            first_arr = np.asarray(values[0])
            if first_arr.ndim > 0:
                return np.full_like(first_arr, float(result), dtype=float)
        return result

    return eval_fn
