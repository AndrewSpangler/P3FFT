import itertools

# Type categories
NUMERIC = ["int", "uint", "float"]
INTEGRAL = ["int", "uint"]

def get_sigs(arity: int, types: list, res_type_override: str = None):
    """Generates all permutations of input types for a given arity."""
    sigs = []
    for combo in itertools.product(types, repeat=arity):
        # Result type defaults to the "highest" type in the combo or the override
        if res_type_override:
            res = res_type_override
        else:
            res = "float" if "float" in combo else combo[0]
        sigs.append((list(combo), res))
    return sigs

CONFIG = {
    # --- Original One-Input Ops ---
    "full":     {1: ("a", get_sigs(1, NUMERIC))},
    "neg":      {1: ("-a", get_sigs(1, NUMERIC))},
    "square":   {1: ("a * a", get_sigs(1, NUMERIC))},
    "is_zero":  {1: ("a == 0", get_sigs(1, NUMERIC, "bool"))},
    "bool":     {1: ("a != 0", get_sigs(1, NUMERIC, "bool"))},

    # --- Variadic Arithmetic ---
    "add": {
        2: ("a + b", get_sigs(2, NUMERIC)),
        3: ("a + b + c", get_sigs(3, NUMERIC)),
        4: ("a + b + c + d", get_sigs(4, NUMERIC))
    },
    "sub": {2: ("a - b", get_sigs(2, NUMERIC))},
    "mult": {
        2: ("a * b", get_sigs(2, NUMERIC)),
        3: ("a * b * c", get_sigs(3, NUMERIC))
    },
    "div": {2: ("(b != 0) ? (a / b) : 0", get_sigs(2, NUMERIC))},
    "mod": {2: ("(b != 0) ? (a % b) : 0", get_sigs(2, INTEGRAL))},
    
    # --- Variadic Averages (Forces float result) ---
    "avg": {
        2: ("(a + b) / 2.0", get_sigs(2, NUMERIC, "float")),
        3: ("(a + b + c) / 3.0", get_sigs(3, NUMERIC, "float")),
        4: ("(a + b + c + d) / 4.0", get_sigs(4, NUMERIC, "float"))
    },

    # --- Bitwise ---
    "and": {2: ("a & b", get_sigs(2, INTEGRAL))},
    "or":  {2: ("a | b", get_sigs(2, INTEGRAL))},
    "xor": {2: ("a ^ b", get_sigs(2, INTEGRAL))},
    "lsh": {2: ("a << b", get_sigs(2, INTEGRAL))},
    "rsh": {2: ("a >> b", get_sigs(2, INTEGRAL))},
    
    # --- Comparisons (Return bool) ---
    "gt": {2: ("a > b", get_sigs(2, NUMERIC, "bool"))},
    "lt": {2: ("a < b", get_sigs(2, NUMERIC, "bool"))},
    "eq": {2: ("a == b", get_sigs(2, NUMERIC, "bool"))},

    # --- Utility ---
    "clamp": {3: ("clamp(a, b, c)", get_sigs(3, ["float"]))}
}