BASIC_ONE_INPUT_SIGNATURES = (
    (("int"), "int"),
    (("float"), "float")
)

ONE_INPUT = {
    "full"  :   ("a",       BASIC_ONE_INPUT_SIGNATURES),
    "neg"   :   ("-a",      BASIC_ONE_INPUT_SIGNATURES),
    "square":   ("a*a",     BASIC_ONE_INPUT_SIGNATURES),
    "cube"  :   ("a*a*a",   BASIC_ONE_INPUT_SIGNATURES),
    "half"  :   ("a/2",     BASIC_ONE_INPUT_SIGNATURES),
    "quart" :   ("a/4",     BASIC_ONE_INPUT_SIGNATURES),
    "eighth":   ("a/8",     BASIC_ONE_INPUT_SIGNATURES),
    "hexth" :   ("a/16",    BASIC_ONE_INPUT_SIGNATURES),
    "bool"  :   ("a?1:0",   BASIC_ONE_INPUT_SIGNATURES),
    "is_zero":  ("a==0.0",  BASIC_ONE_INPUT_SIGNATURES)
}

BASIC_TWO_INPUT_SIGNATURES = (
    (("int", "int"), "int"),
    (("int", "float"), "float"),
    (("float", "int"), "float"),
    (("float", "float"), "float"),
)

TWO_INPUT = {
    # Basic Arithmetic
    "add":  ("a + b",                   BASIC_TWO_INPUT_SIGNATURES),
    "avg": ("(a + b ) / 2",             BASIC_TWO_INPUT_SIGNATURES),

    "sub":  ("a - b",                    BASIC_TWO_INPUT_SIGNATURES),
    "mult": ("a * b",                    BASIC_TWO_INPUT_SIGNATURES),
    "div":  ("(b != 0) ? (a / b) : 0",   BASIC_TWO_INPUT_SIGNATURES),
    "mod":  ("(b != 0) ? (a % b) : 0",   BASIC_TWO_INPUT_SIGNATURES),
        
    # Bitwise
    "and":  ("a & b",                    BASIC_TWO_INPUT_SIGNATURES),
    "or":   ("a | b",                    BASIC_TWO_INPUT_SIGNATURES),
    "xor":  ("a ^ b",                    BASIC_TWO_INPUT_SIGNATURES),
    "lsh":  ("a << b",                   BASIC_TWO_INPUT_SIGNATURES),
    "rsh":  ("a >> b",                   BASIC_TWO_INPUT_SIGNATURES),
    
    # Comparison (returns 0 or 1)
    "gt":   ("(a > b) ? 1 : 0",          BASIC_TWO_INPUT_SIGNATURES),
    "lt":   ("(a < b) ? 1 : 0",          BASIC_TWO_INPUT_SIGNATURES),
    "eq":   ("(a == b) ? 1 : 0",         BASIC_TWO_INPUT_SIGNATURES),
    
    # Complex Logic
    "mandel": ("((a*a - b*b) + a)",      BASIC_TWO_INPUT_SIGNATURES)
}