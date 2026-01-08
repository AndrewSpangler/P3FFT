import string

STANDARD_HEADING = """
#version 430
layout (local_size_x = 64) in;
uniform uint nItems;
""".strip() + "\n"

def _buff_line(idx: int, name: str, instance_name: str, buffer_type: str):
    """Generates a standard SSBO layout line."""
    return f"layout(std430, binding = {idx}) buffer {name} {{ {buffer_type} {instance_name}[]; }};"

def create_shader(expr: str, arg_types: list[str], res_type: str) -> str:
    """
    Generates a shader for an arbitrary number of inputs.
    Maps input buffers to variables a, b, c, d... 
    """
    buffers = []
    assignments = []
    print("Creating shader", expr, arg_types, " : ", res_type)
    # Generate N input buffers (D0, D1, D2...)
    for i, t in enumerate(arg_types):
        var_name = string.ascii_lowercase[i] # a, b, c, d...
        buffers.append(_buff_line(i, f"D{i}", f"data_{i}", t))
        assignments.append(f"{t} {var_name} = data_{i}[gid];")
    
    # Result buffer is always the last binding slot
    res_binding = len(arg_types)
    buffers.append(_buff_line(res_binding, "DR", "results", res_type))

    return f"""
{STANDARD_HEADING}
{" ".join(buffers)}

void main() {{
    uint gid = gl_GlobalInvocationID.x;
    if(gid >= nItems) return;
    
    {" ".join(assignments)}
    results[gid] = {expr}; 
}}
"""