STANDARD_HEADING = """
#version 430
layout (local_size_x = 64) in;
uniform uint nItems;
""".strip()+"\n"

def _buff_line(idx:int, name:str, instance_name:str, buffer_type:str):
    return f"layout(std430, binding = {idx}) buffer {name} {{ {buffer_type} {instance_name}[]; }};"

def create_one_input_shader(
    call:str,
    a_type:str="int",
    res_type:str="int"
) -> str:
    return f"""
    {STANDARD_HEADING}
    {_buff_line(0, "DA", "data_a", a_type)}
    {_buff_line(1, "DR", "results", res_type)}  

    void main() {{
        uint gid = gl_GlobalInvocationID.x;
        if(gid >= nItems) return;
        {a_type} a = data_a[gid];
        results[gid] = {call}; 
    }}
    """

def create_two_input_shader(
    call:str,
    a_type:str="int",
    b_type:str="int",
    res_type:str="int"
) -> str:
    return f"""
    {STANDARD_HEADING}
    {_buff_line(0, "DA", "data_a", a_type)}
    {_buff_line(1, "DB", "data_b", b_type)}
    {_buff_line(2, "DR", "results", res_type)}  

    void main() {{
        uint gid = gl_GlobalInvocationID.x;
        if(gid >= nItems) return;
        {a_type} a = data_a[gid];
        {b_type} b = data_b[gid];
        results[gid] = {call}; 
    }}
    """