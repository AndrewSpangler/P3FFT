import numpy as np
from panda3d.core import (
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomEnums, GeomTriangles, GeomNode, NodePath,
    Shader, TransparencyAttrib, ShaderBuffer,
    CardMaker, Vec4, Vec3, Vec2
)

class Graph:
    """
    Multi-mode graphing class
    Handles both linear and circular graphs
    Can be stacked for use as a visualizer
    """
    def __init__(
        self,
        parent,
        min_val: float = 0.0,
        max_val: float = 100.0,
        max_values: int = 100,
        line_color: Vec4 = Vec4(1, 1, 1, 1),
        bg_color: Vec4 = Vec4(0, 0, 0, 0.5),
        position: Vec2 = Vec2(0, 0),
        size: Vec2 = Vec2(0.5, 0.25),
        thickness: float = 2.0,
        circular: bool = False,
        base_radius: float = 0.3
    ):
        self.parent = parent
        self.min_val = min_val
        self.max_val = max_val
        self.max_values = max_values
        self.line_color = line_color
        self.bg_color = bg_color
        self.thickness = thickness
        self.size = size
        self.position = position
        self.circular = circular
        self.base_radius = base_radius
        self.initial_base_radius = base_radius  # Store initial radius
        self.ssbo = None
        
        self._values = np.full(max_values, min_val, dtype=np.float32)
        self._num_segments = max_values - 1

        self._create_background()
        self._setup_instanced_geometry()

        # Initialize with zeros
        self.set_values(np.zeros(max_values, dtype=np.float32))

    def _create_background(self):
        if self.circular:
            cm = CardMaker('graph_bg')
            cm.setFrame(-0.5, 0.5, -0.5, 0.5)
            self.bg_node = NodePath(cm.generate())
            self.bg_node.reparentTo(self.parent)
            self.bg_node.setPos(self.position.x, 0, self.position.y)
            scale = max(self.size.x, self.size.y)
            self.bg_node.setScale(scale, 1, scale)
            self.bg_node.setColor(self.bg_color)
            self.bg_node.setTransparency(TransparencyAttrib.MAlpha)
            self.bg_node.setBin('fixed', -10)
        else:
            cm = CardMaker('graph_bg')
            cm.setFrame(-0.5, 0.5, -0.5, 0.5)
            self.bg_node = NodePath(cm.generate())
            self.bg_node.reparentTo(self.parent)
            self.bg_node.setPos(self.position.x, 0, self.position.y)
            self.bg_node.setScale(self.size.x, 1, self.size.y)
            self.bg_node.setColor(self.bg_color)
            self.bg_node.setTransparency(TransparencyAttrib.MAlpha)
            self.bg_node.setBin('fixed', -10)

    def _setup_instanced_geometry(self):
        vformat = GeomVertexFormat.get_v3t2()
        vdata = GeomVertexData('segment', vformat, Geom.UH_static)
        vdata.setNumRows(4)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        texcoord = GeomVertexWriter(vdata, 'texcoord')
        
        vertex.addData3(-0.55, 0, -0.5)
        texcoord.addData2(0, 0)
        vertex.addData3(0.55, 0, -0.5)
        texcoord.addData2(1, 0)
        vertex.addData3(0.55, 0, 0.5)
        texcoord.addData2(1, 1)
        vertex.addData3(-0.55, 0, 0.5)
        texcoord.addData2(0, 1)
        
        prim = GeomTriangles(Geom.UH_static)
        prim.addVertices(0, 1, 2)
        prim.addVertices(0, 2, 3)
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode('instanced_segments')
        node.addGeom(geom)
        
        self.graph_node = NodePath(node)
        self.graph_node.set_instance_count(self._num_segments)
        self.graph_node.reparentTo(self.parent)
        self.graph_node.setTransparency(TransparencyAttrib.MAlpha)
        self.graph_node.setBin('fixed', 10)
        
        # Position at the graph's position for rotation
        self.graph_node.setPos(self.position.x, 0, self.position.y)
        
        self._setup_shader()
        self._update_shader_inputs()

    def _update_shader_inputs(self):
        self.graph_node.setShaderInput('min_val', self.min_val)
        self.graph_node.setShaderInput('max_val', self.max_val)
        self.graph_node.setShaderInput('num_segments', float(self._num_segments))
        self.graph_node.setShaderInput('num_values', float(self.max_values))
        self.graph_node.setShaderInput('graph_position', Vec3(0, 0, 0))  # Now relative to node position
        self.graph_node.setShaderInput('graph_size', Vec2(self.size.x, self.size.y))
        self.graph_node.setShaderInput('thickness', self.thickness * 0.001)
        self.graph_node.setShaderInput('line_color', self.line_color)
        self.graph_node.setShaderInput('circular', 1.0 if self.circular else 0.0)
        self.graph_node.setShaderInput('base_radius', self.base_radius)

    def _setup_shader(self):
        vertex_shader = '''
        #version 430
        in vec4 p3d_Vertex;
        uniform mat4 p3d_ModelViewProjectionMatrix;
        uniform float num_segments;
        uniform float num_values;
        uniform float min_val;
        uniform float max_val;
        uniform vec3 graph_position;
        uniform vec2 graph_size;
        uniform float thickness;
        uniform vec4 line_color;
        uniform float circular;
        uniform float base_radius;

        layout(std430, binding = 0) buffer value_data { float values[]; };
        out vec4 frag_line_color;
        out vec2 local_pos;

        #define PI 3.14159265359

        void main() {
            int seg_id = gl_InstanceID;
            if (float(seg_id) >= num_segments) {
                gl_Position = vec4(2.0); 
                return;
            }

            float val1 = values[seg_id];
            float val2 = values[seg_id + 1];
            
            float val_range = max(max_val - min_val, 0.0001);
            float norm_val1 = clamp((val1 - min_val) / val_range, 0.0, 1.0);
            float norm_val2 = clamp((val2 - min_val) / val_range, 0.0, 1.0);

            vec2 start, end;
            
            if (circular > 0.5) {
                float angle1 = 2.0 * PI * float(seg_id) / num_segments;
                float angle2 = 2.0 * PI * float(seg_id + 1) / num_segments;
                
                float radius1 = base_radius + norm_val1 * base_radius * 1.5;
                float radius2 = base_radius + norm_val2 * base_radius * 1.5;
                
                start = vec2(cos(angle1) * radius1, sin(angle1) * radius1);
                end = vec2(cos(angle2) * radius2, sin(angle2) * radius2);
                
                start += vec2(graph_position.x, graph_position.z);
                end += vec2(graph_position.x, graph_position.z);
            } else {
                float x1 = float(seg_id) / num_segments;
                float x2 = float(seg_id + 1) / num_segments;
                
                start = vec2(graph_position.x + (x1 - 0.5) * graph_size.x,
                            graph_position.z + (norm_val1 - 0.5) * graph_size.y);
                end = vec2(graph_position.x + (x2 - 0.5) * graph_size.x,
                          graph_position.z + (norm_val2 - 0.5) * graph_size.y);
            }

            vec2 dir = normalize(end - start);
            vec2 normal = vec2(-dir.y, dir.x);
            float seg_len = distance(start, end);
            vec2 center = (start + end) * 0.5;

            float x_off = p3d_Vertex.x * (0.5 / 0.55);
            vec2 pos_xz = center + dir * (x_off * seg_len) + normal * (p3d_Vertex.z * thickness);

            gl_Position = p3d_ModelViewProjectionMatrix * vec4(pos_xz.x, 0.0, pos_xz.y, 1.0);
            frag_line_color = line_color;
            local_pos = vec2(x_off, p3d_Vertex.z);
        }
        '''
        fragment_shader = '''
        #version 430
        in vec4 frag_line_color;
        in vec2 local_pos;
        out vec4 fragColor;
        void main() {
            float dist = abs(local_pos.y);
            float alpha = frag_line_color.a * (1.0 - smoothstep(0.4, 0.5, dist));
            if (alpha < 0.01) discard;
            fragColor = vec4(frag_line_color.rgb, alpha);
        }
        '''
        self.graph_node.setShader(Shader.make(Shader.SL_GLSL, vertex_shader, fragment_shader))

    def set_values(self, values: np.ndarray):
        """Set all graph values at once."""
        if len(values) != self.max_values:
            raise ValueError(f"Expected {self.max_values} values, got {len(values)}")
        
        self._values[:] = values
        
        self.ssbo = ShaderBuffer('value_data', self._values, GeomEnums.UH_dynamic)
        self.graph_node.setShaderInput('value_data', self.ssbo)
    
    def set_rotation(self, angle_degrees: float):
        """Set the rotation angle around camera axis (roll)"""
        self.graph_node.setR(angle_degrees)
    
    def set_position_offset(self, x_offset: float, z_offset: float):
        """Set position offset for wave effect"""
        self.graph_node.setPos(self.position.x + x_offset, 0, self.position.y + z_offset)
    
    def set_color(self, color: Vec4):
        """Update the line color"""
        self.line_color = color
        self.graph_node.setShaderInput('line_color', self.line_color)
    
    def set_base_radius(self, radius: float):
        """Update the base radius"""
        self.base_radius = radius
        self.graph_node.setShaderInput('base_radius', self.base_radius)
    
    def clear(self):
        """Clear the graph to all zeros."""
        self.set_values(np.zeros(self.max_values, dtype=np.float32))
