from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomTriangles, GeomNode, NodePath, Shader,
    GeomEnums, TransparencyAttrib, ShaderBuffer,
    CardMaker, Vec4, Vec3, Vec2, LineSegs
)
import numpy as np
import math

class Graph:
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
        tics_enabled: bool = False,
        tic_config: dict = None
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
        self.tics = None
        self.ssbo = None
        
        self._data = np.full(max_values, min_val, dtype=np.float32)
        self._data_index = 0
        self._data_count = 0
        
        self._max_segments = max_values - 1
        self._segment_data = np.zeros(self._max_segments * 4, dtype=np.float32)

        self._create_background()
        self._setup_instanced_geometry()

        if tics_enabled:
            # Placeholder for your GraphTics logic
            pass

        for i in range(2):
            self.put(0)

    def _create_background(self):
        cm = CardMaker('graph_bg')
        cm.setFrame(-0.5, 0.5, -0.5, 0.5)
        self.bg_node = NodePath(cm.generate())
        self.bg_node.reparentTo(self.parent)
        # Position using XZ plane (Panda3D 2D standard)
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
        
        # We define the quad on X and Z axes (vertical plane)
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
        self.graph_node.set_instance_count(self._max_segments)
        self.graph_node.reparentTo(self.parent)
        self.graph_node.setTransparency(TransparencyAttrib.MAlpha)
        self.graph_node.setBin('fixed', 10)
        
        self._setup_shader()
        self._update_shader_inputs()

    def _update_shader_inputs(self):
        self.graph_node.setShaderInput('min_val', self.min_val)
        self.graph_node.setShaderInput('max_val', self.max_val)
        self.graph_node.setShaderInput('max_segments', float(self._max_segments))
        self.graph_node.setShaderInput('graph_position', Vec3(self.position.x, 0, self.position.y))
        self.graph_node.setShaderInput('graph_size', Vec2(self.size.x, self.size.y))
        self.graph_node.setShaderInput('thickness', self.thickness * 0.001)
        self.graph_node.setShaderInput('line_color', self.line_color)

    def _setup_shader(self):
        vertex_shader = '''
        #version 430
        in vec4 p3d_Vertex;
        uniform mat4 p3d_ModelViewProjectionMatrix;
        uniform float data_count;
        uniform vec3 graph_position;
        uniform vec2 graph_size;
        uniform float thickness;
        uniform vec4 line_color;

        layout(std430, binding = 0) buffer segment_data { float segments[]; };
        out vec4 frag_line_color;
        out vec2 local_pos;

        void main() {
            int instance_id = gl_InstanceID;
            if (float(instance_id) >= data_count - 1.0) {
                gl_Position = vec4(2.0); return;
            }

            int base_idx = instance_id * 4;
            float x1 = segments[base_idx];
            float z1 = segments[base_idx + 1];
            float x2 = segments[base_idx + 2];
            float z2 = segments[base_idx + 3];

            vec2 start = vec2(graph_position.x + (x1 - 0.5) * graph_size.x,
                              graph_position.z + (z1 - 0.5) * graph_size.y);
            vec2 end = vec2(graph_position.x + (x2 - 0.5) * graph_size.x,
                            graph_position.z + (z2 - 0.5) * graph_size.y);

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

    def put(self, value: float):
        self._data[self._data_index] = value
        self._data_index = (self._data_index + 1) % self.max_values
        if self._data_count < self.max_values:
            self._data_count += 1
        self._update_segment_data()
    
    def clear(self):
        self._data[:] = 0
        self._data_index = 0
        self._data_count = 2
        self._update_segment_data()

    def _update_segment_data(self):
        if self._data_count < 2: return
        num_segments = self._data_count - 1
        
        # Calculate Y range for auto-scaling
        # Note: If you want static scaling, use self.min_val/max_val instead
        current_min = np.min(self._data[:self._data_count])
        current_max = np.max(self._data[:self._data_count])
        v_range = max(current_max - current_min, 0.0001)

        points = []
        for i in range(self._data_count):
            idx = (self._data_index - self._data_count + i) % self.max_values
            x = i / (self._data_count - 1)
            # Map value to 0-1 range based on current min/max
            z = (self._data[idx] - current_min) / v_range
            points.append((x, z))
        
        for i in range(num_segments):
            base = i * 4
            self._segment_data[base:base+4] = [points[i][0], points[i][1], points[i+1][0], points[i+1][1]]
        
        self.ssbo = ShaderBuffer('segment_data', self._segment_data, GeomEnums.UH_dynamic)
        self.graph_node.setShaderInput('segment_data', self.ssbo)
        self.graph_node.setShaderInput('data_count', float(self._data_count))

### Main Application
if __name__ == "__main__":
    class GraphDemo(ShowBase):
        def __init__(self):
            ShowBase.__init__(self)
            self.disableMouse()
            
            # 1. Create Graph
            self.graph1 = Graph(
                parent=self.aspect2d,
                max_values=1000,
                line_color=Vec4(1, 0.3, 0.3, 1.0),
                bg_color=Vec4(0.1, 0.1, 0.1, 0.8),
                position=Vec2(0, 0),
                size=Vec2(1.6, 0.8), # Fits most 16:9 screens well
                thickness=3.0
            )
            
            # 2. Setup Axes (Fixed Rotation)
            self.axes_node = self.aspect2d.attachNewNode("axes")
            self.draw_axes()

        def draw_axes(self):
            self.axes_node.getChildren().detach() # Clear old axes
            ls = LineSegs()
            ls.setThickness(2)
            
            # Get graph bounds
            w, h = self.graph1.size.x * 0.5, self.graph1.size.y * 0.5
            px, pz = self.graph1.position.x, self.graph1.position.y
            
            # X-Axis (Bottom)
            ls.setColor(0, 1, 0, 1)
            ls.moveTo(px - w, 0, pz - h)
            ls.drawTo(px + w, 0, pz - h)
            
            # Y-Axis (Left)
            ls.setColor(0, 0.5, 1, 1)
            ls.moveTo(px - w, 0, pz - h)
            ls.drawTo(px - w, 0, pz + h)
            
            self.axes_node.attachNewNode(ls.create())

        def run_logic(self, task):
            t = task.time
            # Generate a wave that changes amplitude to test scaling
            val = (math.sin(t * 2) * 50) + (math.cos(t * 0.5) * 20)
            self.graph1.put(val)
            return task.cont

    app = GraphDemo()
    app.taskMgr.add(app.run_logic, "run_logic")
    app.run()