import numpy as np
from panda3d.core import CardMaker, Vec2, Vec3, NodePath, Shader
from direct.showbase.ShowBase import ShowBase
from .waterfall_texture import WaterfallTexture

class WaterfallDisplay:
    def __init__(
        self,
        base:ShowBase,
        width:int=512,
        height:int=256,
        volume_range:Vec2=Vec2(-80.0, 0.0), # DB
        position:Vec3=Vec3(0, 0, 0),
        scale:Vec3=Vec3(1,1,1),
        aspect_ratio:float=1.0,
    ):        
        # Create waterfall texture manager
        self.waterfall = WaterfallTexture(width, height)
        self.range = volume_range
        self.shader = Shader.make(
            Shader.SL_GLSL,
            self.vertex_shader,
            self.fragment_shader
        )

        cm = CardMaker("waterfall_card")
        cm.setFrame(-aspect_ratio, aspect_ratio, -1, 1)

        self.card = card = NodePath(cm.generate())
        
        card.setShader(self.shader)
        card.setShaderInput("dataTexture", self.waterfall.data_texture)
        card.setShaderInput("writeRowNorm", self.waterfall.get_write_row_normalized())
        card.setShaderInput("minDb", self.range[0])
        card.setShaderInput("maxDb", self.range[1])
        
        self.card.reparentTo(base.render)
        self.card.setPos(*position)
        self.card.setScale(scale)
        
        # Performance tracking
        self.frame_count = 0
        self.fft_count = 0
                    
    def update(self, row:np.ndarray) -> None:
        """Update waterfall with new FFT data"""
        self.frame_count += 1
        self.card.setShaderInput("writeRowNorm", self.waterfall.get_write_row_normalized())
        self.waterfall.add_row(row)
        
    def set_db_range(self, min_db:float, max_db:float) -> None:
        """Set dB range"""
        self.range = (min_db, max_db)
        self.card.setShaderInput("minDb", min_db)
        self.card.setShaderInput("maxDb", max_db)
    
    @property
    def vertex_shader(self) -> str:
        return """
        #version 150
        
        in vec4 p3d_Vertex;
        in vec2 p3d_MultiTexCoord0;
        
        out vec2 texCoord;
        
        uniform mat4 p3d_ModelViewProjectionMatrix;
        uniform float writeRowNorm;  // Normalized write row position (0-1)
        
        void main() {
            gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
            
            // Adjust V coordinate for circular buffer
            // Make newest data appear at top by offsetting based on write position
            float v = p3d_MultiTexCoord0.y;
            
            // Offset so that the row just written appears at the top
            // and older rows scroll down
            v = v + writeRowNorm;
            
            texCoord = vec2(p3d_MultiTexCoord0.x, v);
        }
        """
    
    @property
    def fragment_shader(self) -> str:
        return """
        #version 150
        
        in vec2 texCoord;
        out vec4 fragColor;
        
        uniform sampler2D dataTexture;
        uniform float minDb;
        uniform float maxDb;
        
        vec3 turboColormap(float t) {
            const vec3 c0 = vec3(0.1140890109226559, 0.06288340699912215, 0.2248337216805064);
            const vec3 c1 = vec3(6.716419496985708, 3.182286745507602, 7.571581586103393);
            const vec3 c2 = vec3(-66.09402360453038, -4.9279827041226, -10.09439367561635);
            const vec3 c3 = vec3(228.7660791526501, 25.04986699771073, -91.54105330182436);
            const vec3 c4 = vec3(-334.8351565777451, -69.31749712757485, 288.5858850615712);
            const vec3 c5 = vec3(218.7637218434795, 67.52150567819112, -305.2045772184957);
            const vec3 c6 = vec3(-52.88903478218835, -21.54527364654712, 110.5174647748972);
            
            t = clamp(t, 0.0, 1.0);
            vec3 color = c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
            return clamp(color, 0.0, 1.0);
        }
        
        void main() {
            // Sample FFT magnitude
            float magnitude = texture(dataTexture, texCoord).r;
            
            // Normalize to 0-1 range
            float normalized = (magnitude - minDb) / (maxDb - minDb);
            normalized = clamp(normalized, 0.0, 1.0);
            
            // Apply colormap
            vec3 color = turboColormap(normalized);
            
            fragColor = vec4(color, 1.0);
        }
        """