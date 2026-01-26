"""
GPU-accelerated Julia set renderer using ModernGL.
"""

import numpy as np
import logging
import moderngl
import glfw

logger = logging.getLogger(__name__)


class GPUJuliaRenderer:
    """GPU-accelerated Julia set renderer using ModernGL and OpenGL shaders."""

    def __init__(self, width: int = 64, height: int = 64):
        """
        Initialize GPU renderer.

        Args:
            width: Image width
            height: Image height
        """
        self.width = width
        self.height = height
        self.ctx = None
        self.program = None
        self.vao = None
        try:
            # Headless context (no window required)
            self.ctx = moderngl.create_context(standalone=True)
            logger.info(
                f"ModernGL initialized (headless): {self.ctx.info['GL_VENDOR']}"
            )
        except Exception as e:
            logger.info(f"Headless context failed: {e}, trying with window...")
            if not glfw.init():
                raise RuntimeError("GLFW init failed")

            glfw.window_hint(glfw.VISIBLE, False)
            window = glfw.create_window(
                self.width, self.height, "Julia", None, None
            )
            if not window:
                raise RuntimeError("GLFW window creation failed")

            glfw.make_context_current(window)
            self.ctx = moderngl.create_context()
            self.glfw_window = window
            logger.info("ModernGL initialized with hidden GLFW window")

        # Vertex shader (simple fullscreen quad)
        vertex_shader = """
        #version 330

        out VS_OUTPUT {
            vec2 uv;
        } vs_out;

        void main() {
            vec2 pos = vec2(gl_VertexID & 1, (gl_VertexID >> 1) & 1) * 2.0 - 1.0;
            vs_out.uv = pos * 0.5 + 0.5;
            gl_Position = vec4(pos, 0.0, 1.0);
        }
        """

        # Fragment shader (Julia set computation)
        fragment_shader = """
        #version 330

        in VS_OUTPUT {
            vec2 uv;
        } fs_in;

        out vec4 color;

        uniform vec2 seed;
        uniform float zoom;
        uniform int max_iter;

        void main() {
            // Map UV to complex plane
            vec2 p = fs_in.uv * 4.0 / zoom - 2.0 / zoom;
            vec2 z = p;
            vec2 c = seed;

            int iter = 0;
            for (int i = 0; i < 256; i++) {
                if (length(z) > 2.0) break;
                // z = z^2 + c
                vec2 z_sq = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
                z = z_sq + c;
                iter++;
                if (iter >= max_iter) break;
            }

            float normalized = float(iter) / float(max_iter);
            color = vec4(vec3(normalized), 1.0);
        }
        """

        self.program = self.ctx.program(
            vertex_shader=vertex_shader, fragment_shader=fragment_shader
        )

        # Create VAO for rendering
        self.vao = self.ctx.vertex_array(self.program, [])

        # Create framebuffer texture
        self.texture = self.ctx.texture((self.width, self.height), 4)
        self.fbo = self.ctx.framebuffer([self.texture])

        logger.info(f"GPU Julia renderer ready ({self.width}x{self.height})")

    def render(
        self, seed_real: float, seed_imag: float, zoom: float = 1.0, max_iter: int = 50
    ) -> np.ndarray:
        """
        Render Julia set.

        Args:
            seed_real: Real part of Julia seed
            seed_imag: Imaginary part of Julia seed
            zoom: Zoom level
            max_iter: Maximum iterations

        Returns:
            Rendered image array (H, W, 3) in [0, 255]
        """
        # Set uniforms
        self.program["seed"] = (seed_real, seed_imag)
        self.program["zoom"] = zoom
        self.program["max_iter"] = max_iter

        # Render to framebuffer
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        # Use a fullscreen quad via triangle strip (4 verts)
        self.vao.render(mode=self.ctx.TRIANGLE_STRIP, vertices=4)

        # Read back as numpy array
        data = self.fbo.read(components=4)
        image = np.frombuffer(data, dtype=np.uint8).reshape(
            (self.height, self.width, 4)
        )

        # Convert RGBA to RGB and flip (OpenGL origin is bottom-left)
        image_rgb = np.flip(image[:, :, :3], axis=0)

        return image_rgb

    def __del__(self):
        """Cleanup OpenGL resources."""
        # Release moderngl resources
        for attr in ["fbo", "texture", "vao", "program", "ctx"]:
            resource = getattr(self, attr, None)
            if resource is not None:
                resource.release()

        # Cleanup GLFW window
        if hasattr(self, "glfw_window"):
            glfw.destroy_window(self.glfw_window)
            glfw.terminate()
