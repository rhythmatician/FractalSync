"""
GPU-accelerated Julia set renderer using ModernGL.
Requires a functioning ModernGL/OpenGL context; there is no CPU fallback.
If GPU initialization fails an explicit error is raised so the caller can handle
the absence of GPU support.
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
        self.ctx: moderngl.Context
        self.program: moderngl.Program
        self.vao: moderngl.VertexArray

        # Try to create headless context
        try:
            # Headless context (no window required)
            self.ctx = moderngl.create_context(standalone=True)
            logger.info(
                f"ModernGL initialized (headless): {self.ctx.info['GL_VENDOR']}"
            )
        except Exception as e:
            logger.warning(f"Headless context failed: {e}, trying with window...")
            if not glfw.init():
                raise RuntimeError("GLFW init failed")
            glfw.window_hint(glfw.VISIBLE, False)
            window = glfw.create_window(self.width, self.height, "Julia", None, None)
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

        # Fragment shader (Julia set computation — fixed-N heightfield normals)
        fragment_shader = """
        #version 330

        in VS_OUTPUT {
            vec2 uv;
        } fs_in;

        out vec4 color;

        uniform vec2 seed;
        uniform float zoom;
        uniform int max_iter;

        const float EPS_POT = 1e-30;
        const int MAX_ITER_CAP = 1024;

        // Iterate to fixed N (max_iter) and return potential f = log(|z_N|)
        float potential_at(vec2 p, vec2 c, int max_iter) {
            vec2 z = p;
            for (int i = 0; i < MAX_ITER_CAP; i++) {
                if (i >= max_iter) break;
                // z = z^2 + c
                vec2 z_sq = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
                z = z_sq + c;
            }
            float r = length(z);
            return log(max(r, EPS_POT));
        }

        void main() {
            // Map UV to complex plane
            vec2 p = fs_in.uv * 4.0 / zoom - 2.0 / zoom;
            vec2 c = seed;

            // Center potential at fixed N
            float f = potential_at(p, c, max_iter);

            // Small offset for finite differences, scale with zoom to remain stable
            float eps = 1e-3 / max(zoom, 1e-6);

            float fxp = potential_at(p + vec2(eps, 0.0), c, max_iter);
            float fxm = potential_at(p - vec2(eps, 0.0), c, max_iter);
            float fyp = potential_at(p + vec2(0.0, eps), c, max_iter);
            float fym = potential_at(p - vec2(0.0, eps), c, max_iter);

            float gx = (fxp - fxm) / (2.0 * eps);
            float gy = (fyp - fym) / (2.0 * eps);

            vec3 normal = normalize(vec3(-gx, -gy, 1.0));

            // Simple directional lighting
            vec3 lightDir = normalize(vec3(0.5, 0.5, 1.0));
            float diff = max(dot(normal, lightDir), 0.0);
            float ambient = 0.2;
            vec3 col = vec3(ambient + 0.8 * diff);

            color = vec4(clamp(col, 0.0, 1.0), 1.0);
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

        self.use_gpu = True
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
        if not self.use_gpu or self.ctx is None:
            raise RuntimeError("GPU renderer not initialized or unavailable")

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
        try:
            # Release moderngl resources
            for attr in ["fbo", "texture", "vao", "program", "ctx"]:
                resource = getattr(self, attr, None)
                if resource is not None:
                    try:
                        resource.release()
                    except Exception:
                        # Ignore release errors
                        pass

            # Cleanup GLFW window
            if hasattr(self, "glfw_window"):
                import glfw

                try:
                    glfw.destroy_window(self.glfw_window)
                    glfw.terminate()
                except Exception:
                    pass
        except Exception:
            # Silently ignore cleanup errors during destruction
            pass
