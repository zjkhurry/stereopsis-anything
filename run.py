import os
import time
import argparse
import threading
import queue
import torch
import numpy as np
import glfw
import OpenGL.GL as gl
from OpenGL.GL.shaders import compileProgram, compileShader
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import coremltools as ct

from Quartz import CoreVideo, CVPixelBufferGetWidth, CVPixelBufferGetHeight, CGRectMake
import ScreenCaptureKit
import AppKit
import objc
import libdispatch
import CoreMedia

parser = argparse.ArgumentParser(
    description="Generate stereo images with adjustable parameters."
)
parser.add_argument(
    "--scale", "-s", type=float, default=0.02, help="Scale factor for the depth map."
)
parser.add_argument(
    "--padding_mode",
    "-p",
    type=str,
    default="zeros",
    choices=["zeros", "border", "reflection"],
    help="Padding mode for grid sampling.",
)
parser.add_argument(
    "--target_distance",
    "-d",
    type=float,
    default=0.4,
    help="Focus distance for depth of field effect.",
)
parser.add_argument(
    "--model_path",
    "-m",
    type=str,
    default="model/depth_anything_v2_vits2.mlpackage",
    help="Path to the CoreML model.",
)
parser.add_argument(
    "--stereo_width",
    "-w",
    type=int,
    default=1920,
    help="Target width for half of the stereo images.",
)
parser.add_argument(
    "--stereo_height",
    "-t",
    type=int,
    default=1080,
    help="Target height for the stereo images.",
)
parser.add_argument(
    "--capture_width", "-cw", type=int, default=0, help="Width of screen capture area."
)
parser.add_argument(
    "--capture_height",
    "-ch",
    type=int,
    default=0,
    help="Height of screen capture area.",
)
parser.add_argument(
    "--bais_x",
    "-bx",
    type=int,
    default=0,
    help="Bais for x-axis of screen capture area.",
)
parser.add_argument(
    "--bais_y",
    "-by",
    type=int,
    default=20,
    help="Bais for y-axis of screen capture area.",
)
parser.add_argument(
    "--compute_unit",
    "-c",
    type=str,
    default="GPU",
    choices=[
        "ALL",
        "GPU",
        "CPU",
        "NE",
    ],
    help="Compute unit for CoreML model.",
)
parser.add_argument(
    "--fps",
    "-f",
    type=int,
    default="30",
    help="The speed to capture your screen.",
)
args = parser.parse_args()

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SCStreamOutput = objc.protocolNamed("SCStreamOutput")

model_path = args.model_path

target_width = args.stereo_width
target_height = args.stereo_height
model_input_width = 518
model_input_height = 392

# 创建网格
grid_y, grid_x = torch.meshgrid(
    torch.linspace(-1, 1, target_height).to(DEVICE),
    torch.linspace(-1, 1, target_width).to(DEVICE),
    indexing="ij",
)
grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(1, 1, target_height, target_width)
grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(1, 1, target_height, target_width)
grid = torch.cat((grid_x, grid_y), dim=1)
zeros = torch.zeros(1, 1, target_height, target_width).to(DEVICE)

scale = args.scale
change = 1e-3


def generate_stereo_images_gpu(raw_image, depth_map):
    global scale
    depth_map = 1 - (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    # 计算视差
    disparity = scale / (depth_map + args.target_distance)

    disparity = torch.cat([disparity, zeros], dim=1)

    # 构建新的网格坐标
    new_grid_left = grid.clone()
    new_grid_left -= disparity
    new_grid_right = grid.clone()
    new_grid_right += disparity

    # 将左右眼的网格坐标拼接在一起
    new_grid = torch.cat([new_grid_left, new_grid_right], dim=0)

    # 将原始图像复制一份，以便处理两个视图
    raw_image = torch.cat([raw_image, raw_image], dim=0)

    # 使用 grid_sample 进行插值
    stereo_images = F.grid_sample(
        raw_image,
        new_grid.permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode=args.padding_mode,
        align_corners=True,
    )
    stereo_images = torch.clamp(stereo_images, 0, 1)

    # 将结果分割成左右眼图像
    left_eye = stereo_images[:1]
    right_eye = stereo_images[1:]

    return left_eye, right_eye


done = True


def record(output_queue):

    global SCStreamOutput

    # done = False
    class SCStreamOutputDelegate(AppKit.NSObject, protocols=[SCStreamOutput]):
        def stream_didOutputSampleBuffer_ofType_(
            self, stream, sample_buffer, output_type
        ):
            nonlocal output_queue

            image_buffer_ref = CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
            # 锁定基址
            CoreVideo.CVPixelBufferLockBaseAddress(image_buffer_ref, 0)
            base_address = CoreVideo.CVPixelBufferGetBaseAddress(image_buffer_ref)
            bytes_per_row = CoreVideo.CVPixelBufferGetBytesPerRow(image_buffer_ref)
            width = CVPixelBufferGetWidth(image_buffer_ref)
            height = CVPixelBufferGetHeight(image_buffer_ref)
            if base_address is objc.NULL:
                return
            im = torch.tensor(
                np.frombuffer(
                    base_address.as_buffer(height * bytes_per_row), dtype=np.uint8
                )
            ).to(DEVICE)
            im = im.view(height, width, 4).permute(2, 0, 1) / 255.0
            im = im[(2, 1, 0), :, :].unsqueeze(0)
            CoreVideo.CVPixelBufferUnlockBaseAddress(image_buffer_ref, 0)
            image = im
            try:
                output_queue.put_nowait(image)
            except queue.Full:
                output_queue.get_nowait()
                output_queue.put_nowait(image)

    def completion_handler(content, error):
        global args
        if error is not None:
            print(error)
            return

        display = content.displays().objectAtIndex_(0)
        filter = (
            ScreenCaptureKit.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                display, AppKit.NSArray.alloc().init()
            )
        )

        config = ScreenCaptureKit.SCStreamConfiguration.alloc().init()
        # config.setWidth_(display.frame().size.width)
        # config.setHeight_(display.frame().size.height)
        if args.capture_width == 0 or args.capture_height == 0:
            width = display.frame().size.width
            height_s = display.frame().size.height
            height = int(width / args.stereo_width * args.stereo_height)
            x = args.bais_x
            y = (height_s - height) // 2 + args.bais_y
        else:
            width = args.capture_width
            height_s = args.capture_height
            x = args.bais_x
            y = args.bais_y

        area = CGRectMake(x, y, width, height)
        config.setWidth_(target_width)
        config.setHeight_(target_height)
        config.setSourceRect_(area)
        config.scalesToFit()
        config.preservesAspectRatio()
        interval = CoreMedia.CMTimeMake(1, args.fps)
        config.setMinimumFrameInterval_(interval)
        config.setQueueDepth_(10)
        config.setPixelFormat_(1111970369)

        stream = (
            ScreenCaptureKit.SCStream.alloc().initWithFilter_configuration_delegate_(
                filter, config, None
            )
        )

        output = SCStreamOutputDelegate.alloc().init().retain()
        queue = libdispatch.dispatch_get_global_queue(0, 0)
        stream.addStreamOutput_type_sampleHandlerQueue_error_(
            output, ScreenCaptureKit.SCStreamOutputTypeScreen, queue, None
        )

        def completion_handler(error):
            # nonlocal done
            global done
            while done:
                time.sleep(0.01)
                continue
            stream.stopCaptureWithCompletionHandler_(lambda error: None)

        stream.startCaptureWithCompletionHandler_(completion_handler)

    ScreenCaptureKit.SCShareableContent.getShareableContentWithCompletionHandler_(
        completion_handler
    )


class ScreenCaptureThread(threading.Thread):
    def __init__(self, output_queue):
        super().__init__()
        self.output_queue = output_queue
        self.image = None

    def run(self):
        record(self.output_queue)


class PreprocessThread(threading.Thread):
    def __init__(self, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True

    def run(self):
        global args
        if args.compute_unit == "ALL":
            u = ct.ComputeUnit.ALL
        elif args.compute_unit == "GPU":
            u = ct.ComputeUnit.CPU_AND_GPU
        elif args.compute_unit == "CPU":
            u = ct.ComputeUnit.CPU_ONLY
        elif args.compute_unit == "NE":
            u = ct.ComputeUnit.CPU_AND_NE
        model = ct.models.MLModel(model_path, compute_units=u)
        while self.running:
            if self.input_queue.empty():
                time.sleep(0.01)
                continue
            raw_image = self.input_queue.get()
            resized_image = TF.resize(
                raw_image,
                (model_input_height, model_input_width),
                interpolation=TF.InterpolationMode.NEAREST,
            ).type(torch.float16)
            depth = model.predict({"image": resized_image.cpu()})
            try:
                self.output_queue.put_nowait((raw_image, depth))
            except queue.Full:
                self.output_queue.get_nowait()
                self.output_queue.put_nowait((raw_image, depth))


class StereoProcessingThread(threading.Thread):
    def __init__(self, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True

    # @profile
    def run(self):
        while self.running:
            if self.input_queue.empty():
                time.sleep(0.005)
                continue
            image, depth = self.input_queue.get()

            depth_tensor = torch.tensor(depth["depth"]).unsqueeze(0).to(DEVICE)
            depth = TF.resize(
                depth_tensor,
                [target_height, target_width],
                interpolation=TF.InterpolationMode.NEAREST,
            )
            left_eye, right_eye = generate_stereo_images_gpu(image, depth)
            stereo_image_gpu = torch.cat((left_eye, right_eye), dim=3) * 255
            stereo_image_gpu = stereo_image_gpu.type(torch.uint8)
            stereo_image = stereo_image_gpu.squeeze(0).permute(1, 2, 0).cpu().numpy()
            try:
                self.output_queue.put_nowait(stereo_image)
            except queue.Full:
                self.output_queue.get_nowait()
                self.output_queue.put_nowait(stereo_image)


class StereoImageViewer:
    def __init__(self, input_queue):
        self.input_queue = input_queue
        self.running = True
        self.prev_plus_key_state = glfw.RELEASE
        self.prev_minus_key_state = glfw.RELEASE
        self.init_glfw()
        self.init_opengl()
        self.run()

    def init_glfw(self):
        if not glfw.init():
            raise Exception("glfw can not be initialized!")

        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(1920, 540, "OpenGL Window", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("glfw window can not be created!")

        # Make the window's context current
        glfw.make_context_current(self.window)
        # Set the window close callback
        glfw.set_window_close_callback(self.window, self.on_close)

    def on_close(self, window):
        self.running = False

    def init_opengl(self):
        # Initialize shaders
        vertex_shader = """
        #version 120
        attribute vec2 position;
        attribute vec2 texCoord;
        varying vec2 _texCoord;

        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            _texCoord = texCoord;
        }
        """

        fragment_shader = """
        #version 120
        uniform sampler2D textureSampler;
        varying vec2 _texCoord;
        void main() {
            gl_FragColor = texture2D(textureSampler, vec2(_texCoord.x, 1.0 - _texCoord.y));
        }
        """

        self.shader = compileProgram(
            compileShader(vertex_shader, gl.GL_VERTEX_SHADER),
            compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER),
        )

        # Set up vertex data (and buffer(s)) and attribute pointers
        vertices = np.array(
            [
                -1.0,
                1.0,
                0.0,
                1.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                0.0,
            ],
            dtype=np.float32,
        )

        indices = np.array([0, 1, 2, 1, 2, 3], dtype=np.uint32)

        self.VBO = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.VBO)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW
        )

        self.EBO = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW
        )

        position = gl.glGetAttribLocation(self.shader, "position")
        gl.glVertexAttribPointer(
            position, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, gl.ctypes.c_void_p(0)
        )
        gl.glEnableVertexAttribArray(position)

        texCoord = gl.glGetAttribLocation(self.shader, "texCoord")
        gl.glVertexAttribPointer(
            texCoord, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, gl.ctypes.c_void_p(8)
        )
        gl.glEnableVertexAttribArray(texCoord)

        # Load texture
        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    def run(self):
        global scale

        frames = 0
        start = time.time()
        try:
            while self.running:
                curr_plus_key_state = glfw.get_key(self.window, glfw.KEY_PERIOD)
                curr_minus_key_state = glfw.get_key(self.window, glfw.KEY_COMMA)
                key = glfw.get_key(self.window, glfw.KEY_Q)
                if key == glfw.PRESS:  # Press 'q' to exit
                    self.running = False

                if (
                    curr_plus_key_state == glfw.PRESS
                    and self.prev_plus_key_state == glfw.RELEASE
                ):  # 按下 '+' 键增加 scale
                    scale += change
                    print(f"Scale increased to {scale}")

                if (
                    curr_minus_key_state == glfw.PRESS
                    and self.prev_minus_key_state == glfw.RELEASE
                ):
                    scale -= change
                    print(f"Scale decreased to {scale}")

                self.prev_plus_key_state = curr_plus_key_state
                self.prev_minus_key_state = curr_minus_key_state

                if self.input_queue.empty():
                    time.sleep(0.005)
                    continue

                img = self.input_queue.get()

                h, w, c = img.shape
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D,
                    0,
                    gl.GL_RGB,
                    w,
                    h,
                    0,
                    gl.GL_RGB,
                    gl.GL_UNSIGNED_BYTE,
                    img,
                )

                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                gl.glUseProgram(self.shader)
                gl.glUniform1i(
                    gl.glGetUniformLocation(self.shader, "textureSampler"), 0
                )
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
                gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
                glfw.swap_buffers(self.window)
                glfw.poll_events()

                frames = frames + 1
                if frames % 50 == 0:
                    fps = frames / (time.time() - start)
                    print("FPS:", fps)
                    start = time.time()
                    frames = 0
        finally:
            glfw.terminate()


def main():
    global done
    input_queue = queue.Queue(maxsize=3)  # 从屏幕捕获的原始图像
    preprocess_queue = queue.Queue(maxsize=3)  # 处理后的图像
    stereo_queue = queue.Queue(maxsize=3)  # 立体图像

    screen_capture_thread = ScreenCaptureThread(input_queue)
    screen_capture_thread.daemon = True
    screen_capture_thread.start()

    preprocess_thread = PreprocessThread(input_queue, preprocess_queue)
    preprocess_thread.daemon = True
    preprocess_thread.start()

    stereo_processing_thread = StereoProcessingThread(preprocess_queue, stereo_queue)
    stereo_processing_thread.daemon = True
    stereo_processing_thread.start()

    viewer = StereoImageViewer(stereo_queue)

    print("Exiting...")
    done = False
    preprocess_thread.running = False
    stereo_processing_thread.running = False
    viewer.running = False
    screen_capture_thread.join()
    preprocess_thread.join()
    stereo_processing_thread.join()


if __name__ == "__main__":
    main()
