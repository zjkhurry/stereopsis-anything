import os
import time
import argparse
import threading
import queue
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from depth_anything_v2.dpt import DepthAnythingV2
import mss

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
    default="model/depth_anything_v2_vits.pth",
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
    default=0,
    help="Bais for y-axis of screen capture area.",
)
args = parser.parse_args()

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

scale = args.scale
change = 1e-3


def generate_stereo_images_gpu(raw_image, depth_map):
    global scale
    raw_image = raw_image[:, :, :3]
    image = TF.to_tensor(raw_image).unsqueeze(0).to(DEVICE)
    depth_map = TF.to_tensor(depth_map).unsqueeze(0).to(DEVICE)
    # 获取图像尺寸
    _, _, height, width = image.shape

    if args.stereo_height != height or args.stereo_width != width:
        image = TF.resize(
            image,
            (args.stereo_height, args.stereo_width),
            interpolation=TF.InterpolationMode.NEAREST,
        )
        depth_map = TF.resize(
            depth_map,
            (args.stereo_height, args.stereo_width),
            interpolation=TF.InterpolationMode.NEAREST,
        )
        height = args.stereo_height
        width = args.stereo_width

    # 创建网格
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij"
    )
    grid_y = grid_y.reshape(1, 1, height, width).expand(1, 1, height, width)
    grid_x = grid_x.reshape(1, 1, height, width).expand(1, 1, height, width)
    grid = torch.cat((grid_x, grid_y), dim=1).to(DEVICE)

    depth_map = 1 - (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    # 计算视差
    disparity = scale / (depth_map + args.target_distance)

    disparity = torch.cat([disparity, torch.zeros_like(disparity)], dim=1)

    # 构建新的网格坐标
    new_grid_left = grid.clone()
    new_grid_left -= disparity
    new_grid_right = grid.clone()
    new_grid_right += disparity

    new_grid = torch.cat([new_grid_left, new_grid_right], dim=0)

    image = torch.cat([image, image], dim=0)

    # 使用 grid_sample 进行插值
    stereo_images = F.grid_sample(
        image,
        new_grid.permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode=args.padding_mode,
        align_corners=True,
    )
    stereo_images = torch.clamp(stereo_images, 0, 1)
    left_eye = stereo_images[:1]
    right_eye = stereo_images[1:]

    return left_eye, right_eye


class ScreenCaptureThread(threading.Thread):
    def __init__(self, monitor, output_queue):
        super().__init__()
        self.monitor = monitor
        self.output_queue = output_queue
        self.running = True
        self.image = None

    def run(self):
        with mss.mss(with_cursor=True) as sct:
            while self.running:
                image = sct.grab(self.monitor)
                try:
                    self.output_queue.put_nowait(image)
                except queue.Full:
                    self.output_queue.get_nowait()
                    self.output_queue.put_nowait(image)


class PreprocessThread(threading.Thread):
    def __init__(self, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True

    def run(self):
        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
        }
        model = DepthAnythingV2(**model_configs["vits"])
        model.load_state_dict(
            torch.load(
                args.model_path,
                map_location="cpu",
                weights_only=True,
            )
        )
        model = model.to(DEVICE).eval()

        frames = 0
        start_time = time.time()
        while self.running:
            if self.input_queue.empty():
                continue
            raw_image = self.input_queue.get()
            raw_image = np.array(raw_image)
            depth = model.infer_image(raw_image, 64)

            frames += 1
            fps = frames / (time.time() - start_time)
            if frames % 100 == 0:
                print(f"FPS: {fps:.2f}")

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

    def run(self):
        while self.running:
            if self.input_queue.empty():
                time.sleep(0.005)
                continue
            image, depth = self.input_queue.get()

            left_eye, right_eye = generate_stereo_images_gpu(image, depth)
            stereo_image_gpu = torch.cat((left_eye, right_eye), dim=3) * 255
            stereo_image_gpu = stereo_image_gpu.type(torch.uint8)
            stereo_image = stereo_image_gpu.squeeze(0).permute(1, 2, 0).cpu().numpy()
            try:
                self.output_queue.put_nowait(stereo_image)
            except queue.Full:
                self.output_queue.get_nowait()
                self.output_queue.put_nowait(stereo_image)


class StereoImageViewer:  # QWidget
    def __init__(self, input_queue):
        super().__init__()
        self.input_queue = input_queue
        self.current_image = None
        self.update_from_queue()

    def update_from_queue(self):
        global scale
        cv2.namedWindow("Window_name", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            "Window_name", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        while True:
            key = cv2.waitKey(1)
            if key == ord("q"):  # 按下q键退出
                break
            elif key == ord(","):
                scale -= change
                print(f"Scale increased to {scale}")
            elif key == ord("."):
                scale += change
                print(f"Scale decreased to {scale}")
            if self.input_queue.empty():
                continue
            else:
                img = self.input_queue.get()
                cv2.imshow("image", np.asarray(img))


def main():

    input_queue = queue.Queue(maxsize=3)  # 从屏幕捕获的原始图像
    preprocess_queue = queue.Queue(maxsize=3)  # 处理后的图像
    stereo_queue = queue.Queue(maxsize=3)  # 立体图像
    with mss.mss(with_cursor=True) as sct:
        monitor = sct.monitors[1]
        screen_width, screen_height = monitor["width"], monitor["height"]
    if args.capture_width != 0 and args.capture_height != 0:
        screen_width = args.capture_width
        screen_height = args.capture_height
    monitor = {
        "top": args.bais_y,
        "left": args.bais_x,
        "width": screen_width,
        "height": screen_height,
    }
    screen_capture_thread = ScreenCaptureThread(monitor, input_queue)
    screen_capture_thread.daemon = True
    screen_capture_thread.start()

    preprocess_thread = PreprocessThread(input_queue, preprocess_queue)
    preprocess_thread.daemon = True
    preprocess_thread.start()

    stereo_processing_thread = StereoProcessingThread(preprocess_queue, stereo_queue)
    stereo_processing_thread.daemon = True
    stereo_processing_thread.start()

    StereoImageViewer(stereo_queue)

    print("Exiting...")
    preprocess_thread.running = False
    stereo_processing_thread.running = False
    screen_capture_thread.running = False
    screen_capture_thread.join()
    preprocess_thread.join()
    stereo_processing_thread.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
