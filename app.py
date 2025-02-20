
import cv2
import numpy as np
import os
from tkinter import filedialog, Tk, Button, Label, Canvas, PhotoImage
from PIL import Image, ImageTk

# Create GUI Window
root = Tk()
root.title("Photo Editor")
root.geometry("1200x627")

# Load Background Image
bg_image_path = "772a8876-ea87-4ee9-9b3f-1b5116bcbfb0.jpg"
bg_image = Image.open(bg_image_path)
bg_image = bg_image.resize((1200, 627), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

# Create Canvas for Background
canvas_bg = Canvas(root, width=1200, height=627)
canvas_bg.pack(fill="both", expand=True)
canvas_bg.create_image(0, 0, image=bg_photo, anchor="nw")

# Folder for processed images
downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
os.makedirs(downloads_folder, exist_ok=True)

# Global variables for images
selected_image_path = None
original_image = None
edited_image = None
edited_image_path = None

# List of available image effects
effects = [
    "Negative", "Threshold", "T-Background", "Sepia", "Blur", "Edges",
    "Cartoon", "Sketch", "Brightness", "Contrast", "Emboss", "Sharpen", "BlackWhite",
    "Dilation", "Erosion", "Inverted", "HDR", "OilPainting", "Watercolor", "Glitch",
    "NeonGlow", "MotionBlur", "Vignette", "Pixelation", "FrostedGlass", "WarmEffect",
    "CoolEffect", "Halftone", "Infrared", "Thermal", "XRay", "SketchColor", "Glow",
    "Rainbow", "Cyberpunk", "SoftBlur", "PopArt", "Duotone", "CartoonPop", "ComicBook",
    "CharcoalSketch", "Gotham", "Sunset", "VHS", "ColorSwap", "Newspaper", "GlassReflection",
    "GradientMap", "GlitchArt", "Anaglyph", "Lomo", "Matrix", "WaterReflection", "PastelDream",
    "Mirror", "FireAndIce", "Bokeh", "GlassShatter", "Crosshatch", "TVStatic", "PixelArt",
    "NightVision", "ASCII", "Pointillism", "VinylRecord", "Hologram", "Snowfall", "GlowingEdges"
]

effect_index = 0  # Index to track selected effect

# Function to update effect label
def update_effect_label():
    effect_label.config(text=effects[effect_index])

# Function to move left in effect list
def prev_effect():
    global effect_index
    effect_index = (effect_index - 1) % len(effects)
    update_effect_label()

# Function to move right in effect list
def next_effect():
    global effect_index
    effect_index = (effect_index + 1) % len(effects)
    update_effect_label()

# Function to open file dialog and select an image
def select_image():
    global selected_image_path, original_image
    selected_image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    
    if selected_image_path:
        img = Image.open(selected_image_path)
        img.thumbnail((400, 400))
        original_image = ImageTk.PhotoImage(img)
        canvas_original.create_image(200, 200, image=original_image)

# Function to apply effects
def apply_effect():
    global edited_image, edited_image_path
    
    if selected_image_path:
        img = cv2.imread(selected_image_path, 0)  # Read as grayscale
        img_color = cv2.imread(selected_image_path)
        effect = effects[effect_index]

        if effect == "Negative":
            img_processed = 255 - img
            output_name = "Negative.png"
        elif effect == "Threshold":
            T = 150
            img_processed = np.zeros_like(img)
            img_processed[img >= T] = 255
            output_name = "Threshold.png"   



        elif effect == "T-Background":
            T1, T2 = 100, 180
            img_processed = np.where((img >= T1) & (img <= T2), 255, img)
            output_name = "T-Background.png"
        elif effect == "Sepia":
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            img_processed = cv2.transform(img_color, sepia_filter)
            img_processed = np.clip(img_processed, 0, 255)
            output_name = "Sepia.png"
        elif effect == "Blur":
            img_processed = cv2.GaussianBlur(img_color, (15, 15), 0)
            output_name = "Blur.png"
        elif effect == "Edges":
            img_processed = cv2.Canny(img_color, 100, 200)
            output_name = "Edges.png"
        elif effect == "Cartoon":
            gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            blur = cv2.medianBlur(gray, 7)
            edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
            color = cv2.bilateralFilter(img_color, 9, 300, 300)
            img_processed = cv2.bitwise_and(color, color, mask=edges)
            output_name = "Cartoon.png"
        elif effect == "Sketch":
            gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            inverted = cv2.bitwise_not(gray)
            blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
            inverted_blurred = cv2.bitwise_not(blurred)
            img_processed = cv2.divide(gray, inverted_blurred, scale=256.0)
            output_name = "Sketch.png"
        elif effect == "Brightness":
            hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.add(v, 50)
            v = np.clip(v, 0, 255)
            img_processed = cv2.merge((h, s, v))
            img_processed = cv2.cvtColor(img_processed, cv2.COLOR_HSV2BGR)
            output_name = "Brightness.png"
        elif effect == "Contrast":
            img_processed = cv2.convertScaleAbs(img_color, alpha=1.5, beta=0)
            output_name = "Contrast.png"
        elif effect == "Emboss":
            kernel_emboss = np.array([[0, -1, -1], 
                                     [1,  0, -1], 
                                     [1,  1,  0]])
            img_processed = cv2.filter2D(img_color, -1, kernel_emboss)
            output_name = "Emboss.png"
        elif effect == "Sharpen":
            kernel_sharpen = np.array([[0, -1, 0], 
                                       [-1, 5, -1], 
                                       [0, -1, 0]])
            img_processed = cv2.filter2D(img_color, -1, kernel_sharpen)
            output_name = "Sharpen.png"
        elif effect == "BlackWhite":
            _, img_processed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            output_name = "BlackWhite.png"
        elif effect == "Dilation":
            kernel_dilation = np.ones((5, 5), np.uint8)
            img_processed = cv2.dilate(img, kernel_dilation, iterations=2)
            output_name = "Dilated.png"
        elif effect == "Erosion":
            kernel_erosion = np.ones((5, 5), np.uint8)
            img_processed = cv2.erode(img, kernel_erosion, iterations=2)
            output_name = "Eroded.png"
        elif effect == "Inverted":
            img_processed = cv2.bitwise_not(img_color)
            output_name = "Inverted.png"
        elif effect == "HDR":
            img_processed = cv2.detailEnhance(img_color, sigma_s=12, sigma_r=0.15)
            output_name = "HDR.png"
        elif effect == "OilPainting":
            img_processed = cv2.xphoto.oilPainting(img_color, 7, 1)
            output_name = "OilPainting.png"
        elif effect == "Watercolor":
            img_processed = cv2.stylization(img_color, sigma_s=60, sigma_r=0.6)
            output_name = "Watercolor.png"
        elif effect == "Glitch":
            def apply_glitch_effect(img):
                b, g, r = cv2.split(img)
                shift = 10  
                rows, cols, _ = img.shape
                M = np.float32([[1, 0, shift], [0, 1, 0]])  # Transformation matrix
                r_shifted = cv2.warpAffine(r, M, (cols, rows))
                g_shifted = cv2.warpAffine(g, -M, (cols, rows))
                img_glitch = cv2.merge((b, g_shifted, r_shifted))
                return img_glitch
            img_processed = apply_glitch_effect(img_color)
            output_name = "Glitch.png"
        elif effect == "NeonGlow":
            edges = cv2.Canny(img_color, 100, 200)
            img_processed = cv2.addWeighted(img_color, 0.8, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.4, 0)
            output_name = "NeonGlow.png"
        elif effect == "MotionBlur":
            kernel_motion_blur = np.zeros((15, 15))
            kernel_motion_blur[:, 7] = 1 / 15  # Vertical motion blur
            img_processed = cv2.filter2D(img_color, -1, kernel_motion_blur)
            output_name = "MotionBlur.png"
        elif effect == "Vignette":
            rows, cols = img_color.shape[:2]
            X, Y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
            vignette_mask = np.exp(-2.5 * (X**2 + Y**2))
            img_processed = np.uint8(img_color * vignette_mask[:, :, np.newaxis])
            output_name = "Vignette.png"
        elif effect == "Pixelation":
            def apply_pixelation(img, pixel_size=10):
                h, w = img.shape[:2]
                img_small = cv2.resize(img, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
                img_processed = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_NEAREST)
                return img_processed
            img_processed = apply_pixelation(img_color, pixel_size=10)
            output_name = "Pixelation.png"
        elif effect == "FrostedGlass":
            def frosted_glass(img, kernel_size=5):
                h, w = img.shape[:2]
                frosted = np.zeros_like(img)
                for i in range(h):
                    for j in range(w):
                        random_x = np.clip(i + np.random.randint(-kernel_size, kernel_size), 0, h - 1)
                        random_y = np.clip(j + np.random.randint(-kernel_size, kernel_size), 0, w - 1)
                        frosted[i, j] = img[random_x, random_y]
                return frosted
            img_processed = frosted_glass(img_color, kernel_size=5)
            output_name = "FrostedGlass.png"
        elif effect == "WarmEffect":
            warm_filter = np.array([[1.2, 1.0, 0.8],
                                   [1.2, 1.0, 0.8],
                                   [1.2, 1.0, 0.8]], dtype=np.float32)
            img_processed = cv2.transform(img_color.astype(np.float32), warm_filter)
            img_processed = np.clip(img_processed, 0, 255).astype(np.uint8)
            output_name = "WarmEffect.png"
        elif effect == "CoolEffect":
            cool_filter = np.array([[0.8, 1.0, 1.2],
                                   [0.8, 1.0, 1.2],
                                   [0.8, 1.0, 1.2]], dtype=np.float32)
            img_processed = cv2.transform(img_color.astype(np.float32), cool_filter)
            img_processed = np.clip(img_processed, 0, 255).astype(np.uint8)
            output_name = "CoolEffect.png"
        elif effect == "Halftone":
            def apply_halftone(img, dot_size=8):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rows, cols = gray.shape
                halftone = np.zeros((rows, cols), dtype=np.uint8)
                for i in range(0, rows, dot_size):
                    for j in range(0, cols, dot_size):
                        region = gray[i:i+dot_size, j:j+dot_size]
                        avg_color = np.mean(region)
                        halftone[i:i+dot_size, j:j+dot_size] = avg_color
                return cv2.cvtColor(halftone, cv2.COLOR_GRAY2BGR)
            img_processed = apply_halftone(img_color)
            output_name = "Halftone.png"
        elif effect == "Infrared":
            img_processed = cv2.applyColorMap(img_color, cv2.COLORMAP_JET)
            output_name = "Infrared.png"
        elif effect == "Thermal":
            img_processed = cv2.applyColorMap(img_color, cv2.COLORMAP_HOT)
            output_name = "Thermal.png"
        elif effect == "XRay":
            img_processed = cv2.bitwise_not(img_color)
            img_processed = cv2.applyColorMap(img_processed, cv2.COLORMAP_WINTER)
            output_name = "XRay.png"
        elif effect == "SketchColor":
            gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            inverted = cv2.bitwise_not(gray)
            blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
            inverted_blurred = cv2.bitwise_not(blurred)
            img_processed = cv2.divide(gray, inverted_blurred, scale=256.0)
            img_processed = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)
            img_processed = cv2.addWeighted(img_color, 0.5, img_processed, 0.5, 0)
            output_name = "SketchColor.png"
        elif effect == "Glow":
            img_processed = cv2.GaussianBlur(img_color, (0, 0), 10)
            img_processed = cv2.addWeighted(img_color, 1.5, img_processed, -0.5, 0)
            output_name = "Glow.png"
        elif effect == "Rainbow":
            rainbow = np.zeros_like(img_color)
            rainbow[:, :, 0] = np.linspace(0, 255, img_color.shape[1])  # Blue channel gradient
            rainbow[:, :, 1] = np.flipud(np.linspace(0, 255, img_color.shape[0]))[:, None]  # Green channel gradient
            rainbow[:, :, 2] = np.linspace(255, 0, img_color.shape[1])  # Red channel gradient
            img_processed = cv2.addWeighted(img_color, 0.5, rainbow, 0.5, 0)
            output_name = "Rainbow.png"
        elif effect == "Cyberpunk":
            img_processed = cv2.convertScaleAbs(img_color, alpha=1.5, beta=20)  # Increase contrast
            img_processed = cv2.applyColorMap(img_processed, cv2.COLORMAP_COOL)  # Cool color tint
            output_name = "Cyberpunk.png"
        elif effect == "SoftBlur":
            img_processed = cv2.GaussianBlur(img_color, (5, 5), 2)
            output_name = "SoftBlur.png"
        elif effect == "PopArt":
            img_processed = cv2.stylization(img_color, sigma_s=60, sigma_r=0.25)
            output_name = "PopArt.png"
        elif effect == "Duotone":
            blue_channel, green_channel, red_channel = cv2.split(img_color)
            img_processed = cv2.merge([blue_channel, red_channel, green_channel])  # Swap colors
            output_name = "Duotone.png"
        elif effect == "CartoonPop":
            gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            cartoon = cv2.stylization(img_color, sigma_s=150, sigma_r=0.3)
            mask = cv2.cvtColor(cartoon, cv2.COLOR_BGR2GRAY)
            mask_inv = cv2.bitwise_not(mask)
            img_processed = cv2.bitwise_and(img_color, img_color, mask=mask_inv)
            output_name = "CartoonPop.png"
        elif effect == "ComicBook":
    # Convert the image to grayscale
            gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.medianBlur(gray, 5)
            edges_comic = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
            img_processed = cv2.bitwise_and(img_color, img_color, mask=edges_comic)
            output_name = "ComicBook.png"
        elif effect == "Duotone":
            blue_channel, green_channel, red_channel = cv2.split(img_color)
            img_processed = cv2.merge([blue_channel, red_channel, green_channel])  # Swap colors
            output_name = "Duotone.png"
        elif effect == "CartoonPop":
            gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            cartoon = cv2.stylization(img_color, sigma_s=150, sigma_r=0.3)
            mask = cv2.cvtColor(cartoon, cv2.COLOR_BGR2GRAY)
            mask_inv = cv2.bitwise_not(mask)
            img_processed = cv2.bitwise_and(img_color, img_color, mask=mask_inv)
            output_name = "CartoonPop.png"
        elif effect == "Emboss":
            kernel_emboss = np.array([[0, -1, -1], 
                                    [1,  0, -1], 
                                    [1,  1,  0]])
            img_processed = cv2.filter2D(img_color, -1, kernel_emboss)
            output_name = "Emboss.png"
        elif effect == "CharcoalSketch":
            img_processed = cv2.pencilSketch(img_color, sigma_s=60, sigma_r=0.07, shade_factor=0.05)[0]
            output_name = "CharcoalSketch.png"
        elif effect == "Gotham":
            img_processed = cv2.convertScaleAbs(img_color, alpha=0.7, beta=-30)
            output_name = "Gotham.png"
        elif effect == "Sunset":
            sunset_filter = np.array([1.2, 0.8, 0.5])
            img_processed = np.clip(img_color * sunset_filter, 0, 255).astype(np.uint8)
            output_name = "Sunset.png"
        elif effect == "VHS":
            noise = np.random.normal(0, 15, img_color.shape).astype(np.uint8)
            img_processed = cv2.add(img_color, noise)
            img_processed = cv2.GaussianBlur(img_processed, (5, 5), 2)
            output_name = "VHS.png"
        elif effect == "ColorSwap":
            img_processed = img_color.copy()
            img_processed[:, :, [0, 2]] = img_processed[:, :, [2, 0]]  # Swap B and R channels
            output_name = "ColorSwap.png"
        elif effect == "Newspaper":
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            img_sepia = cv2.applyColorMap(img_gray, cv2.COLORMAP_BONE)
            noise = np.random.normal(0, 20, img_sepia.shape).astype(np.uint8)
            img_processed = cv2.add(img_sepia, noise)
            output_name = "Newspaper.png"
        elif effect == "GlassReflection":
            glass_blur = cv2.GaussianBlur(img_color, (25, 25), 5)
            alpha = 0.6  # Transparency level
            img_processed = cv2.addWeighted(img_color, 1 - alpha, glass_blur, alpha, 0)
            output_name = "GlassReflection.png"
        elif effect == "GradientMap":
            rows, cols = img_color.shape[:2]
            gradient = np.tile(np.linspace(0, 255, cols).reshape(1, -1), (rows, 1)).astype(np.uint8)
            gradient_map = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
            img_processed = cv2.addWeighted(img_color, 0.6, gradient_map, 0.4, 0)
            output_name = "GradientMap.png"
        elif effect == "GlitchArt":
            def glitch_art(img):
                rows, cols, _ = img.shape
                for _ in range(10):
                    rand_x = np.random.randint(0, cols - 10)
                    rand_y = np.random.randint(0, rows - 10)
                    img[rand_y:rand_y + 10, rand_x:rand_x + 10] = np.roll(img[rand_y:rand_y + 10, rand_x:rand_x + 10], shift=5, axis=1)
                return img
            img_processed = glitch_art(img_color.copy())
            output_name = "GlitchArt.png"
        elif effect == "Anaglyph":
            b, g, r = cv2.split(img_color)
            img_processed = cv2.merge([b, g, r - 50])  # Offset red channel
            output_name = "Anaglyph.png"
        elif effect == "Halftone":
            gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            img_processed = cv2.applyColorMap(cv2.GaussianBlur(gray, (7, 7), 5), cv2.COLORMAP_SUMMER)
            output_name = "Halftone.png"
        elif effect == "Lomo":
            vignette = np.zeros_like(img_color, dtype=np.uint8)
            rows, cols = img_color.shape[:2]
            center_x, center_y = cols // 2, rows // 2
            for x in range(cols):
                for y in range(rows):
                    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    vignette[y, x] = np.clip(img_color[y, x] * (1 - 0.0008 * distance), 0, 255)
            img_processed = vignette
            output_name = "Lomo.png"
        elif effect == "Matrix":
            img_processed = np.zeros((rows, cols, 3), dtype=np.uint8)
            cv2.putText(img_processed, "Jeetendar", (cols//4, rows//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            output_name = "Matrix.png"
        elif effect == "Pixelated":
            scale = 8  # Pixel size
            img_small = cv2.resize(img_color, (cols // scale, rows // scale), interpolation=cv2.INTER_LINEAR)
            img_processed = cv2.resize(img_small, (cols, rows), interpolation=cv2.INTER_NEAREST)
            output_name = "Pixelated.png"
        elif effect == "WaterReflection":
            img_processed = cv2.vconcat([img_color, cv2.flip(img_color, 0)])
            output_name = "WaterReflection.png"
        elif effect == "OilPainting":
            img_processed = cv2.stylization(img_color, sigma_s=60, sigma_r=0.5)
            output_name = "OilPainting.png"
        elif effect == "PastelDream":
            img_processed = cv2.addWeighted(img_color, 0.6, np.full_like(img_color, 255, dtype=np.uint8), 0.4, 0)
            output_name = "PastelDream.png"
        elif effect == "Mirror":
            rows, cols = img_color.shape[:2]
            map_x = np.zeros((rows, cols), dtype=np.float32)
            map_y = np.zeros((rows, cols), dtype=np.float32)
            for i in range(rows):
                for j in range(cols):
                    map_x[i,j] = j + 20*np.sin(i/30)
                    map_y[i,j] = i + 20*np.sin(j/30)
            img_processed = cv2.remap(img_color, map_x, map_y, cv2.INTER_LINEAR)
            output_name = "Mirror.png"
        elif effect == "FireAndIce":
            img_processed = img_color.copy()
            img_processed[:, :cols//2] = np.clip(img_processed[:, :cols//2] * [1.5, 0.5, 0.3], 0, 255)
            img_processed[:, cols//2:] = np.clip(img_processed[:, cols//2:] * [0.5, 0.5, 1.5], 0, 255)
            output_name = "FireAndIce.png"
        elif effect == "Rainbow":
            img_processed = cv2.applyColorMap(img_color, cv2.COLORMAP_RAINBOW)
            output_name = "Rainbow.png"
        elif effect == "Bokeh":
            img_processed = cv2.GaussianBlur(img_color, (35, 35), 15)
            output_name = "Bokeh.png"
        elif effect == "GlassShatter":
            img_processed = img_color.copy()
            for i in range(10):  # Random white lines
                x1, y1 = np.random.randint(0, cols), np.random.randint(0, rows)
                x2, y2 = np.random.randint(0, cols), np.random.randint(0, rows)
                cv2.line(img_processed, (x1, y1), (x2, y2), (255, 255, 255), 2)
            output_name = "GlassShatter.png"
        elif effect == "Crosshatch":
            img_processed = cv2.ximgproc.anisotropicDiffusion(img_color, alpha=0.5, K=10, niters=5)
            output_name = "Crosshatch.png"
        elif effect == "TVStatic":
            noise = np.random.randint(0, 256, img_color.shape, dtype=np.uint8)
            img_processed = cv2.addWeighted(img_color, 0.5, noise, 0.5, 0)
            output_name = "TVStatic.png"
        elif effect == "PixelArt":
            img_small = cv2.resize(img_color, (32, 32), interpolation=cv2.INTER_NEAREST)
            img_processed = cv2.resize(img_small, (img_color.shape[1], img_color.shape[0]), interpolation=cv2.INTER_NEAREST)
            output_name = "PixelArt.png"
        elif effect == "Cyberpunk":
            cyberpunk_filter = np.array([1.5, 0.5, 2.0])  # Boost blue and pink tones
            img_processed = np.clip(img_color * cyberpunk_filter, 0, 255).astype(np.uint8)
            output_name = "Cyberpunk.png"
        elif effect == "NightVision":
            img_processed = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            img_processed = cv2.applyColorMap(img_processed, cv2.COLORMAP_BONE)
            output_name = "NightVision.png"
        elif effect == "ASCII":
            gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            ascii_chars = "@%#*+=-:. "  # ASCII mapping
            img_processed = np.zeros_like(gray, dtype=np.uint8)
            for i in range(gray.shape[0]):
                for j in range(gray.shape[1]):
                    img_processed[i, j] = ord(ascii_chars[gray[i, j] // 32])  # Map pixels to ASCII characters
            output_name = "ASCII.png"
        elif effect == "Pointillism":
            gray_blurred = cv2.GaussianBlur(gray, (9, 9), 5)
            img_processed = cv2.applyColorMap(gray_blurred, cv2.COLORMAP_PARULA)
            output_name = "Pointillism.png"
        elif effect == "VinylRecord":
            border = 50
            img_processed = cv2.copyMakeBorder(img_color, border, border, border, border, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            output_name = "VinylRecord.png"
        elif effect == "Hologram":
            holo_filter = np.array([0.3, 0.6, 1.8])  # Increase blue tone
            img_processed = np.clip(img_color * holo_filter, 0, 255).astype(np.uint8)
            for i in range(0, img_color.shape[0], 5):
                img_processed[i:i+1, :, :] = img_processed[i:i+1, :, :] // 2  
            output_name = "Hologram.png"
        elif effect == "PopArt":
            img_processed = cv2.stylization(img_color, sigma_s=60, sigma_r=0.25)
            output_name = "PopArt.png"
        elif effect == "XRay":
            img_processed = cv2.bitwise_not(gray)
            output_name = "XRay.png"
        elif effect == "Snowfall":
            img_processed = img_color.copy()
            for _ in range(1000):
                x, y = np.random.randint(0, img_color.shape[1]), np.random.randint(0, img_color.shape[0])
                cv2.circle(img_processed, (x, y), 2, (255, 255, 255), -1)
            output_name = "Snowfall.png"
        elif effect == "MotionBlur":
            size = 15
            kernel_motion = np.zeros((size, size))
            kernel_motion[int((size - 1) / 2), :] = np.ones(size)
            kernel_motion /= size
            img_processed = cv2.filter2D(img_color, -1, kernel_motion)
            output_name = "MotionBlur.png"
        elif effect == "GlowingEdges":
            edges = cv2.Canny(img_color, 100, 200)
            img_processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) 
            img_processed = cv2.addWeighted(img_color, 0.7, img_processed, 0.3, 0)
            output_name = "GlowingEdges.png"
        else:
            img_processed = img
            output_name = "Effect.png"

        edited_image_path = os.path.join(downloads_folder, output_name)
        cv2.imwrite(edited_image_path, img_processed)

        img_pil = Image.open(edited_image_path)
        img_pil.thumbnail((400, 400))
        edited_image = ImageTk.PhotoImage(img_pil)
        canvas_edited.create_image(200, 200, image=edited_image)

# Function to download image
def download_image():
    if edited_image_path:
        os.startfile(downloads_folder)

# Create Image Canvases
canvas_original = Canvas(root, width=400, height=400, bg="#bcf5c8")
canvas_edited = Canvas(root, width=400, height=400, bg="#bcf5c8")
canvas_bg.create_window(300, 300, window=canvas_original)
canvas_bg.create_window(900, 300, window=canvas_edited)
Label(root, text="Original Image", font=("Arial", 15, "bold"), bg="#d41717").place(x=250, y=20)
Label(root, text="Edited Image", font=("Arial", 15, "bold"), bg="#d41717").place(x=850, y=20)

effect_label = Label(root, text=effects[effect_index], font=("Arial", 12, "bold"), bg="#4CAF50", fg="white")
canvas_bg.create_window(600, 550, window=effect_label)

# Buttons
btn_select = Button(root, text="Select Image", command=select_image, bg="#008CBA", fg="white", font=("Arial", 12, "bold"))
canvas_bg.create_window(300, 550, window=btn_select)

btn_prev = Button(root, text="<", command=prev_effect, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
canvas_bg.create_window(500, 550, window=btn_prev)

btn_next = Button(root, text=">", command=next_effect, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
canvas_bg.create_window(700, 550, window=btn_next)

btn_apply = Button(root, text="Apply Effect", command=apply_effect, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
canvas_bg.create_window(850, 550, window=btn_apply)

btn_download = Button(root, text="Download", command=download_image, bg="#f44336", fg="white", font=("Arial", 10, "bold"))
canvas_bg.create_window(1000, 550, window=btn_download)

root.mainloop()
