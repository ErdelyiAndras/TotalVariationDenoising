from PIL import Image
import numpy as np

def add_noise(input_path, output_path, noise_std = 25):
    img = Image.open(input_path).convert('L')
    arr = np.array(img).astype(np.float32)
    
    noise = np.random.normal(0, noise_std, arr.shape)
    noisy_arr = arr + noise
    noisy_arr = np.clip(noisy_arr, 0, 255).astype(np.uint8)
    
    noisy_img = Image.fromarray(noisy_arr, mode = 'L')
    noisy_img.save(output_path)
