import noisy_image_generator as gen
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Add Gaussian noise to an image.")
    parser.add_argument("input_path", help = "Path to the input image")
    parser.add_argument("output_path", help = "Path to save the noisy image")
    parser.add_argument("--noise_std", type = float, default = 25, help = "Noise standard deviation (default: 25)")
    args = parser.parse_args()
    
    gen.add_noise(args.input_path, args.output_path, args.noise_std)
