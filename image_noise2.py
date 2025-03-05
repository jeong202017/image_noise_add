import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# 이미지 경로 설정
image_path = "input_image_path"

# 파일 확장자 확인
input_format = Path(image_path).suffix[1:].upper()  # ".webp" -> "WEBP"

if input_format == 'JPG':
    input_format = 'JPEG'

# 이미지 로드
original_image = Image.open(image_path).convert('RGB')

# Gaussian 노이즈 추가 함수
def add_gaussian_noise(image, intensity=0.1):
    np_image = np.array(image, dtype=np.float32)
    mean = 0
    stddev = intensity * 255
    noise = np.random.normal(mean, stddev, np_image.shape)
    noisy_image = np.clip(np_image + noise, 0, 255)
    return Image.fromarray(noisy_image.astype(np.uint8))

# Correlated Noise 추가 함수
def add_correlated_noise(image, intensity=0.1):
    np_image = np.array(image, dtype=np.float32)
    mean = 0
    stddev = intensity * 255
    noise = np.random.normal(mean, stddev, np_image.shape[:2])

    kernel = np.array([[0.1, 0.4, 0.1], [0.4, 1.0, 0.4], [0.1, 0.4, 0.1]])
    correlated_noise = np.zeros_like(noise)
    padded_noise = np.pad(noise, 1, mode='edge')

    for i in range(correlated_noise.shape[0]):
        for j in range(correlated_noise.shape[1]):
            correlated_noise[i, j] = np.sum(padded_noise[i:i+3, j:j+3] * kernel)

    correlated_noise = np.repeat(correlated_noise[:, :, np.newaxis], 3, axis=2)
    noisy_image = np.clip(np_image + correlated_noise, 0, 255)
    return Image.fromarray(noisy_image.astype(np.uint8))

# # Salt-and-Pepper 노이즈 추가 함수
# def add_salt_and_pepper_noise(image, intensity=0.05):
#     np_image = np.array(image, dtype=np.uint8)
#     prob = intensity
#     noisy_image = np_image.copy()

#     num_salt = np.ceil(prob * np_image.size * 0.5)
#     num_pepper = np.ceil(prob * np_image.size * 0.5)

#     coords_salt = [np.random.randint(0, i, int(num_salt)) for i in np_image.shape[:2]]
#     noisy_image[coords_salt[0], coords_salt[1], :] = 255

#     coords_pepper = [np.random.randint(0, i, int(num_pepper)) for i in np_image.shape[:2]]
#     noisy_image[coords_pepper[0], coords_pepper[1], :] = 0

#     return Image.fromarray(noisy_image)


# # Striped Noise 추가 함수
# def add_striped_noise_fixed(image, stripe_width=20, intensity=10):
#     np_image = np.array(image, dtype=np.int32)
#     height, width, channels = np_image.shape

#     stripe_pattern = np.zeros((height, width), dtype=np.int32)
#     for i in range(0, height, stripe_width * 2):
#         stripe_pattern[i:i + stripe_width, :] = intensity

#     striped_image = np_image + stripe_pattern[:, :, None]
#     striped_image = np.clip(striped_image, 0, 255)
#     return Image.fromarray(striped_image.astype(np.uint8))

# 노이즈 적용 함수
def apply_noise():
    gaussian_intensity = 0.1
    correlated_intensity = 0.01
    # sap_intensity
    # striped_intensity 

    global gaussian_noisy_image, correlated_noisy_image, combined_gaussian_corr_image
    global sap_noisy_image, striped_noisy_image

    gaussian_noisy_image = add_gaussian_noise(original_image, intensity=gaussian_intensity)
    correlated_noisy_image = add_correlated_noise(original_image, intensity=correlated_intensity)
    combined_gaussian_corr_image = add_correlated_noise(gaussian_noisy_image, intensity=correlated_intensity)

    # sap_image = add_salt_and_pepper_noise( input_image ,sap_intensity )
    # striped_image = def add_striped_noise_fixed( input_image, striped_intensity )

    # OpenCV는 NumPy 배열을 필요로 함
    # OpenCV는 BGR을 사용하므로 RGB → BGR 변환 필요
    cv2.imshow("Noisy Image", cv2.cvtColor(np.array(combined_gaussian_corr_image), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 실행
def main():
    apply_noise()

if __name__ == '__main__':
    main()
