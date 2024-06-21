
import numpy as np
import scipy.io as sio

# Load data
data = sio.loadmat('./data/digits.mat')
X = data['X']
y = data['y']
Xvalid = data['Xvalid']
yvalid = data['yvalid']
Xtest = data['Xtest']
ytest = data['ytest']

from PIL import Image
import numpy as np
import random


# 随机选择3000个图片进行翻转
num_samples = 2000

# 设置随机种子
random.seed(42)

# 随机选取2000个数字
random_numbers1 = sorted(random.sample(range(5000), num_samples))
random_numbers2 = sorted(random.sample(range(5000), num_samples))


### Resize
new_data1 = np.empty((2000,256))
# # 对选定的图片进行水平翻转

i = 0
for idx in random_numbers1:
    canvas = Image.new('L', (16, 16), color='black')

    offset = ((16 - 8) // 2, (16 - 8) // 2)

    img = Image.fromarray(X[idx].reshape((16,16)).astype(np.uint8))
    smaller_image = img.resize((8,8))

    canvas.paste(smaller_image, offset)
    canvas_array = np.asarray(canvas).reshape(1, -1)
    new_data1[i] = canvas_array
    i += 1

new_y1 = np.empty((2000,))

j = 0
for idx in random_numbers1:
    new_y1[j] = y[idx][0] - 1
    j += 1

arrays_dict = {
    'new_data1': new_data1,
    'new_y1': new_y1
}

np.savez('./data/data_resize.npz', **arrays_dict)
print("resize generate all")

### 随机加噪

new_data2 = np.empty((2000,256))

x = 0
for idx in random_numbers2:
    noise = np.random.normal(loc=0, scale=50, size=(1, 256))

    noisy_image_array = np.clip(X[idx] + noise, 0, 255).astype(np.uint8) 
    new_data1[x] = noisy_image_array
    x += 1

new_y2 = np.empty((2000,))

m = 0
for idx in random_numbers2:
    new_y2[m] = y[idx][0] - 1
    m += 1
    
arrays_dict2 = {
    'new_data2': new_data2,
    'new_y2': new_y2
}

np.savez('./data/data_noise.npz', **arrays_dict2)
print("noise generate all")
