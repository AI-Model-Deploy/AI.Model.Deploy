import numpy as np

a = np.float32(1e30)
b = np.float32(-1e30)
c = np.float32(9.5)
d = np.float32(-2.3)

print(f'{a+b+c+d} expected 7.2')
print(f'{a+c+b+d} expected -2.3')
print(f'{a+c+d+b} expected 0')
