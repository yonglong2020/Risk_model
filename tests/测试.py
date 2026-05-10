import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

shape = 1.5   # k
scale = 3   # λ
x_min, x_max = 0, 12
x = np.linspace(x_min, x_max, 500)

# 原始 Weibull PDF
pdf_orig = weibull_min.pdf(x, c=shape, scale=scale)

# 截断归一化因子
cdf_upper = weibull_min.cdf(x_max, c=shape, scale=scale)
pdf_trunc = pdf_orig / cdf_upper   # 几乎等于 pdf_orig

plt.figure(figsize=(10, 6))
plt.plot(x, pdf_trunc, 'b-', linewidth=2, label=f'Weibull(k=2, λ=3) truncated to [0,12]')
plt.fill_between(x, pdf_trunc, alpha=0.2, color='blue')
plt.title('Truncated Weibull Distribution PDF')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(alpha=0.3)
plt.legend()
plt.show()