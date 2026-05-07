import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

mu = 3.0
sigma = 0.5
# scipy 中 lognorm 参数：s = sigma, scale = exp(mu)
scale = np.exp(mu)

x = np.linspace(0.01, 60, 500)   # 覆盖到约 60 已包含 99% 以上质量
pdf = lognorm.pdf(x, s=sigma, scale=scale)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'b-', linewidth=2, label=f'Lognormal PDF (μ={mu}, σ={sigma})')
plt.fill_between(x, pdf, alpha=0.2, color='blue')
plt.title('LogNormal Distribution Density Curve')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(alpha=0.3)
plt.legend()
plt.show()