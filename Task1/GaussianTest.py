import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import scipy.stats as stats

if __name__ == '__main__':
    df = pd.read_csv('JC-201903-citibike-tripdata.csv')

    print(df)
    '''
    trip = df['tripduration'].value_counts().sort_index()
    print(trip)
    '''
    trip = np.array(df['tripduration'])
    trip.sort()
    trip = trip[: np.argmax(trip > 1800)]
#    trip = np.log(trip)
    '''
    trip = np.array([141, 148, 132, 138, 154, 142, 150, 146, 155, 158,\
        150, 140, 147, 148, 144, 150, 149, 145, 149, 158,\
        143, 141, 144, 144, 126, 140, 144, 142, 141, 140,\
        145, 135, 147, 146, 141, 136, 140, 146, 142, 137,\
        148, 154, 137, 139, 143, 140, 131, 143, 141, 149,\
        148, 135, 148, 152, 143, 144, 141, 143, 147, 146,\
        150, 132, 142, 142, 143, 153, 149, 146, 149, 138,\
        142, 149, 142, 137, 134, 144, 146, 147, 140, 142,\
        140, 137, 152, 145])
    '''
    n = trip.shape[0]
    print('n =', n)
    mu_like = trip.mean()
    sigma2_like = np.sum((trip - trip.mean()) ** 2) / n
    sigma_like = sigma2_like ** 0.5
    print('mu_like = %f, sigma_like = %f' %(mu_like, sigma_like))

# 偏度、峰度检验高斯分布

    alpha = 0.5
    sigma1 = (6 * (n - 2) / ((n + 1) * (n + 3))) ** 0.5
    sigma2 = (24 * n * (n - 2) * (n - 3) / (((n + 1) ** 2) * (n + 3) * (n + 5))) ** 0.5
    mu2 = 3 - 6 / (n + 1)
    print('alpha = %f, sigma1 = %f, sigma2 = %f, mu2 = %f' %(alpha, sigma1, sigma2, mu2))
    A = np.array([0] + [np.sum(trip ** k) / n for k in range(1, 5)])
    B2 = A[2] - A[1] ** 2
    B3 = A[3] - 3 * A[2] * A[1] + 2 * (A[1] ** 3)
    B4 = A[4] - 4 * A[3] * A[1] + 6 * A[2] * (A[1] ** 2) - 3 * (A[1] ** 4)
    print('A:', A[1: ])
    print('B2 = %f, B3 = %f, B4 = %f' %(B2, B3, B4))
    g1 = B3 / (B2 ** 1.5)
    g2 = B4 / (B2 ** 2)
    print('g1 = %f, g2 = %f' %(g1, g2))
    mu1 = g1 / sigma1
    mu2_new = (g2 - mu2) / sigma2
    print('mu1 = %f, mu2_new = %f' %(mu1, mu2_new))

    ax = plt.subplot('111')
    y_l, x_l, _ = plt.hist(trip, bins = 40, density = 1, zorder = 0)
    pd.DataFrame(trip).plot.kde(ax = ax, zorder = 10)
    y_fit = stats.norm.pdf(x_l, mu_like, sigma_like)
    plt.plot(x_l, y_fit, '-', zorder = 20)
    plt.scatter(x_l[: -1], y_l, s = 5, c = 'r', zorder = 30)
    plt.legend(['KDE', 'normal with MLE'])
    plt.xlabel('Tripduration(s)')
    plt.ylabel('$f_i / n \Delta$')
    plt.xlim([50, 620])
    plt.title('Normal Distribution')
    plt.show()
