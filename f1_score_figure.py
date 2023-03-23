# 用户名：lenovo
# 时间：2023/2/18  21:32
# 用途：画论文中的f1图
import matplotlib.pyplot as plt
import numpy as np

ours = [
    0.98,
    0.91,
    0.98,
    0.99,
    0.86,
    0.96,
    0.93,
    0.92,
    0.92]
temp = [
    0.65,
    0.07,
    0.02,
    0.01,
    0.07,
    0.15,
    0,
    0.29,
    0.38
    ]
wave = [
    0,
    0.92,
    0,
    0,
    0.06,
    0.99,
    0,
    1,
    0.93
    ]
x_label = ['Motor', 'SARS-CoV-2', 'Vibrio1', 'Vibrio2',
                    'Belt', 'Cell section', 'Nitrosop',
                    'Hemocyanin1', 'Hemocyanin2']
width = 0.25
x = np.arange(len(x_label))
plt.figure(figsize=(11,6))
plt.bar(x-width, ours, lw=0.5, fc='r', width=width, label = 'Ours')
plt.bar(x, temp, lw=0.5, fc='b', width=width, label = 'Template matching')
plt.bar(x+width, wave, lw=0.5, fc='g', width=width, label = 'Wavelet Segmentation')

plt.legend()
# fig = plt.figure(figsize=(8,6))
#
# ax = fig.add_subplot(1,1,1)
# # fig.subplots_adjust(bottom=-5)
# plt.plot(x-width, ours, 'ro--')
# plt.plot(x, temp, 'bo--')
# plt.plot(x+width, wave, 'go--')
plt.xticks(x, labels=x_label, rotation=20)
plt.ylabel('F1 score')
# # ax.legend()
# # ax.set_xlabel('Dataset')
# ax.set_ylabel('F1 score')
# ax.set_xticklabels(['','Motor', 'SARS-CoV-2', 'Vibrio1', 'Vibrio2',
#                     'Belt', 'Cell section', 'Nitrosop',
#                     'Hemocyanin1', 'Hemocyanin2'], rotation=30, fontsize='small')
plt.show()

# parameter d of wavelet
y = np.array([2, 0, 0, 0, 0, 8, 0, 0, 0, 4], dtype='int64')
x = np.array([10. , 10.2, 10.4, 10.6, 10.8, 11. , 11.2, 11.4, 11.6, 11.8, 12. ])
plt.figure(figsize=(6,4))
plt.bar(range(10), y)
plt.xticks(range(10), [np.round(10.1+i*0.2, 2) for i in range(10)])
plt.xlabel('d of connected domain')
plt.ylabel('Number')
plt.show()