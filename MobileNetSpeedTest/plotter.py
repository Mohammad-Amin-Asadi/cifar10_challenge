import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make data:
x = 0.5 + np.arange(3)
y = [100,9,24]

# plot
fig, ax = plt.subplots()
bars = (".keras","onnx","QuanOnnx")
y_pos = np.arange(len(bars))

ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
plt.xticks(y_pos+0.5, bars, color='orange', rotation=0, fontweight='bold', fontsize='17')
ax.set(xlim=(0, 3),
       ylim=(0, 500), 
       yticks=np.arange(0, 500,10),
       )
plt.show()