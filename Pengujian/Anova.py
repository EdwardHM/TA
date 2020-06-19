import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

#data akurasi model terbaik
LeNet5 = [0.85, 0.67, 0.87, 0.80, 0.93, 0.90, 0.88, 0.87, 0.98, 0.75]
AlexNet = [0.22, 0.20, 0.95, 0.20, 0.20, 0.25, 0.87, 0.20, 0.60, 0.25]
MiniVGGNet= [0.95, 0.62, 0.70, 0.40, 0.73, 0.85, 0.60, 0.93, 0.80, 0.93] 


all_scores = LeNet5 + AlexNet + MiniVGGNet
model_names = (['LeNet5(ADAM)'] * len(LeNet5)) + (['AlexNet(ADAM)'] * len(AlexNet)) + (['MiniVGGNet(NADAM)'] * len(MiniVGGNet))
# print(all_scores)

data = pd.DataFrame({'model': model_names, 'akurasi': all_scores})
# print(data)

rata = data.groupby('model').mean()
# print(rata)

#One way Anova
lm = ols('akurasi ~ model', data=data).fit()
table = sm.stats.anova_lm(lm)
# print (table)

# print(data['akurasi'])

#boxplot
fig= plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_title("Box Plot Dari Model Berdasarkan Akurasi", fontsize=20)
ax.set

# grouped = data.groupby('model')
# for key, value in grouped:
#     print(key, value)

dataBox = [LeNet5, AlexNet, MiniVGGNet]

ax.boxplot(dataBox, labels=[ 'LeNet5(ADAM)', 'AlexNet(ADAM)', 'MiniVGGNet(NADAM)'], showmeans = True)
plt.xlabel("Model")
plt.ylabel("Akurasi")
plt.show()