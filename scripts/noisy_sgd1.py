import matplotlib.pyplot as plt

dict_60 = {'No Noise':[81,85,121,111,140,124,97,136,129,117,123,97,93,115,121,114,111,109,120,107], 'Noise':[14,17,18,21,17,16,16,16,19,19,16,17,15,16,17,23,16,12,16,15]}
dict_70 = {'No Noise':[99,102,136,133,153,140,117,153,149,128,142,129,115,134,133,132,123,125,130,119], 'Noise':[42,35,38,44,55,52,56,33,39,45,42,51,45,42,45,45,45,40,58,42]}
fig, ax = plt.subplots()
ax.boxplot(dict_60.values())
ax.set_xticklabels(dict_60.keys())
plt.ylabel('60% Threshold Passed')
plt.title('Effect of Noise')
plt.savefig('figures/noisy_sgd60.png')
plt.clf()

fig, ax = plt.subplots()
ax.boxplot(dict_70.values())
ax.set_xticklabels(dict_70.keys())
plt.ylabel('70% Threshold Passed')
plt.title('Effect of Noise')
plt.savefig('figures/noisy_sgd70.png')
plt.clf()
