import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import network_prov1 as network
import random

plot_settings = network.Settings()

plt.clf()
# filename = ['CNN+ATT', 'Hoffmann', 'MIMLRE', 'Mintz', 'PCNN+ATT']
# color = ['red', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']
# for i in range(len(filename)):
#     precision = np.load('./data/' + filename[i] + '_precision.npy')
#     recall = np.load('./data/' + filename[i] + '_recall.npy')
#     plt.plot(recall, precision, color=color[i], lw=2, label=filename[i])

# ATTENTION: put the model iters you want to plot into the list
model_iter = 17200
y_true = np.load('./data/allans_long.npy')
y_scores = np.load('./out/allprob_iter_' + str(model_iter) + '_long.npy')
y_scores_shape = y_scores.shape
y_true = y_true[0:y_scores_shape[0]]
CLASSSTART = 1
CLASSNUM = 22
SKIP = [7, 8, 15, 19, 21, 22]

color = [np.array([0.15105336, 0.2836836, 0.12960373]), np.array([0.09092545, 0.97830118, 0.19633458]),
         np.array([0.34004794, 0.45488394, 0.15263606]), np.array([0.47406442, 0.16929077, 0.14673176]),
         np.array([0.48465484, 0.50405873, 0.64581196]), np.array([0.82301133, 0.54616595, 0.93838273]),
         np.array([0.56706757, 0.60242266, 0.7184122]), np.array([0.51141475, 0.9761238, 0.12079166]),
         np.array([0.86953677, 0.8077808, 0.22016268]), np.array([0.37194619, 0.15886088, 0.67946636]),
         np.array([0.76919515, 0.4772325, 0.98913148]), np.array([0.08542925, 0.72991103, 0.78492492]),
         np.array([0.84384041, 0.56369754, 0.03312675]), np.array([0.53923253, 0.10281096, 0.29859458]),
         np.array([0.60953433, 0.28777263, 0.9375951]), np.array([0.05443915, 0.96477385, 0.19871125]),
         np.array([0.21760586, 0.34425657, 0.32717757]), np.array([0.91363704, 0.6282451, 0.41903352]),
         np.array([0.29968812, 0.85743705, 0.55275921]), np.array([0.94249054, 0.38322872, 0.64642493]),
         np.array([0.635561, 0.96393388, 0.40774075]), np.array([0.4837945, 0.36245953, 0.25444434]),
         np.array([0.90979568, 0.77485958, 0.78802176])]

for class_label in xrange(CLASSSTART, CLASSNUM + CLASSSTART):
    if class_label in SKIP:
        continue
    cur_y_true = y_true[class_label - 1:y_scores_shape[0]:plot_settings.num_classes - 1]
    print class_label, np.sum(cur_y_true)
    cur_y_scores = y_scores[class_label - 1:y_scores_shape[0]:plot_settings.num_classes - 1]
    precision, recall, threshold = precision_recall_curve(cur_y_true, cur_y_scores)
    plt.plot(recall[:], precision[:], lw=2, color=color[class_label - 1], label='Class ' + str(class_label))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall')
plt.legend(loc='center right', bbox_to_anchor=(1.01, 0, 0.3, 1),
           ncol=1)
# plt.legend(loc="upper right")
plt.grid(True)
plt.savefig('iter_' + str(model_iter) + 'pr_all_oversp_skip', bbox_inches='tight')
