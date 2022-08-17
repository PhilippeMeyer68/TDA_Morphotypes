import utils

np = utils.importer('numpy')
plt = utils.importer('matplotlib.pyplot')
pickle = utils.importer('pickle')

with open('E:/TDA/TDA_Morphotypes/Wasserstein.bin', 'rb') as fp:
    GDI_W = pickle.load(fp)

with open('E:/TDA/TDA_Morphotypes/Silhouette.bin', 'rb') as fp:
    GDI_S = pickle.load(fp)

# plt.figure(figsize=(12, 8))
# plt.plot(range(2, 31), GDI_W[0], '.-', color='grey',
#          linewidth=1, label='Wasserstein - Complete')
# plt.plot(range(2, 31), GDI_W[1], '.-', color='orange',
#          linewidth=1, label='Wasserstein - Ward')
# plt.plot(range(2, 31), GDI_S[0], '.-', color='blue',
#          linewidth=1, label='Silhouette - Ward')
# plt.plot(range(2, 31), GDI_S[1], '.-', color='green',
#          linewidth=1, label='Silhouette - K-Means')
# plt.plot(range(2, 31), GDI_S[2], '.-', color='red',
#          linewidth=1, label='Silhouette - K-Medoids')
# plt.ylim(0, 1)
# plt.xticks(range(2, 31, 2))
# plt.xlim(1.5, 30.5)
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.xlabel(r'$K$ (number of clusters)')
# plt.ylabel(r'GDI score')
# plt.grid(True)
# plt.legend(loc='best')
# plt.tight_layout()
# plt.savefig('E:/Projet-M2/2021-m2-ditex-morphotypes/Images/MFGDI_All.jpg', dpi=200)
# plt.show()

GDI_list = GDI_S[0]
plt.figure(figsize=(12, 8))
plt.plot(range(2, 31), GDI_list, 'o-')
plt.plot([7, 7], [0, 1], 'r--')
plt.ylim(0, 1)
plt.xticks(range(2, 31, 2))
plt.xlim(1.5, 30.5)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel(r'$K$ (number of clusters)')
plt.ylabel(r'GDI score')
plt.grid(True)
plt.tight_layout()
plt.savefig('E:/Projet-M2/2021-m2-ditex-morphotypes/Images/MFSWGDI.jpg', dpi=200)
plt.show()
