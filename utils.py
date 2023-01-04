import torch 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE



def tensor_equal(t1, t2):
	a1, a2 = t1.detach().numpy(), t2.detach().numpy()   
	np.testing.assert_allclose(a1, a2)



def visualize_data(data, model, device):

	model = model.to(device)
	for images, labels in data:
		if device:
			images = images.to(device)
			labels = labels.to(device)
		
		output = model(images)
	
	output = output.cpu().detach().numpy()
	labels = labels.to('cpu').numpy()

	tsne = TSNE(n_components = 2)
	embeddings = tsne.fit_transform(output)

	plt.figure(figsize = (10, 10))
	plt.title('The embeddings learned by ViT')
	plt.scatter(embeddings[:, 0], embeddings[:, 1], c = labels, s = 50, cmap = 'Paired')
	plt.colorbar()
	plt.show()



