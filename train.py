import torch 
import torch.nn as nn 



def train_func(train_loader, model, optimizer, loss_func, max_epochs = 20, validation_loader = None, 
			   batch_size = 64, scheduler = None, device = None, test = None):

	''' 
	This function takes raw data as input and converst it to data loader itself.
	Also, it does apply the model on the test data if test data is given. 

	'''


	'''validation, test = torch.utils.data.random_split(val_data, [2000, 8000])
			
				train_loader = torch.utils.data.DataLoader(transformed_cifar10, batch_size = batch_size, shuffle = True, drop_last = True)
				val_loader = torch.utils.data.DataLoader(validation, batch_size = batch_size, shuffle = True, drop_last = True) '''


	n_batches_train = len(train_loader)
	n_batches_val = len(validation_loader)
	n_samples_train = batch_size * n_batches_train
	n_samples_val = batch_size * n_batches_val


	losses = []
	accuracy = []
	validation_loss = []
	validation_accuracy = []


	for epoch in range(max_epochs):
		running_loss, correct = 0, 0
		for images, labels in train_loader:
			if device:
				images = images.to(device)
				labels = labels.to(device)

			model.train()
			outputs = model(images)
			loss = loss_func(outputs, labels)
			predictions = outputs.argmax(1)
			correct += int(sum(predictions == labels))
			running_loss += loss.item()


			#BACKWARD AND OPTIMZIE
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()


		loss_epoch = running_loss / n_batches_train
		accuracy_epoch = correct / n_samples_train
		scheduler.step(loss_epoch)

		losses.append(loss_epoch)
		accuracy.append(accuracy_epoch)

		print('Epoch [{}/{}], Training Accuracy [{:.4f}], Training Loss: {:.4f}'
            .format(epoch + 1, max_epochs, accuracy_epoch, loss_epoch), end = '  ')
 		print('Correct/ Total: [{}/{}]'.format(correct, n_samples_train), end = '   ')

        if validation_loader:
            model.eval()     
                       
            val_loss, val_corr = 0, 0
            for val_images, val_labels in validation_loader:
                if device:
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)

                outputs = model(val_images)
                loss = loss_func(outputs, val_labels)
                _, predictions = outputs.max(1)
                val_corr += int(sum(predictions == val_labels))
                val_loss += loss.item()


            loss_val = val_loss / n_batches_val
            accuracy_val = val_corr / n_samples_val

			validation_loss.append(loss_val)
			validation_accuracy.append(accuracy_val)


			print('Validation accuracy [{:.4f}], Validation Loss: {:.4f}'
                 .format(accuracy_val, loss_val))


	if test:
		
		test_loader = torch.utils.data.DataLoader(test, batch_size = 64, shuffle= True, drop_last = True)   
		correct = 0
		total = 0

		for images, labels in test_loader:
			if device:
				images = images.to(device)
				labels = labels.to(device)

			n_data = images[0]
			total += n_data
			outputs = model(images)
			predictions = outputs.argmax(1)
			correct += int(sum(predictions == labels))

		accuracy = correct / total 
		print('Test Accuracy: {}'.format(accuracy))


	model_save_name = 'vit.pt'
	path = F"/content/gdrive/My Drive/{model_save_name}" 
	torch.save(model.state_dict(), path)

    

    return {'loss': losses, 'accuracy': accuracy, 
            'val_loss': validation_loss, 'val_accuracy': validation_accuracy}




def test_func(test_loader, model, device = None):
	'''
	Eliminated--> Ino bordam toye function train.
	'''
	correct = 0
	total = 0

	for images, labels in test_loader:
		if device:
			images = images.to(device)
			labels = labels.to(device)

		n_data = images.shape[0]
		total += n_data
		outputs = model(images)
		predictions = outputs.argmax(1)
		correct += int(sum(predictions == labels))


	accuracy = correct / total 
	print('Test Accuracy: {}'.format(accuracy))

	return accuracy




def count_params(model):
	'''Returns the number of parameters of a model'''

	return sum([params.numel() for params in model.parameters() if params.requires_grad == True])



if __name__ == '__main__'

	# NORMALIZING OUR DATA
	imgs = torch.stack([img for img, _ in cifar10_tensor], dim = 3)
	mean = imgs.view(3, -1).mean(dim = 1)
	std = imgs.view(3, -1).std(dim = 1)


	transformed_cifar10 = datasets.CIFAR10(path, train = True, 
	                                       transform = transforms.Compose([transforms.ToTensor(), 
	                                                                       transforms.Normalize(mean, std)]))
	transformed_cifar10_test = datasets.CIFAR10(path, train = False, 
	                                            transform = transforms.Compose([transforms.ToTensor(), 
	                                                                            transforms.Normalize(mean, std)]))	


	validation, test = torch.utils.data.random_split(transformed_cifar10_test, [2000, 8000])


	train_loader = torch.utils.data.DataLoader(transformed_cifar10, batch_size = 64, shuffle = True, drop_last = True)
	test_loader = torch.utils.data.DataLoader(test, batch_size = 64, shuffle= True, drop_last = True)   
	val_loader = torch.utils.data.DataLoader(validation, batch_size = 64, shuffle = True, drop_last = True) 


	device = torch.device('cuda' if torch.cuda_is_available() else 'cpu')
	model = ViT().to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-4)
	scheduler = ReduceLROnPlateau(optimizer, 'min')
	history = train_func(train_loader, model, optimizer, loss_func = criterion)

	test_accuracy = test_func(test_loader, model)









