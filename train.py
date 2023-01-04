import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):

	def __init__(self, img_size, patch_size, embed_dim, in_channels = 3, early_cnn = True):
		super(PatchEmbedding, self).__init__() 
		self.img_size = img_size 
		self.patch_size = patch_size 
		self.n_patches = (img_size // patch_size) ** 2
		self.early_cnn = early_cnn

		self.cnn = nn.Conv2d(in_channels = in_channels, out_channels = 3, kernel_size = 3, padding = 1)
		self.bn = nn.BatchNorm2d(3)
		self.relu = nn.ReLU()
		self.transform = nn.Conv2d(in_channels = in_channels, out_channels = embed_dim, 
								   kernel_size = patch_size, stride = patch_size) 


	def forward(self, x):

		#(n, 3, 32, 32) doesn't change the size ofthe image
		if self.early_cnn :
			x = self.cnn(x)  
		x = self.bn(x)
		x = self.relu(x)

		x = self.transform(x) #(n_samples, embed_dim, width, height)
		x = x.flatten(2)
		x = x.transpose(1, 2) #(n_samples, n_patches, embed_dim)

		return x



class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads = 8, proj_drop = 0.1):

	super(SelfAttention, self).__init__()
	self.dim = dim 
	self.n_heads = n_heads 
	self.head_dim = dim // n_heads 
	self.scale = n_heads ** -0.5

	self.query = nn.Linear(dim, dim)
	self.key = nn.Linear(dim, dim)
	self.value = nn.Linear(dim, dim)
	self.fc_out = nn.Linear(dim, dim)
	self.fc_drop = nn.Dropout(proj_drop)



    def forward(self, x):
	n_samples, n_patches, dim = x.shape	

	assert dim == self.dim, 'dim should be equal to the dimension declared in the constructor'

	q = self.query(x)  #Each with dim: (n_samples, n_patches + 1, dim)
	k = self.key(x)
	v = self.value(x)


	q = q.reshape(n_samples, n_patches, self.n_heads, self.head_dim) #(n_samples, n_patches, self.n_heads, self.head_dim)
	k = k.reshape(n_samples, n_patches, self.n_heads, self.head_dim)      
	v = v.reshape(n_samples, n_patches, self.n_heads, self.head_dim)

	q = q.permute(0, 2, 1, 3) #(n_samples, n_heads, n_patches, head_dim)
	k = k.permute(0, 2, 1, 3)
	v = v.permute(0, 2, 1, 3)

	k_t = k.transpose(-1, -2) #(n_samples, n_heads, head_dim, n_patches)

	weights = (torch.matmul(q, k_t)) * self.scale #(n_samples, n_heads, n_patches, n_patches)

	scores = weights.softmax(dim = -1)

	weighted_avg = scores @ v  #(n_smples, n_heads, n_patches, head_dim)
	weighted_avg = weighted_avg.transpose(1, 2) #(n_smples, n_patches, n_heads, head_dim)

	weighted_avg = weighted_avg.flatten(2) #(n_samples, n_patches, n_heads*head_dim)

	x = self.fc_out(weighted_avg)
	x = self.fc_drop(x) #checked

	return x 



class MLP(nn.Module):

    def __init__(self, in_features, out_features, drop_mlp):
	super(MLP, self).__init__()

	self.fc1 = nn.Linear(in_features, out_features)
	self.dropout = nn.Dropout(drop_mlp)
	self.fc2 = nn.Linear(out_features, out_features)
	self.gelu = nn.GELU()


    def forward(self, x):
	x = self.fc1(x)  #(n_samples, n_patches, hidden_features)
	x = self.gelu(x)
	x = self.dropout(x)
	x = self.fc2(x)
	#x = self.gelu(x)  Why NOT???? IN paper
	x = self.dropout(x)

	return x 



class Block(nn.Module):

    def __init__(self,  dim, n_heads, p_drop = 0.1):
	super(Block, self).__init__()

	self.norm1 = nn.LayerNorm(dim)
	self.norm2 = nn.LayerNorm(dim)
	self.attention = SelfAttention(dim = dim, n_heads = n_heads)	 

	self.mlp = MLP(in_features = dim, out_features = dim, drop_mlp = p_drop)



    def forward(self, x):
	x = x + self.attention(self.norm1(x))
	x = x + self.mlp(self.norm2(x))

	return x   




class ViT(nn.Module):

    def __init__(self, img_size, patch_size = 16, in_channels = 3, n_classes = 10, embed_dim = 768, 
		layers = 6, n_heads = 12, p_drop = 0, early_cnn = True):
	super(ViT, self).__init__()

	self.patch_embed = PatchEmbedding(img_size = img_size, patch_size = patch_size, embed_dim = embed_dim, early_cnn = early_cnn)
	self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
	self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.patch_embed.n_patches, embed_dim)) 


	self.pos_drop = nn.Dropout(p_drop)

	self.blocks = nn.ModuleList([
	    Block(dim = embed_dim, n_heads = n_heads, p_drop = p_drop)
		    for _ in range(layers)])

	self.norm = nn.LayerNorm(embed_dim)
	self.head = nn.Linear(in_features = embed_dim, out_features = n_classes)




    def forward(self, x):
	n_samples = x.shape[0]

	x = self.patch_embed(x) #(n_samples, n_pathces, embed_dim)

	cls_token = self.cls_token.expand(n_samples, -1, -1) #(n_samples, 1, embed_dim)

	x = torch.cat([cls_token, x], dim = 1) #(n_samples, 1+n_patches, embed_dim)
	x = x + self.pos_embed	
	x = self.pos_drop(x)

	for block in self.blocks:
	    x = block(x) 

	x = self.norm(x) 

	output = x[:, 0]
	x = self.head(output)

	return x

