class sparse_matrix():
	def __init__(self,r,c):
		self.r = r
		self.c = c
		self.mat = {}

	def addelem(self,s,a,val):
		self.mat[s,a] = val

	def T(self):
		newmat = {}
		for key in self.mat:
			newmat[key[1],key[0]] = mat[key]

		trans = sparse_matrix(self.c,self.r)
		trans.mat = newmat
		return trans

	def mult(self,b):
		newmat = {}

		if(self.c != b.r):
			print("Invalid operation")
			return

		res = sparse_matrix(self.r,b.c)

		for key1 in self.mat:
			data_a = self.mat[key1]

			for key2 in b.mat:
				data_b = b.mat[key2]
				if(key1[1] == key2[0]):
					if((key1[0],key2[1]) in newmat.keys()):
						newmat[key1[0],key2[1]] += data_a*data_b
					else:
						newmat[key1[0],key2[1]] = data_a*data_b

		res.mat = {key:val for key, val in newmat.items() if val != 0}
		return res

	def print_mat(self):
		for key in self.mat:
			print(str(key[0]) + str(key[1]) + " : " + str(self.mat[key])) 
