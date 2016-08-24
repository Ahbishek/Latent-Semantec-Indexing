#!/usr/bin/python

# reference => http://www.puffinwarellc.com/index.php/news-and-articles/articles/33.html
import numpy as np
from numpy import zeros
from scipy.linalg import svd
from math import log	# needed for TFIDF
from numpy import asarray, sum
import matplotlib.pyplot as plt	
'''
titles = ["The Neatest Little Guide to Stock Market Investing",
		"Investing For Dummies, 4th Edition",
		"The Little Book of Common Sense Investing: The Only Way to Guarantee Your Fair Share of Stock Market Returns",
		"The Little Book of Value Investing",
		"Value Investing: From Graham to Buffett and Beyond",
		"Rich Dad's Guide to Investing: What the Rich Invest in, That the Poor and the Middle Class Do Not!",
		"Investing in Real Estate, 5th Edition",
		"Stock Investing For Dummies",
		"Rich Dad's Advisors: The ABC's of Real Estate Investing: The Secrets of Finding Hidden Profits Most Investors Miss"
		]
'''
titles = ["A survey of user opinion of computer system response time","Human machine interface for Lab ABC computer applications",
		"The EPS user interface management system",
		"System and human system engineering testing of EPS",
		"Relation of user-perceived response time to error measurement",
		"The generation of random, binary, unordered trees",
		"The intersection graph of paths in trees",
		"Graph minors IV: Widths of trees and well-quasi-ordering",
		
]
sizen=len(titles)	
querysearch =["human system engineering for computer management and applications"]

#stopwords = ['and','edition','for','in','little','of','the','to','of','a']
#stopwords = ['pict']
######################################################human computer interaction
stopwords=[]
    
f=open('stopword.txt')

for line in f :

    line=line[0:-1]
    stopwords.append(line)
f.close()
##############################################################

ignorechars = ''',:'!'''

class LSA(object):
	def __init__(self, stopwords, ignorechars):
		self.stopwords = stopwords
		self.ignorechars = ignorechars
		self.wdict = {}
		self.wdict1 = {}
		self.dcount = 0
		self.dcount1 = 0

	def parse(self, doc):
		#print self.stopwords
		words = doc.split();
		for w in words:
			w = w.lower().translate(None, self.ignorechars)
			if w in self.stopwords:
				continue
			elif w in self.wdict:
				self.wdict[w].append(self.dcount)
			else:
				self.wdict[w] = [self.dcount]
				#print w
			print "   "
		self.dcount += 1
		print "_______________________________________________________________________"



	####for query q
	def parseq(self,doc1):
		words1 = doc1.split();
		for w in self.wdict:
			if len(self.wdict[w]) > 1:
				self.wdict1[w]=[0]
		
		for w in words1:
			w = w.lower().translate(None, self.ignorechars)
			if w in self.stopwords:
				continue
			elif w in self.wdict1:
				#print "word in wdict"
				#self.wdict1[w] += 1			
				self.wdict1[w].append(self.dcount1)
				#self.wdict1[w]=[1]
			else:
				print ""
				#self.wdict1[w] = [self.dcount1]
			print w
			print "   "
		self.dcount1 += 1
		print "_______________________________________________________________________"
		
		#for w in self.wdict1:
		#	for w2 in w:
		#		print w2
			



	####
	# rows -> keywords (occur more than twice), cols -> documentID


	def buildq(self):
		
		self.keys1 = [k for k in self.wdict1.keys() if len(self.wdict1[k]) > 0]
		self.keys1.sort()
		self.Q = zeros([len(self.keys1), self.dcount1])
		for i, k in enumerate(self.keys1):
			for d in self.wdict1[k]:
				self.Q[i,d] += 1
		
		
		



####   buildq:- q  ka seprate A formation 
	
	def build(self):
		#print self.wdict
		self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
		self.keys.sort()
		self.A = zeros([len(self.keys), self.dcount])
		for i, k in enumerate(self.keys):
			for d in self.wdict[k]:
				self.A[i,d] += 1
		



	
	
	def calc(self):
		self.U, self.S, self.Vt = svd(self.A)
	
	def TFIDF(self):
		WordsPerDoc = sum(self.A, axis=0)        
		DocsPerWord = sum(asarray(self.A > 0, 'i'), axis=1)
		rows, cols = self.A.shape
		for i in range(rows):
			for j in range(cols):
				self.A[i,j] = (self.A[i,j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i])
	
	def printA(self):
		print "----------------------------------------------------------------------"
		print 'Here is the count matrix'
		print self.A
		print "\n \n \n \n \n \n \n \n"
		print "-----------------------------------------------------------------------"

	def printQ(self):
		print '\n \n \n \n Here is the Query search matrix'
		#print self.Q
		self.new_Q = []
		for i in self.Q:
			for j in i:
				j-=1
				self.new_Q.append(j)
		print self.new_Q
		print "\n \n \n \n"
		print "--------------------------------------------------------------------------"
	def printSVD(self):
		print "---------------------------------------------------------------------------"
		#print 'Here are the singular values'
		#print self.S
		l = len(self.S)
		self.new_S = [[] for i in range(l)]
		for i in self.new_S:
			for j in range(l):
				i.append(0)
		ind = 0
		#print self.new_S
		for i in range(l):
			for j in range(l):
				if i == j:
					self.new_S[i][j] = 1/self.S[ind]
					ind += 1
		print "\n \n \n \n"
		print "--------------------------------------------------------------------------"
		#print self.new_S
		print 'Here are  U matrix'
		print self.U[:, 0:sizen]
		print "\n \n \n \n"
		print "--------------------------------------------------------------------------"
		print 'Here are  Vt matrix'
		print self.Vt[0:sizen, :]
		
	def TFIDF(self):
		WordsPerDoc = sum(self.A, axis=0) 
		DocsPerWord = sum(asarray(self.A > 0, 'i'), axis=1) 
		rows, cols = self.A.shape 
		for i in range(rows):
			for j in range(cols):
				self.A[i,j] = (self.A[i,j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i])
	def newdocvectorcordinates(self):
		print "\n \n \n \n"
		print "--------------------------------------------------------------------------"
		#print 'DOcument vector coordinates'
		self.Vk =  self.Vt[0:sizen, 0:2]
		#print self.Vk
		self.Uk = self.U[:, 0:2]
		print "\n \n \n \n _____________________________________________________________________________________"		
		print "  printing  UK "
		
		print self.Uk[:,0:2]
		self.Sk= self.new_S[0:2]
		print "\n \n \n \n"
		print "--------------------------------------------------------------------------"
		print "  printing  SK "
		self.new_Sk = []
		for i in self.Sk:
			i = i[0:2]
			self.new_Sk.append(i)
		print self.new_Sk
		self.nSk = np.matrix(self.new_Sk)
		self.nUk = np.matrix(self.Uk)
		
		self.nVk = np.matrix(self.Vk)
		self.nVk=self.nVk.transpose()
		self.Xhihat =self.nUk*self.nSk*self.nVk
		print "--------------------------------------------------------------------------- \n \n \n"
		print " printing  Xhihat"
		print self.Xhihat
 



	def step_5(self):
		self.Qt = [self.new_Q[j] for j in range(len(self.new_Q))] 
		self.Qt = np.matrix(self.Qt)
		self.Sk = np.matrix(self.new_Sk)
		print "\n \n \n \n \n qt"
		print self.Qt
		print "\n \n \n \n \n uk"
		print self.Uk
		print "\n \n \n \n sk"
		print self.Sk
		self.final = self.Qt * self.Uk * self.Sk
		print "\n \n \n \n \n \n QUERY COORDINATES \n"
		print self.final
		print "\n \n \n \n \n"
		x = np.array(self.final)[0].tolist()
		#print type(x)
		mod_x = np.linalg.norm(x)
		values = []
		print "--------------------------------------Document Coordinates------------------------------"
		for i in self.Vk:
			print i
			values.append(np.dot(i,x) / (mod_x * np.linalg.norm(i)))
		print "\n \n \n \n \n--------------------------------------------------------------------------------------------"
		print "Rank of the documents are as follows"
		print values
		new_index=values.index(max(values))
		print "\n \n \n \n \n \n DOcument to be fetched"
		print titles[new_index]
		print "\n \n \n \n \n \n \n \n \n \n \n \n \n \n \n"
		x_coor = []
		y_coor = []
		for i in values:
			y_coor.append(i)
		for i in range(1,9):
			x_coor.append(i)
		n=["d1","d2","d3","d4","d5","d6","d7","d8"]
		fig,ax = plt.subplots()
		ax.scatter(x_coor,y_coor)	
		for i, txt in enumerate(n):
			ax.annotate(txt,(x_coor[i],y_coor[i]))
		plt.show()

	
		
	def step_6(self):
		x_coor = []
		y_coor = []
		for i in self.Vk:
			#print i[0]
			x_coor.append(i[0])
		x = np.array(self.final)[0].tolist()
		#print x[0]
		x_coor.append(x[0])
		for i in self.Vk:
			#print i[0]
			y_coor.append(i[1])
		y_coor.append(x[1])
		#plt.plot(x_coor,y_coor, 'ro')
		#plt.axis([0, 1, 0, 1])
		#plt.show()
		n=["d1","d2","d3","d4","d5","d6","d7","d8","q1"]
		fig,ax = plt.subplots()
		ax.scatter(x_coor,y_coor)	
		for i, txt in enumerate(n):
			ax.annotate(txt,(x_coor[i],y_coor[i]))
		plt.show()

	@staticmethod
	def main():
		mylsa = LSA(stopwords, ignorechars)
		for t in titles:
			mylsa.parse(t)
		for t1 in querysearch:
			mylsa.parseq(t1)		
		

		mylsa.build()
		mylsa.buildq()
		
		mylsa.printA()
		mylsa.printQ()
		mylsa.calc()
		mylsa.printSVD()
		mylsa.newdocvectorcordinates()

		mylsa.step_5()
		mylsa.step_6()

if __name__ == '__main__':
	LSA.main()
