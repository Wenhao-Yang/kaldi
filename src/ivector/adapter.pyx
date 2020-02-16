# distutils: language = c++

from cplda cimport Plda

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.

cdef class PyPlda:
	cdef Plda c_myplda
	
	def __cinit__(self):
		self.c_myplda = Plda()

	def TransformIvector(self, config, ivector, num_examples, transformed_ivector):
		self.c_myplda.TransformIvector(config, ivector, num_examples, transformed_ivector)
