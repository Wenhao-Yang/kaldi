cdef extern from "plda.cpp":
    pass

cdef extern from "plda.h" namespace "kaldi":
    cdef ccpclass Plda:
        Plda()
        Plda(Plda)
        double TransformIvector(PldaConfig, VectorBase<double>, int32, VectorBase<double>*)
        float TransformIvector(PldaConfig&, VectorBase<float>&, int32,VectorBase<float>*)
        double LogLikelihoodRatio(const VectorBase<double>&, int32, VectorBase<double>&)
    	void SmoothWithinClassCovariance(double)
        void ApplyTransform(const Matrix<double>&)
        int32 Dim()
        void Write(std::ostream&, bool)
        void Read(std::istream&, bool)

        void ComputeDerivedVars()

        Vector<double> mean_
        Matrix<double> transform_
        Vector<double> psi_
        Vector<double> offset_
        
        Plda &operator = (const Plda &other);  // disallow assignment
        double GetNormalizationFactor(VectorBase<double>&, int32)