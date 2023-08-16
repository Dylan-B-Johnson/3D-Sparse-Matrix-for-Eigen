// Copyright 2023
// Dylan Johnson
#ifndef SRC_SPARSE_MAT_3D_H_
#define SRC_SPARSE_MAT_3D_H_
#include<Eigen>
#include<vector>

// A sparse 3d matrix class for Eigen 3.4 that allows
// for vector-matrix multiplication resulting
// in a 2d matrix, as specified
// https://math.stackexchange.com/questions/409460/multiplying-3d-matrix.
// This stype of structure is also implemented when one
// performes matrix multiplication on 3d NumPy Arrays
// (although it is dense).
// Also supports matrix addition / subtraction.
class SparseMat3D {
 public:
    // Constructs an N x N x N SparseMat3D object initialized with 0s
    // @param n   The size of the cubic matrix object
    explicit SparseMat3D(int n);
    // Constructs an N x N x N SparseMat3D object initialized with 0s
    // @param n1   The size of the first dimension of the sparse 3d
    // matrix object
    // @param n2   The size of the second dimension of the sparse 3d
    // matrix object
    // @param n3   The size of the third dimension of the sparse 3d
    // matrix object
    SparseMat3D(int n1, int n2, int n3);
    // Constructs a deep copy of a SparseMat3D object
    // @param toCopy The SparseMat3D object to create a duplicate of
    SparseMat3D(const SparseMat3D & toCopy);
    // Default SparseMat3D destructor
    ~SparseMat3D() = default;
    // Sets the left-hand-side SparseMat3D object to be a deep copy of the
    // right-hand-side SparseMat3D object
    // @param toCopy The right-hand-side SparseMat3D object
    // that the left-hand-side will be set equal to
    // @return Returns the left-hand-side SparseMat3D objec, which is now
    // equal to the right-hand-side SparseMat3D object
    const SparseMat3D& operator=(const SparseMat3D & toCopy);
    // Adds the element-wise sum of the left-hand-side
    // and right-hand-side SparseMat3D objects
    // @param rhs The right-hand-side SparseMat3D
    // @return A new SpraseMat3D object that is the sum of the
    // left-hand and right-hand sides
    const SparseMat3D operator+(const SparseMat3D & rhs) const;
    // Subtracts (element-wise) the right-hand-side SparseMat3D
    // object's entries from the left-hand-side SparseMat3D object
    // @param rhs The right-hand-side SparseMat3D
    // @return A new SparseMat3D object that is the sum of
    // the left-hand and right-hand sides
    const SparseMat3D operator-(const SparseMat3D & rhs) const;
    // Computes the element-wise sum of the left-hand-side
    // and right-hand-side SparseMat3D objects and
    // sets it to be the left-hand-side SparseMat3D object.
    // @param rhs The right-hand-side SparseMat3D
    // @return The resulting element-wise sum of the
    // left-hand-side and right-hand-side SparseMat3D objects
    const SparseMat3D& operator+=(const SparseMat3D & rhs);
    // Subtracts (element-wise) the right-hand-side
    // SparseMat3D object's entries from the left-hand-side
    // SparseMat3D object and sets it to be the left-hand-side SparseMat3D
    // object.
    // @param rhs The right-hand-side SparseMat3D
    // @return The resulting element-wise difference of
    // the left-hand-side and right-hand-side SparseMat3D objects
    const SparseMat3D& operator-=(const SparseMat3D & rhs);
    // Checks if this SparseMat3D is identical to the
    // right-hand-side SparseMat3D object
    // @param rhs The right-hand-side SparseMat3D object
    // @return True iff the dimensions and elements of the
    // right-hand-side SparseMat3D object are the
    // same as this SparseMat3D object
    bool operator==(const SparseMat3D & rhs) const;
    // Indexes into the SparseMat3D object, and returns a reference, allowing
    // for access and mutation.
    // E.g. a_SparseMat3D_object(2,2,3) = 2.4;
    // or a_double = a_SparseMat3D_object(2,2,3);
    // @param a   The index (from 0) into the first
    // dimension of the SparseMat3D object
    // @param b   The index (from 0) into the second
    // dimension of the SparseMat3D object
    // @param c   The index (from 0) into the third
    // dimension of the SparseMat3D object
    // @return The entry of the SparseMat3D object at a,b,c.
    double & operator()(int a, int b, int c);
    // Gets the size of the first dimension of the SparseMat3D object
    // @return The size of the first dimension of the SparseMat3D object
    int GetN1() const;
    // Gets the size of the second dimension of the SparseMat3D object
    // @return The size of the second dimension of the SparseMat3D object
    int GetN2() const;
    // Gets the size of the third dimension of the SparseMat3D object
    // @return The size of the third dimension of the SparseMat3D object
    int GetN3() const;
    // Gets the dimensions of the SparseMat3D object as a vector of ints
    // @return An int vector of length 3, containing the size of the first,
    // second, and third dimensions of the SparseMat3D object
    std::vector<int> GetShape() const;
    // Multiplies this SparseMat3D object with the passed
    // in vector on the right and returns the result.
    // This mirrors the NumPy 3D-Array-with-2D-vector-Array
    // matrix multiplcication behavior.
    // Mathematically it consists of applying right multiplication
    // N times to N different 2D matricies and forming a 2D matrix
    // where each column is the result of the respective matrix multiplication
    // (where N is the size of the first dimension of 3D the matrix).
    // @param vec The vector to multiply this SparseMat3D
    // object by. It must be N3 x 1 in dimension (where N3 is
    // the size of the third dimension of this SparseMat3D object).
    // @return The resulting 2D sparse matrix of the
    // right-matrix multiplication.
    Eigen::SparseMatrix<double> RhtMatMul(const Eigen::MatrixXd & vec) const;
    // Multiplies this SparseMat3D object with the passed in
    // vector on the right and returns the result. This mirrors
    // the NumPy 3D-Array-with-2D-vector-Array matrix multiplcication behavior.
    // Mathematically it consists of applying right multiplication
    // N times to N different 2D matricies and forming a 2D matrix
    // where each column is the result of the respective matrix multiplication
    // (where N is the size of the first dimension of the 3D matrix).
    // @param vec The vector to multiply this SparseMat3D object by.
    // It must be N3 x 1 in dimension (where N3 is the size
    // of the third dimension of this SparseMat3D object).
    // @return The resulting 2D sparse matrix of the
    // right-matrix multiplication.
    Eigen::SparseMatrix<double> operator*(const Eigen::MatrixXd & vec) const;
    // Multiplies this SparseMat3D object with the passed
    // in vector on the left and returns the result. This
    // mirrors the NumPy 3D-Array-with-2D-vector-Array matrix
    // multiplcication behavior. Mathematically it consists of
    // applying left multiplication N times to N different 2D matricies
    // and forming a 2D matrix where each row is the result
    // of the respective matrix multiplication (where
    // N is the size of the first dimension of the 3D matrix).
    // @param vec The vector to multiply this SparseMat3D object by.
    // It must be 1 x N2 in dimension (where N2 is the size
    // of the second dimension of this SparseMat3D object).
    // @return The resulting 2D sparse matrix of the left-matrix multiplication.
    Eigen::SparseMatrix<double> LftMatMul(const Eigen::MatrixXd & vec) const;
    // Multiplies this SparseMat3D object with the passed
    // in vector on the left and returns the result. This
    // mirrors the NumPy 3D-Array-with-2D-vector-Array matrix
    // multiplcication behavior. Mathematically it consists of
    // applying left multiplication N times to N different 2D matricies
    // and forming a 2D matrix where each row is the result of
    // the respective matrix multiplication (where N is the size of
    // the first dimension of the 3D matrix).
    // @param vec The vector to multiply the SparseMat3D object
    // by. It must be 1 x N2 in dimension (where N2 is the size
    // of the second dimension of this SparseMat3D object).
    // @param mat The SparseMat3D object that is being
    // multiplied by the vector on the left
    // @return The resulting 2D sparse matrix of the left-matrix multiplication.
    friend Eigen::SparseMatrix<double> operator*(const Eigen::MatrixXd & vec,
      const SparseMat3D & mat);
    // Sets all enteries in the matrix to 0, but does not
    // free any memory (ZeroFree() does)
    void Zero();
    // Sets all enteries in the matrix to 0 and frees allocated memory.
    void ZeroFree();
    // Returns a specific entry of the SparseMat3D object
    // @param a   The index (from 0) into the first dimension
    // of the SparseMat3D object
    // @param b   The index (from 0) into the second dimension
    // of the SparseMat3D object
    // @param c   The index (from 0) into the third dimension
    // of the SparseMat3D object
    // @return The entry in the SparseMat3D object at (a,b,c)
    double Get(int a, int b, int c) const;
    // Sets a specific entry of the SparseMat3D object to the passed in double
    // @param a   The index (from 0) into the first dimension
    // of the SparseMat3D object
    // @param b   The index (from 0) into the second dimension
    // of the SparseMat3D object
    // @param c   The index (from 0) into the third dimension
    // of the SparseMat3D object
    // @param val The value to assing (a,b,c)th entry
    void Set(int a, int b, int c, double val);

 private:
    std::vector<Eigen::SparseMatrix<double>> data_;
    int n1_, n2_, n3_;
};

#endif  // SRC_SPARSE_MAT_3D_H_
