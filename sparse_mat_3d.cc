// Copyright 2023
// Dylan Johnson
#ifndef SRC_SPARSE_MAT_3D_CC_
#define SRC_SPARSE_MAT_3D_CC_
#include<src/sparse_mat_3d.h>
#include<Eigen>
#include<iostream>
#include<vector>

// Constructs an N x N x N SparseMat3D object initialized with 0s
// @param n   The size of the cubic matrix object
SparseMat3D::SparseMat3D(int n) {
    n1_ = n2_ = n3_ = n;
    for (int i(0); i < n; i++) {
        Eigen::SparseMatrix<double> tmp(n, n);
        data_.push_back(tmp);
    }
    Zero();
}
// Constructs an N x N x N SparseMat3D object initialized with 0s
// @param n1   The size of the first dimension of the sparse 3d matrix object
// @param n2   The size of the second dimension of the sparse 3d matrix object
// @param n3   The size of the third dimension of the sparse 3d matrix object
SparseMat3D::SparseMat3D(int n1, int n2, int n3) {
    this->n1_ = n1;
    this->n2_ = n2;
    this->n3_ = n3;
    for (int i(0); i < n1; i++) {
        Eigen::SparseMatrix<double> tmp(n2_, n3_);
        data_.push_back(tmp);
    }
    Zero();
}
// Constructs a deep copy of a SparseMat3D object
// @param toCopy The SparseMat3D object to create a duplicate of
SparseMat3D::SparseMat3D(const SparseMat3D & toCopy) {
    this->n1_ = toCopy.n1_;
    this->n2_ = toCopy.n2_;
    this->n3_ = toCopy.n3_;
    for (int i(0); i < n1_; i++) {
        Eigen::SparseMatrix<double> tmp(n2_, n3_);
        data_.push_back(tmp);
    }
    Zero();
    for  (int i(0); i < n1_; i++) {
        for (int j(0); j < n2_; j++) {
            for (int k(0); k < n3_; k++) {
                if (toCopy.Get(i, j, k) != 0) {
                    data_[i].coeffRef(j, k) = toCopy.Get(i, j, k);
                }
            }
        }
    }
}
// Sets the left-hand-side SparseMat3D object to be a deep copy of the
// right-hand-side SparseMat3D object
// @param toCopy The right-hand-side SparseMat3D object that the left-hand-side
// will be set equal to
// @return Returns the left-hand-side SparseMat3D objec, which is now
// equal to the right-hand-side SparseMat3D object
const SparseMat3D& SparseMat3D::operator=(const SparseMat3D & toCopy) {
    this->n1_ = toCopy.n1_;
    this->n2_ = toCopy.n2_;
    this->n3_ = toCopy.n3_;
    data_.clear();
    for (int i(0); i < n1_; i++) {
        Eigen::SparseMatrix<double> tmp(n2_, n3_);
        data_.push_back(tmp);
    }
    Zero();
    for  (int i(0); i < n1_; i++) {
        for (int j(0); j < n2_; j++) {
            for (int k(0); k < n3_; k++) {
                if (toCopy.Get(i, j, k) != 0) {
                    data_[i].coeffRef(j, k) = toCopy.Get(i, j, k);
                }
            }
        }
    }
    return *this;
}
// Adds the element-wise sum of the left-hand-side
// and right-hand-side SparseMat3D objects
// @param rhs The right-hand-side SparseMat3D
// @return A new SpraseMat3D object that is the sum of the
// left-hand and right-hand sides
const SparseMat3D SparseMat3D::operator+(const SparseMat3D & rhs) const {
    if (rhs.n1_ != this->n1_ || rhs.n2_ != this->n2_ || rhs.n3_ != this->n3_) {
        std::cerr << "ERROR: Dimensions don\'t match." << std::endl;
        exit(1);
    }
    SparseMat3D rtn(n1_, n2_, n3_);
    for (int i(0); i < n1_; i++) {
        for (int j(0); j < n2_; j++) {
            for (int k(0); k < n3_; k++) {
                rtn.Set(i, j, k, this->Get(i, j, k) + rhs.Get(i, j, k));
            }
        }
    }
    return rtn;
}
// Subtracts (element-wise) the right-hand-side SparseMat3D
// object's entries from the left-hand-side SparseMat3D object
// @param rhs The right-hand-side SparseMat3D
// @return A new SparseMat3D object that is the sum of
// the left-hand and right-hand sides
const SparseMat3D SparseMat3D::operator-(const SparseMat3D & rhs) const {
    if (rhs.n1_ != this->n1_ || rhs.n2_ != this->n2_ || rhs.n3_ != this->n3_) {
        std::cerr << "ERROR: Dimensions don\'t match." << std::endl;
        exit(1);
    }
    SparseMat3D rtn(n1_, n2_, n3_);
    for (int i(0); i < n1_; i++) {
        for (int j(0); j < n2_; j++) {
            for (int k(0); k < n3_; k++) {
                rtn.Set(i, j, k, this->Get(i, j, k) - rhs.Get(i, j, k));
            }
        }
    }
    return rtn;
}
// Computes the element-wise sum of the left-hand-side
// and right-hand-side SparseMat3D objects and
// sets it to be the left-hand-side SparseMat3D object.
// @param rhs The right-hand-side SparseMat3D
// @return The resulting element-wise sum of the
// left-hand-side and right-hand-side SparseMat3D objects
const SparseMat3D& SparseMat3D::operator+=(const SparseMat3D & rhs) {
    return *this = *this + rhs;
}
// Subtracts (element-wise) the right-hand-side
// SparseMat3D object's entries from the left-hand-side
// SparseMat3D object and sets it to be the left-hand-side SparseMat3D
// object.
// @param rhs The right-hand-side SparseMat3D
// @return The resulting element-wise difference of
// the left-hand-side and right-hand-side SparseMat3D objects
const SparseMat3D& SparseMat3D::operator-=(const SparseMat3D & rhs) {
    return *this = *this - rhs;
}
// Checks if this SparseMat3D is identical to the
// right-hand-side SparseMat3D object
// @param rhs The right-hand-side SparseMat3D object
// @return True iff the dimensions and elements of the
// right-hand-side SparseMat3D object are the
// same as this SparseMat3D object
bool SparseMat3D::operator==(const SparseMat3D & rhs) const {
    if (rhs.n1_ != n1_ || rhs.n2_ != n2_ || rhs.n3_ != n3_) {
        return false;
    }
    for (int i(0); i < n1_; i++) {
        for (int j(0); j < n2_; j++) {
            for (int k(0); k < n3_; k++) {
                if (rhs.Get(i, j, k) != this->Get(i, j, k)) {
                    return false;
                }
            }
        }
    }
    return true;
}
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
double & SparseMat3D::operator()(int a, int b, int c) {
    return data_[a].coeffRef(b, c);
}
// Gets the size of the first dimension of the SparseMat3D object
// @return The size of the first dimension of the SparseMat3D object
int SparseMat3D::GetN1() const {
    return n1_;
}
// Gets the size of the second dimension of the SparseMat3D object
// @return The size of the second dimension of the SparseMat3D object
int SparseMat3D::GetN2() const {
    return n2_;
}
// Gets the size of the third dimension of the SparseMat3D object
// @return The size of the third dimension of the SparseMat3D object
int SparseMat3D::GetN3() const {
    return n3_;
}
// Gets the dimensions of the SparseMat3D object as a vector of ints
// @return An int vector of length 3, containing the size of the first,
// second, and third dimensions of the SparseMat3D object
std::vector<int> SparseMat3D::GetShape() const {
    std::vector<int> rtn;
    rtn.push_back(n1_);
    rtn.push_back(n2_);
    rtn.push_back(n3_);
    return rtn;
}
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
// @return The resulting 2D sparse matrix of the right-matrix multiplication.
Eigen::SparseMatrix<double> SparseMat3D::RhtMatMul(
        const Eigen::MatrixXd & vec) const {
    if (vec.rows() != n3_ || vec.cols() != 1) {
        if (vec.rows() == 1 && vec.cols() == n2_) {
            std::cerr << "ERROR: Dimensions don\'t match." <<
                " Did you mean to left multiply?" << std::endl;
        } else {
            std::cerr << "ERROR: Dimensions don\'t match." << std::endl;
        }
        exit(1);
    }
    Eigen::SparseMatrix<double> rtn(n1_, n2_);
    for (int i(0); i < n1_; i++) {
        Eigen::SparseMatrix<double> tmp = (data_[i] * vec).sparseView();
        for (int j(0); j < n2_; j++) {
            rtn.coeffRef(i, j) = tmp.coeff(j, 0);
        }
    }
    return rtn;
}
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
// @return The resulting 2D sparse matrix of the right-matrix multiplication.
Eigen::SparseMatrix<double> SparseMat3D::operator*(
        const Eigen::MatrixXd & vec) const {
    if (vec.rows() != n3_ || vec.cols() != 1) {
        if (vec.rows() == 1 && vec.cols() == n2_) {
            std::cerr << "ERROR: Dimensions don\'t match." <<
                " Did you mean to left multiply?" << std::endl;
        } else {
            std::cerr << "ERROR: Dimensions don\'t match." << std::endl;
        }
        exit(1);
    }
    Eigen::SparseMatrix<double> rtn(n1_, n2_);
    for (int i(0); i < n1_; i++) {
        Eigen::SparseMatrix<double> tmp = (data_[i] * vec).sparseView();
        for (int j(0); j < n2_; j++) {
            rtn.coeffRef(i, j) = tmp.coeff(j, 0);
        }
    }
    return rtn;
}
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
Eigen::SparseMatrix<double> SparseMat3D::LftMatMul(
        const Eigen::MatrixXd & vec) const {
    if (vec.rows() != 1 || vec.cols() != n2_) {
        if (vec.rows() == n3_ && vec.cols() == 1) {
            std::cerr << "ERROR: Dimensions don\'t match." <<
                " Did you mean to right multiply?" << std::endl;
        } else {
            std::cerr << "ERROR: Dimensions don\'t match." << std::endl;
        }
        exit(1);
    }
    Eigen::SparseMatrix<double> rtn(n1_, n3_);
    for (int i(0); i < n1_; i++) {
        Eigen::SparseMatrix<double> tmp = (vec * data_[i]).sparseView();
        for (int j(0); j < n3_; j++) {
            rtn.coeffRef(i, j) = tmp.coeff(0, j);
        }
    }
    return rtn;
}
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
Eigen::SparseMatrix<double> operator*(const Eigen::MatrixXd & vec,
        const SparseMat3D & mat) {
    if (vec.rows() != 1 || vec.cols() != mat.n2_) {
        if (vec.rows() == mat.n3_ && vec.cols() == 1) {
            std::cerr << "ERROR: Dimensions don\'t match." <<
                " Did you mean to right multiply?" << std::endl;
        } else {
            std::cerr << "ERROR: Dimensions don\'t match." << std::endl;
        }
        exit(1);
    }
    Eigen::SparseMatrix<double> rtn(mat.n1_, mat.n3_);
    for (int i(0); i < mat.n1_; i++) {
        Eigen::SparseMatrix<double> tmp = (vec * mat.data_[i]).sparseView();
        for (int j(0); j < mat.n3_; j++) {
            rtn.coeffRef(i, j) = tmp.coeff(0, j);
        }
    }
    return rtn;
}
// Sets all enteries in the matrix to 0, but does not
// free any memory (ZeroFree() does).
void SparseMat3D::Zero() {
    for (int i(0); i < n1_; i++) {
        data_[i].setZero();
    }
}
// Sets all enteries in the matrix to 0 and frees allocated memory.
void SparseMat3D::ZeroFree() {
    for (int i(0); i < n1_; i++) {
        data_[i].resize(n2_, n3_);
        data_[i].data().squeeze();
    }
}
// Returns a specific entry of the SparseMat3D object
// @param a   The index (from 0) into the first dimension
// of the SparseMat3D object
// @param b   The index (from 0) into the second dimension
// of the SparseMat3D object
// @param c   The index (from 0) into the third dimension
// of the SparseMat3D object
// @return The entry in the SparseMat3D object at (a,b,c)
double SparseMat3D::Get(int a, int b, int c) const {
    return data_[a].coeff(b, c);
}
// Sets a specific entry of the SparseMat3D object to the passed in double
// @param a   The index (from 0) into the first dimension
// of the SparseMat3D object
// @param b   The index (from 0) into the second dimension
// of the SparseMat3D object
// @param c   The index (from 0) into the third dimension
// of the SparseMat3D object
// @param val The value to assing (a,b,c)th entry
void SparseMat3D::Set(int a, int b, int c, double val) {
    data_[a].coeffRef(b, c) = val;
}

#endif  // SRC_SPARSE_MAT_3D_CC_
