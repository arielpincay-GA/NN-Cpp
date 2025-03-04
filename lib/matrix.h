#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>

class Matrix {
public:
    float* matrix; // Pointer to store matrix data
    int n;         // Number of rows
    int m;         // Number of columns

    // Constructors
    Matrix(float* matrix_rw, int n_s, int m_s); // Constructor that copies data from an array
    Matrix(int n_s, int m_s);                  // Constructor for an empty matrix

    // Destructor
    ~Matrix();

    // Methods
    Matrix *get_row(int i); // Retrieve a row from the matrix
    static Matrix* randomMatrix(int n_s, int m_s, float min_val = 0.0, float max_val = 1.0);
    static Matrix* transpose(Matrix& matrix); // Transpose the matrix
    static Matrix* hadamard(Matrix& a, Matrix& b); // Hadamard product of two matrices

    // Overloaded operators
    Matrix* operator*(const Matrix& other); // Matrix multiplication
    Matrix* operator*(float scale);         // Matrix scaling
    Matrix* operator+(const Matrix& other); // Matrix addition
    Matrix* operator-(const Matrix& other); // Matrix subtraction
};

// Overload of the * operator for scalar * matrix multiplication
Matrix* operator*(float scale, Matrix& other);

// Overload of the << operator to print the matrix
std::ostream& operator<<(std::ostream& COUT, const Matrix& matrix);

#endif // MATRIX_H
