#include "matrix.h"
#include <random>
#include <stdexcept>

// Constructor that copies data from an array
Matrix::Matrix(float* matrix_rw, int n_s, int m_s) : n(n_s), m(m_s) {
    matrix = new float[n * m];
    for (int i = 0; i < n * m; i++){
        matrix[i] = matrix_rw[i];
    }
}

// Constructor for an empty matrix
Matrix::Matrix(int n_s, int m_s) : n(n_s), m(m_s) {
    matrix = new float[n * m]();
}

// Destructor
Matrix::~Matrix() {
    delete[] matrix;
}

// Retrieve a row from the matrix
Matrix* Matrix::get_row(int i) {
    Matrix* row = new Matrix(m, 1);
    for (int j = 0; j < m; j++) {
        row->matrix[j] = matrix[i * m + j];
    }
    return row;
}

// Generate a random matrix
Matrix* Matrix::randomMatrix(int n_s, int m_s, float min_val, float max_val) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> distribution(min_val, max_val);

    Matrix* result = new Matrix(n_s, m_s);
    for (int i = 0; i < n_s * m_s; i++) {
        result->matrix[i] = distribution(generator);
    }
    return result;
}

// Transpose the matrix
Matrix* Matrix::transpose(Matrix& matrix) {
    Matrix* A = new Matrix(matrix.m, matrix.n);
    for (int i = 0; i < matrix.n; i++) {
        for (int j = 0; j < matrix.m; j++) {
            A->matrix[j * A->m + i] = matrix.matrix[i * matrix.m + j];
        }
    }
    return A;
}

// Hadamard product of two matrices
Matrix* Matrix::hadamard(Matrix& a, Matrix& b) {
    if (a.n != b.n || a.m != b.m) throw std::invalid_argument("Incompatible dimensions for Hadamard product.");

    Matrix* A = new Matrix(a.n, a.m);
    for (int i = 0; i < a.n * a.m; i++) {
        A->matrix[i] = a.matrix[i] * b.matrix[i];
    }
    return A;
}

// Matrix multiplication
Matrix* Matrix::operator*(const Matrix& other) {
    if (m != other.n){
        std::cout<< n << ' ' << m <<' '<< other.n << ' ' << other.m <<'\n';
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication.");
    }

    Matrix* A = new Matrix(n, other.m);
    for (int i = 0; i < A->n; i++) {
        for (int j = 0; j < A->m; j++) {
            for (int k = 0; k < m; k++) {
                A->matrix[i * A->m + j] += matrix[i * m + k] * other.matrix[k * other.m + j];
            }
        }
    }
    return A;
}

// Matrix addition
Matrix* Matrix::operator+(const Matrix& other) {
    if (n != other.n || m != other.m) throw std::invalid_argument("Incompatible dimensions for matrix addition.");

    Matrix* A = new Matrix(n, m);
    for (int i = 0; i < n * m; i++) {
        A->matrix[i] = matrix[i] + other.matrix[i];
    }
    return A;
}

// Matrix subtraction
Matrix *Matrix::operator-(const Matrix& other) {
    if (n != other.n || m != other.m) throw std::invalid_argument("Incompatible dimensions for matrix subtraction.");

    Matrix* A = new Matrix(n, m);
    for (int i = 0; i < n * m; i++) {
        A->matrix[i] = matrix[i] - other.matrix[i];
    }
    return A;
}

// Matrix scaling
Matrix* Matrix::operator*(float scale) {
    Matrix* A = new Matrix(n, m);
    for (int i = 0; i < n * m; i++) {
        A->matrix[i] = scale * matrix[i];
    }
    return A;
}

// Scalar * matrix multiplication
Matrix* operator*(float scale, Matrix& other) {
    return other * scale;
}

// Print matrix
std::ostream& operator<<(std::ostream& COUT, const Matrix& matrix) {
    for (int i = 0; i < matrix.n; ++i) {
        for (int j = 0; j < matrix.m; ++j) {
            COUT << matrix.matrix[i * matrix.m + j] << " ";
        }
        COUT << "\n";
    }
    return COUT;
}
