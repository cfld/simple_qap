#pragma GCC diagnostic ignored "-Wunused-result"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <iostream>
#include <omp.h>
#include <chrono>

using namespace std::chrono;

// --
// Global defs

typedef int Int;
typedef int Real;

void swap(int *a, int *b)  { 
    int temp = *a; 
    *a = *b; 
    *b = temp; 
} 

void permute(Int* x, Int n)  { 
  for (Int i = n - 1; i > 0; i--)  { 
    Int j = rand() % (i + 1); 
    swap(&x[i], &x[j]); 
  }
} 

void perturb(Int* perm, Int n) {
  Int i, j, k;
  
  i = rand() % n;
  j = rand() % n;
  k = rand() % n;
  
  while(j == i           ) j = rand() % n;
  while(k == i || k == j ) k = rand() % n;
  
  swap(perm + i, perm + j); // i gets j, i now in j
  swap(perm + j, perm + k); // j gets k, i now in k
}

Real compute_score(Real* A, Real* B, Int n, Int* perm) {
  Real acc = 0;
  for(Int i = 0; i < n; i++) {
    for(Int j = 0; j < n; j++) {
      acc += A[i * n + j] * B[perm[i] * n + perm[j]];
    }
  }
  return acc;
}

Real update_score(const Real* A, const Real* B, Int n, Int* perm, Int i, Int j) {
  Real old_row_i = 0;  
  Real old_row_j = 0;
  Real new_row_i = 0;
  Real new_row_j = 0;
  
  Real d_i = 0;
  Real d_j = 0;
  for(Int c = 0; c < n; c++) {
    Int old_c = perm[c];
    Int new_c;
    if(c == i) {
      new_c = perm[j];
    } else if(c == j) {
      new_c = perm[i];
    } else {
      new_c = old_c;
    }
    
    old_row_i += A[i * n + c] * B[perm[i] * n + old_c];
    old_row_j += A[j * n + c] * B[perm[j] * n + old_c];
    
    new_row_i += A[i * n + c] * B[perm[j] * n + new_c];
    new_row_j += A[j * n + c] * B[perm[i] * n + new_c];
  }
  
  Real old_col_i = 0;
  Real old_col_j = 0;
  Real new_col_i = 0;
  Real new_col_j = 0;
    
  Real d_x = 0;
  for(Int r = 0; r < n; r++) {
    if(r == i) continue;
    if(r == j) continue;
    
    d_x += (A[r * n + i] - A[r * n + j]) * (B[perm[r] * n + perm[j]] - B[perm[r] * n + perm[i]]);
  }
  
  return (new_row_i - old_row_i) + (new_row_j - old_row_j) + d_x;
}


// Real update_score(Real* A, Real* B, Int n, Int* perm, Int i, Int j) {
//   Real d_i = 0;
//   Real d_j = 0;
//   for(Int c = 0; c < n; c++) {
//     Int p_c;
//     if(c == i) {
//       p_c = perm[j];
//     } else if(c == j) {
//       p_c = perm[i];
//     } else {
//       p_c = perm[c];
//     }
    
//     d_i += A[i * n + c] * (B[perm[j] * n + p_c] - B[perm[i] * n + perm[c]]);
//     d_j += A[j * n + c] * (B[perm[i] * n + p_c] - B[perm[j] * n + perm[c]]);
//   }
  
//   Real d_x = 0;
//   for(Int r = 0; r < n; r++) {
//     if(r == i) continue;
//     if(r == j) continue;
//     d_x += (A[r * n + i] - A[r * n + j]) * (B[perm[r] * n + perm[j]] - B[perm[r] * n + perm[i]]);
//   }
  
//   return d_i + d_j + d_x;
// }

Real run_one(Real* A, Real* B, Int n, Int* gperm, Int piters) {
    Int* perm      = (Int*)malloc(n * sizeof(Int));
    Int* best_perm = (Int*)malloc(n * sizeof(Int));
    Int* idx0      = (Int*)malloc(n * sizeof(Int));
    Int* idx1      = (Int*)malloc(n * sizeof(Int));
    
    for(Int i = 0; i < n; i++) perm[i] = i;
    for(Int i = 0; i < n; i++) idx0[i] = i;
    for(Int i = 0; i < n; i++) idx1[i] = i;
    
    // permute(perm, n);
    
    Real score = compute_score(A, B, n, perm);
    // std::cout << "init_score: " << score << std::endl;
    Real best_score = score;
    memcpy(best_perm, perm, n * sizeof(Int));
    
    int counter      = 0;
    int last_improve = 0;
    
    auto t1 = high_resolution_clock::now();
    for(Int piter = 0; piter < piters; piter++) {
      
      // Local minimum (2opt)
      bool done = false;
      while(!done) {
        done = true;
        
        permute(idx0, n);
        permute(idx1, n);
        
        for(Int ridx = 0; ridx < n; ridx++) {
          // if(score == 0) break;
          // if(counter - last_improve > 5000) break;
          Int x0 = idx0[ridx];
          
          for(Int cidx = 0; cidx < n; cidx++) {
            // if(score == 0) break;
            // if(counter - last_improve > 5000) break;

            Int x1 = idx1[cidx];
            
            counter += 1;

            Real delta = update_score(A, B, n, perm, x0, x1);
            
            if(delta > 0) {
              score += delta;
              done  = false;
              last_improve = counter;
              auto elapsed = high_resolution_clock::now() - t1;
              long long ms = duration_cast<microseconds>(elapsed).count();
              // std::cout << score << " " << ms << " " << counter << std::endl;
              swap(perm + x0, perm + x1);
            } 
          }
        }
      }
      
      // Store best result
      if(score > best_score) {
        best_score = score;
        memcpy(best_perm, perm, n * sizeof(Int));
      } else {
        memcpy(perm, best_perm, n * sizeof(Int));
      }
      
      // if(score == 0) break;
      
      // Perturb
      perturb(perm, n);
      score = compute_score(A, B, n, perm);
    }
    
    memcpy(gperm, best_perm, n * sizeof(Int));
    
    free(perm);
    free(best_perm);
    free(idx0);
    free(idx1);
    
    // std::cout << "final: " << score << " " << counter << std::endl;
    return best_score;
}


// --
// Run

void run_all(Real* A, Real* B, Int n, Int* output, Int piters, Int popsize, bool verbose) {
    Real* scores = (Real*)malloc(popsize * sizeof(Real));
    for(Int i = 0; i < popsize; i++) {
      scores[i] = -1; //std::numeric_limits<Int>::min(); // !!
    }
    
    Int* perms = (Int*)malloc(popsize * n * sizeof(Int));
    for(Int i = 0; i < popsize * n; i++) perms[i] = i % n;
    
    // --
    // Run
    
    #pragma omp parallel for
    for(Int i = 0; i < popsize; i++) {
      scores[i] = run_one(A, B, n, perms + i * n, piters);
      if(verbose) {
        std::cout << "scores[" << i << "]=" << scores[i] << std::endl;
      }
    }
    
    // --
    // Get best result
    
    Real best_score = scores[0];
    Int  best_idx   = 0;
    for(Int i = 0; i < popsize; i++) {
      if(scores[i] > best_score) {
        best_score = scores[i];
        best_idx   = i;
      }
    }
    
    memcpy(output, perms + best_idx * n, n * sizeof(Int));
}

py::array_t<Int> qap(py::array_t<Real> A_arr, py::array_t<Real> B_arr, Int piters, Int popsize, Int seed, bool verbose) {
    if(seed == -1) {
      srand(time(NULL));
    } else {
      srand(seed);
    }
    
    py::buffer_info A_buf = A_arr.request();
    py::buffer_info B_buf = B_arr.request();

    Int n  = A_buf.shape[0];
    Real *A = static_cast<Real *>(A_buf.ptr);
    Real *B = static_cast<Real *>(B_buf.ptr);

    auto output_arr = py::array_t<Int>(n);
    py::buffer_info output_buff = output_arr.request();
    
    Int *perm = static_cast<Int *>(output_buff.ptr);    
    
    run_all(A, B, n, perm, piters, popsize, verbose);
    
    return output_arr;
}

PYBIND11_MODULE(simple_qap, m) {
    m.def("qap", &qap, "Simple QAP solver", 
      py::arg("A"),
      py::arg("B"),
      py::arg("piters")  = 1, 
      py::arg("popsize") = 1,
      py::arg("seed")    = -1,
      py::arg("verbose") = false
    );
}