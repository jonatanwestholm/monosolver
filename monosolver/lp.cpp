#include <iostream>
#include <limits>

#include "lp.h"

int select_pivot_column(Vector Tr){
  // next step is random pick

  int pc = -1;
  int i = 0;
  double max_c = 0;
  for(auto c : Tr){
    if(c >= max_c){
      max_c = c;
      pc = i;
    }
    i++;
  }
  return pc;
}

int select_pivot_row(Vector Tc, Vector b){
  /*
      """
    Which ceiling are we going to hit our head in first?
    """

    tol = 1e-8

    #print(Tc)
    if all(Tc <= 0): # no roof over our head - to the stars!
        return np.inf, None

    ratios = [bi / Tci if Tci > tol else np.inf for Tci, bi in zip(Tc, b)]
    return min(ratios), np.argmin(ratios)
  */
  bool some_blocker = false;
  for(auto c : Tc){
    if(c > 0){
      some_blocker = true;
      break;
    }
  }
  if(!some_blocker){
    return -1;
  }

  double min_ratio = std::numeric_limits<double>::max();
  int min_ratio_idx = -1;
  double tci;

  for(auto i : range(0, Tc.size())){
    tci = Tc[i];
    if(tci > 0 && b[i] / tci < min_ratio){
      min_ratio = b[i] / tci;
      min_ratio_idx = i;
    }
  }

  return min_ratio_idx;
}

void pivot(Matrix T, int pc, int pr, int shift){
  /*
    #print("pc:", pc, "pr:", pr)
    pe = T[pr, pc] # pivot element
    pivot_row = T[pr, :] * 1.0 # stupid numpy copy gotcha
    pivot_row /= pe
    offset = np.dot(T[:, pc].reshape([-1, 1]), pivot_row.reshape([1, -1]))
    T -= offset
    T[pr, :] = pivot_row
    return T
  */

  const auto pe = T[pr+shift][pc];

  int num_slack = T.size() - shift;
  int num_vars = T[0].size() - num_slack - 1;
  Vector Tc(num_slack+shift);
  Vector Tr(num_slack+num_vars+1);

  for(auto i : range(0, num_slack+shift)){
    Tc[i] = T[i][pc];
  }

  for(auto i : range(0, num_slack+num_vars+1)){
    Tr[i] = T[pr+shift][i] / pe;
  }

  for(auto i : range(0, num_slack+shift)){
    for(auto j : range(0, num_slack+num_vars+1)){
      T[i][j] -= Tc[i] * Tr[j];
    }
  }

  for(auto i : range(0, num_slack+num_vars+1)){
    T[pr+shift][i] = Tr[i];
  }
}


int lp(Matrix A, Vector b, Vector c){
  /*
  max c * x, s.t.
  A*x <= b
  x >= 0

  assuming b >= 0
  */

  // build tableau
  /*
  [0 c 0
   I A b]
  */

  const auto num_slack = b.size();
  const auto num_vars = c.size();

  int shift = 1;

  Matrix T = Matrix(num_slack+shift, Vector(num_slack+num_vars+1, 0));
  for(auto i : range(0, num_vars)){
    T[0][i+num_slack] = c[i];
  }

  for(auto i : range(0, num_slack)){
    T[i+shift][i] = 1;
    T[i+shift][num_slack+num_vars] = b[i];
  }

  for(auto i : range(0, num_slack)){
    for(auto j : range(0, num_vars)){
      T[i+shift][j+num_slack] = A[i][j];
    }
  }

  // main loop
  /*
  */
  int pr, pc;
  Vector Tc(num_slack);

  while(true){
    int pc = select_pivot_column(T[0]);
    if(pc < 0){
      break;
    }
    for(auto i : range(0, num_slack)){
      Tc[i] = T[i+shift][pc];
    } 
    int pr = select_pivot_row(Tc, b);
    pivot(T, pc, pr, shift);
  }

  return 0;
}

int main(){
  int num_vars = 2;
  int num_slack = 3;
  Matrix A(num_slack, Vector(num_vars, 4));
  Vector b(num_slack, 3);
  Vector c(num_vars, 2);

  //print_vector(b);
  //print_matrix(A);

  lp(A, b, c);


  return 0;
}