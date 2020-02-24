import numpy as np

# or-tools convention
status2str = ["OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNBOUNDED", 
              "ABNORMAL", "MODEL_INVALID", "NOT_SOLVED"]

debug = False
def dbg(s0, *s):
    if debug:
        print(s0, *s)

def pivot(T, pc, pr):
    #print("pc:", pc, "pr:", pr)
    pe = T[pr, pc] # pivot element
    pivot_row = T[pr, :] * 1.0 # stupid numpy copy gotcha
    pivot_row /= pe
    offset = np.dot(T[:, pc].reshape([-1, 1]), pivot_row.reshape([1, -1]))
    T -= offset
    T[pr, :] = pivot_row
    return T


def select_pivot_column(z):
    """
    Pick one variable that increases the objective

    for i, zi in enumerate(z):
            return i + 1
    else:
        return None
    """
    tol = 1e-8

    positive = np.where(z > tol)[0]
    if len(positive):
        #print("len(positive):", len(positive))
        #return positive[0] + 1 #
        return np.random.choice(positive)
    else:
        return None


def select_pivot_row(Tc, b):
    """
    Which ceiling are we going to hit our head in first?
    """

    tol = 1e-8

    #print(Tc)
    if all(Tc <= 0): # no roof over our head - to the stars!
        return None, 0

    ratios = [bi / Tci if Tci > tol else np.inf for Tci, bi in zip(Tc, b)]
    return np.argmin(ratios), min(ratios)


def collect_solution(T, basic):
    num_slack = len(basic)
    num_vars = len(T[0]) - num_slack - 1
    b = T[1:, -1]

    solution = np.zeros([num_vars])

    for pr, pc in enumerate(basic):
        if pc < num_slack: # is a slack variable
            continue

        solution[pc - num_slack] = T[pr + 1, -1] / T[pr + 1, pc]

    return solution


def lp(A, b, c):
    """
    maximize c * x 
    such that
    Ax <= b
    x >= 0
    b >= 0
    """

    # build a tableau T
    #T = [1 -c 0 0;
    #     0  A I b]
    # where I corresponds to s, slack variables

    # Loop, terminate when no pivot column can be selected

        # select pivot column
        # given pivot column, select pivot row
        # given pivot element, perform pivot operation

    # need to keep track of:
    #   which variables are basic

    # returning the solution: identify basic variables among original variables
    # set nonbasic variables to 0

    phase = 1
    skip = 2
    b_sgn = np.sign(b) >= 0
    sgn = 2*b_sgn - 1

    num_slack, num_vars = A.shape
    z_s = np.zeros([num_slack])
    z_v = np.zeros([num_vars])

    T1 = np.hstack([z_s, z_v, np.array([0])])
    T2 = np.hstack([z_s,   c, np.array([0])])
    T3 = np.hstack([np.eye(num_slack), A, b])
    T3 = T3 * sgn
    #print(T3)

    #print(T3[:, num_slack:-1] * (1 - b_sgn))
    #print(T3[:, -1] * (1 - b_sgn).T)
    T1[num_slack:-1] = np.sum(T3[:, num_slack:-1] * (1 - b_sgn), axis=0)
    T1[-1] = np.sum(T3[:, -1] * (1 - b_sgn).T)
    #exit(0)
    #return 1, 1, 1

    T = np.vstack([T1, T2, T3])
    #print(T)
    #exit(0)
    #return 1, 1, 1

    #dbg(T)

    basic = list(range(num_slack))
    #print("slack vars in base0:", len(np.where(np.array(basic) < num_slack)[0]))

    #for i in range(100000):
    while True:
        #print(T)

        b = T[skip:, -1]
        if np.any(b < -1e-8):
            print(b)
            raise Exception("b < 0")

        pc = select_pivot_column(T[0, :-1])
        if pc is None: # found optimum
            if phase == 1:
                if T[0, -1] <= 0:
                    print("found feasible")
                    phase = 2
                    T = T[1:, :] # cut the first row
                    skip = 1
                    continue
                else:
                    print("residual score:", T[0, -1])
                    return None, None, 2
            else:
                print("found optimum")
                break
        #print(pc, "{0:.5f}".format(T[0, pc]))


        pr, pivot_dist = select_pivot_row(T[skip:, pc], T[skip:, -1])
        if pr is None: # unbounded
            return None, np.inf, 3
        #if pivot_dist == 0.0: # will not increase objective
        #    continue
        #print("pc:", pc)
        #print("pr:", pr)
        #print()

        T = pivot(T, pc, pr + skip)
        #print(T)

        basic[pr] = pc
        #print("objective: {}".format(-T[0, -1]))
        #[print(row) for row in T]
        #print(T.shape)
        #print("slack vars in base:", len(np.where(np.array(basic) < num_slack)[0]))
    else:
        print("iteration limit reached")

    #print(basic)
    return collect_solution(T, basic), -T[0, -1], 0


def main():
    global debug
    debug = False

    all_tests = ["basic_1", "basic_2", "basic_3",
                 "basic_4", "basic_5", "basic_6",
                  "infeasible_1", "infeasible_2",
                  "unbounded"]

    #test_cases = ["basic_3"]
    test_cases = all_tests

    if "basic_1" in test_cases:
        A = np.array([[2, 1], [1, 2]])
        b = np.array([[1], [1]])
        c = np.array([1, 1])
        print("Should be OPTIMAL")
        x_opt, opt_val, status = lp(A, b, c)
        print(x_opt, opt_val, status2str[status])
        print()

    if "basic_2" in test_cases:
        A = np.array([[2, 1], [1, 2], [4, 4]])
        b = np.array([[1], [1], [1]])
        c = np.array([1, 1])
        print("Should be OPTIMAL")
        x_opt, opt_val, status = lp(A, b, c)
        print(x_opt, opt_val, status2str[status])
        print()

    if "basic_3" in test_cases:
        A = np.array([[2, 1], [1, 2], [-2, -2]])
        b = np.array([[1], [1], [-1]])
        c = np.array([1, 1])
        print("Should be OPTIMAL")
        x_opt, opt_val, status = lp(A, b, c)
        print(x_opt, opt_val, status2str[status])
        print()

    if "basic_4" in test_cases:
        A = np.array([[3, 1], [1, 3], [2, 3]])
        b = np.array([[1], [1], [1]])
        c = np.array([1, 1])
        print("Should be OPTIMAL")
        x_opt, opt_val, status = lp(A, b, c)
        print(x_opt, opt_val, status2str[status])
        print()

    if "basic_5" in test_cases:
        # this should take just two pivots
        A = np.array([[-1, 1], [1, -1], [1, 1]])
        b = np.array([[1], [1], [10]])
        c = np.array([1, 1])
        print("Should be OPTIMAL")
        x_opt, opt_val, status = lp(A, b, c)
        print(x_opt, opt_val, status2str[status])
        print()

    if "basic_6" in test_cases:
        A = np.array([[-1, 1], [1, -1], [1, 1]])
        b = np.array([[1], [1], [1]])
        c = np.array([1, 1])
        print("Should be OPTIMAL")
        x_opt, opt_val, status = lp(A, b, c)
        print(x_opt, opt_val, status2str[status])
        print()

    if "infeasible_1" in test_cases:
        A = np.array([[2, 1], [1, 2], [1, 1]])
        b = np.array([[1], [1], [-1]])
        c = np.array([1, 1])
        print("Should be INFEASIBLE")
        x_opt, opt_val, status = lp(A, b, c)
        print(x_opt, opt_val, status2str[status])
        print()

    if "infeasible_2" in test_cases:
        A = np.array([[2, 1], [1, 2], [-1, -1]])
        b = np.array([[1], [1], [-1]])
        c = np.array([1, 1])
        print("Should be INFEASIBLE")
        x_opt, opt_val, status = lp(A, b, c)
        print(x_opt, opt_val, status2str[status])
        print()

    if "unbounded" in test_cases:
        A = np.array([[-1, -1]])
        b = np.array([[-1]])
        c = np.array([1, 1])
        print("Should be UNBOUNDED")
        x_opt, opt_val, status = lp(A, b, c)
        print(x_opt, opt_val, status2str[status])
        print()
        

if __name__ == '__main__':
    main()