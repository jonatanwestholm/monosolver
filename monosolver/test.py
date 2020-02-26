from sentian_miami import get_solver
from lp import lp

def main():
    solver = get_solver("mono")

    x = solver.NumVar(lb=0)
    y = solver.NumVar(lb=0)

    solver.Add(2*x + y >= 1)
    solver.Add(x + 2*y >= 1)
    z = x + y

    solver.SetObjective(z, maximize=False)

    solver.Solve(time_limit=10)

    print("x: {0:.3f}".format(solver.solution_value(x)))
    print("y: {0:.3f}".format(solver.solution_value(y)))
    print("z: {0:.3f}".format(solver.solution_value(z)))


if __name__ == '__main__':
    main()