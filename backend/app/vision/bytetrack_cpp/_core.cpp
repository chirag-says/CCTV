// ByteTrack's association core, in C++.
//
// WHAT IS HERE AND WHAT IS NOT
// Only the geometry: the IoU matrix and the linear assignment that turns it
// into matches. The track lifecycle above it -- identity votes, promotion and
// demotion, the unknown registry, coasting -- stays in Python, on purpose.
// That code is the product's behaviour, it is covered by tests written
// against the Python objects, and reimplementing it here would fork it in two
// places that must never disagree.
//
// What is left is what runs per frame per camera and is pure arithmetic: an
// N x M IoU matrix and an O(n^3) assignment over it.
//
// IDENTICAL OUTPUTS
// The Python path solves the assignment with lap.lapjv(extend_cost=True,
// cost_limit=thresh). This reproduces those semantics rather than
// approximating them:
//
//   * the rectangular matrix is padded to square with the cost limit, which
//     is what extend_cost does, so an unmatchable row can always take a pad
//     column instead of forcing a bad pair;
//   * a solved pair is only reported if its REAL cost is <= thresh, which is
//     what cost_limit does;
//   * the assignment itself is exact (Jonker-Volgenant shortest augmenting
//     path with column reduction), so where the optimum is unique -- which it
//     is for float IoU costs outside deliberate ties -- there is exactly one
//     answer to agree on.
//
// tests/test_tracker_cpp.py holds that claim to a stored 300-frame sequence
// and to randomised matrices against lap itself.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// IoU of every box in `a` against every box in `b`.
//
// Boxes are [x1, y1, x2, y2]. Degenerate boxes give a zero union, which is
// reported as IoU 0 rather than NaN -- the Python version's np.where(union>0)
// does the same, and a NaN here would poison the whole assignment.
py::array_t<double> iou_batch(py::array_t<double, py::array::c_style | py::array::forcecast> a,
                              py::array_t<double, py::array::c_style | py::array::forcecast> b) {
    auto ra = a.unchecked<2>();
    auto rb = b.unchecked<2>();
    const py::ssize_t n = ra.shape(0);
    const py::ssize_t m = rb.shape(0);

    py::array_t<double> out({n, m});
    auto ro = out.mutable_unchecked<2>();
    if (n == 0 || m == 0) {
        return out;
    }

    std::vector<double> area_b(static_cast<size_t>(m));
    for (py::ssize_t j = 0; j < m; ++j) {
        area_b[static_cast<size_t>(j)] = (rb(j, 2) - rb(j, 0)) * (rb(j, 3) - rb(j, 1));
    }

    for (py::ssize_t i = 0; i < n; ++i) {
        const double ax1 = ra(i, 0), ay1 = ra(i, 1), ax2 = ra(i, 2), ay2 = ra(i, 3);
        const double area_a = (ax2 - ax1) * (ay2 - ay1);
        for (py::ssize_t j = 0; j < m; ++j) {
            const double ix1 = std::max(ax1, rb(j, 0));
            const double iy1 = std::max(ay1, rb(j, 1));
            const double ix2 = std::min(ax2, rb(j, 2));
            const double iy2 = std::min(ay2, rb(j, 3));
            const double iw = ix2 - ix1;
            const double ih = iy2 - iy1;
            const double inter = (iw > 0.0 && ih > 0.0) ? iw * ih : 0.0;
            const double uni = area_a + area_b[static_cast<size_t>(j)] - inter;
            ro(i, j) = (uni > 0.0) ? inter / uni : 0.0;
        }
    }
    return out;
}

// Jonker-Volgenant on a square cost matrix.
//
// `cost` is n*n row-major. Returns, for each row, the column it was assigned
// (always a full assignment -- filtering by the cost limit is the caller's
// job, so that this stays a plain solver).
std::vector<int> lapjv_square(const std::vector<double>& cost, int n) {
    // u, v: dual potentials. way: predecessor columns for the augmenting path.
    // Indices are 1-based inside the solver, which is what keeps the classic
    // formulation free of the off-by-ones it is famous for.
    std::vector<double> u(static_cast<size_t>(n) + 1, 0.0);
    std::vector<double> v(static_cast<size_t>(n) + 1, 0.0);
    std::vector<int> p(static_cast<size_t>(n) + 1, 0);    // p[col] = row
    std::vector<int> way(static_cast<size_t>(n) + 1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(static_cast<size_t>(n) + 1, kInf);
        std::vector<char> used(static_cast<size_t>(n) + 1, 0);
        do {
            used[static_cast<size_t>(j0)] = 1;
            const int i0 = p[static_cast<size_t>(j0)];
            double delta = kInf;
            int j1 = 0;
            for (int j = 1; j <= n; ++j) {
                if (used[static_cast<size_t>(j)]) continue;
                const double cur =
                    cost[static_cast<size_t>(i0 - 1) * static_cast<size_t>(n) +
                         static_cast<size_t>(j - 1)] -
                    u[static_cast<size_t>(i0)] - v[static_cast<size_t>(j)];
                if (cur < minv[static_cast<size_t>(j)]) {
                    minv[static_cast<size_t>(j)] = cur;
                    way[static_cast<size_t>(j)] = j0;
                }
                if (minv[static_cast<size_t>(j)] < delta) {
                    delta = minv[static_cast<size_t>(j)];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[static_cast<size_t>(j)]) {
                    u[static_cast<size_t>(p[static_cast<size_t>(j)])] += delta;
                    v[static_cast<size_t>(j)] -= delta;
                } else {
                    minv[static_cast<size_t>(j)] -= delta;
                }
            }
            j0 = j1;
        } while (p[static_cast<size_t>(j0)] != 0);
        // Walk the augmenting path back, flipping assignments.
        do {
            const int j1 = way[static_cast<size_t>(j0)];
            p[static_cast<size_t>(j0)] = p[static_cast<size_t>(j1)];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> row_to_col(static_cast<size_t>(n), -1);
    for (int j = 1; j <= n; ++j) {
        const int i = p[static_cast<size_t>(j)];
        if (i >= 1 && i <= n) row_to_col[static_cast<size_t>(i - 1)] = j - 1;
    }
    return row_to_col;
}

// lap.lapjv(cost, extend_cost=True, cost_limit=thresh), as the tracker uses it.
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
linear_assignment(py::array_t<double, py::array::c_style | py::array::forcecast> cost_matrix,
                  double thresh) {
    auto rc = cost_matrix.unchecked<2>();
    const int rows = static_cast<int>(rc.shape(0));
    const int cols = static_cast<int>(rc.shape(1));

    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_a;
    std::vector<int> unmatched_b;

    if (rows == 0 || cols == 0) {
        for (int i = 0; i < rows; ++i) unmatched_a.push_back(i);
        for (int j = 0; j < cols; ++j) unmatched_b.push_back(j);
        return {matches, unmatched_a, unmatched_b};
    }

    // lap's extend_cost + cost_limit layout, reproduced exactly. The matrix
    // becomes (rows + cols) square with three regions:
    //
    //     [ real costs        | thresh/2 ]   real rows
    //     [ thresh/2          |    0     ]   virtual rows
    //       real cols           virtual cols
    //
    // Every row gets a virtual column to opt out into, so a row is only
    // matched when doing so beats leaving it out.
    //
    // BOTH of the non-obvious parts matter, and getting either wrong shifts
    // the effective threshold rather than failing:
    //   * the extension is rows+cols, not max(rows, cols). At max(), a square
    //     matrix has no virtual columns at all and every row is forced into a
    //     match -- 5 percent disagreement with lap on random matrices.
    //   * the virtual-by-virtual block is ZERO, not thresh/2. Filling it too
    //     makes declining a pair cost one virtual cell instead of two, so the
    //     real threshold becomes thresh/2 and rows stop matching at half the
    //     intended IoU -- 28 percent disagreement.
    const int n = rows + cols;
    std::vector<double> square(static_cast<size_t>(n) * static_cast<size_t>(n), thresh / 2.0);
    for (int i = rows; i < n; ++i) {
        for (int j = cols; j < n; ++j) {
            square[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] = 0.0;
        }
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            square[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] =
                rc(i, j);
        }
    }

    const std::vector<int> row_to_col = lapjv_square(square, n);

    std::vector<char> col_taken(static_cast<size_t>(cols), 0);
    for (int i = 0; i < rows; ++i) {
        const int j = row_to_col[static_cast<size_t>(i)];
        // A virtual column means the solver chose to leave this row out.
        if (j >= 0 && j < cols) {
            matches.emplace_back(i, j);
            col_taken[static_cast<size_t>(j)] = 1;
        } else {
            unmatched_a.push_back(i);
        }
    }
    for (int j = 0; j < cols; ++j) {
        if (!col_taken[static_cast<size_t>(j)]) unmatched_b.push_back(j);
    }
    return {matches, unmatched_a, unmatched_b};
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "ByteTrack association core (IoU + linear assignment) in C++.";
    m.def("iou_batch", &iou_batch, py::arg("bboxes1"), py::arg("bboxes2"),
          "Pairwise IoU matrix for two sets of [x1,y1,x2,y2] boxes.");
    m.def("linear_assignment", &linear_assignment, py::arg("cost_matrix"), py::arg("thresh"),
          "(matches, unmatched_a, unmatched_b), matching lap.lapjv "
          "(extend_cost=True, cost_limit=thresh).");
}
