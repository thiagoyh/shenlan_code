#pragma once

#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <eigen3/Eigen/Core>
#include <cmath>
#include <chrono>

#include <gaussian_newton.h>

class Slam_Vertex : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginImpl()
    {
        _estimate << 0, 0, 0;
    }

    virtual void oplusImpl(const double *update)
    {
        _estimate += Eigen::Vector3d(update);
    }

    //存盘和
    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}
};
