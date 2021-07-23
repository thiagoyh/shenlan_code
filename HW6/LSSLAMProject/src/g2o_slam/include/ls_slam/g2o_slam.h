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

Eigen::Matrix3d PoseToTrans(Eigen::Vector3d x)
{
    Eigen::Matrix3d trans;
    trans << cos(x(2)), -sin(x(2)), x(0),
        sin(x(2)), cos(x(2)), x(1),
        0, 0, 1;

    return trans;
}

Eigen::Vector3d TransToPose(Eigen::Matrix3d trans)
{
    Eigen::Vector3d pose;
    pose(0) = trans(0, 2);
    pose(1) = trans(1, 2);
    pose(2) = atan2(trans(1, 0), trans(0, 0));

    return pose;
}

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

class Slam_Edge : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, Slam_Vertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Slam_Edge() : BaseUnaryEdge() {}
    virtual void computeError()
    {
        const Slam_Vertex *v_1 = static_cast<const Slam_Vertex *>(_vertices[0]);
        const Slam_Vertex *v_2 = static_cast<const Slam_Vertex *>(_vertices[1]);

        const Eigen::Vector3d xi = v_1->estimate();
        const Eigen::Vector3d xj = v_2->estimate();

        Eigen::Matrix3d Xi = PoseToTrans(xi);
        Eigen::Matrix3d Xj = PoseToTrans(xj);
        Eigen::Matrix3d Z = PoseToTrans(_measurement);

        Eigen::Matrix3d Ei = Z.inverse() * Xi.inverse() * Xj;

        _error = TransToPose(Ei);
    }

    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}
    // virtual void linearizeOplus()
    // {
    // }
};