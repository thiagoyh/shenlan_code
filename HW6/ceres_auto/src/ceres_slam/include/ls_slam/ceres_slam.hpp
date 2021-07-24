#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <iostream>

template <typename T>
Eigen::Matrix<T, 3, 3> PoseToTrans(Eigen::Matrix<T, 3, 1> xi)
{
    Eigen::Matrix<T, 3, 3> Xi;
    Xi << cos(xi(2)), -sin(xi(2)), xi(0),
        sin(xi(2)), cos(xi(2)), xi(1),
        T(0), T(0), T(1);
    return Xi;
}

//转换矩阵－－＞位姿
template <typename T>
Eigen::Matrix<T, 3, 1> TransToPose(Eigen::Matrix<T, 3, 3> trans)
{
    Eigen::Matrix<T, 3, 1> pose;
    pose(0) = trans(0, 2);
    pose(1) = trans(1, 2);
    pose(2) = atan2(trans(1, 0), trans(0, 0));
    return pose;
}

class AutoDiffFunctor
{
public:
    AutoDiffFunctor(const Edge &edge) : measurement(edge.measurement), sqrt_info_matrix(edge.infoMatrix.array().sqrt()) {}

    template <typename T>
    bool operator()(const T *const v1, const T *const v2, T *residual) const
    {

        Eigen::Matrix<T, 3, 1> xi{v1[0], v1[1], v1[2]};
        Eigen::Matrix<T, 3, 1> xj{v2[0], v2[1], v2[2]};
        Eigen::Matrix<T, 3, 1> m{T(measurement(0)), T(measurement(1)), T(measurement(2))};
        // Eigen::Map<Eigen::Matrix<T, 3, 1>> error{residual};
        Eigen::Matrix<T, 3, 1> error{residual[0], residual[1], residual[2]};
        // calculate error from translation and rotation respectively and combine them together
        Eigen::Matrix<T, 3, 3> Xi = PoseToTrans(xi);
        Eigen::Matrix<T, 3, 3> Xj = PoseToTrans(xj);
        Eigen::Matrix<T, 3, 3> Z = PoseToTrans(m);

        Eigen::Matrix<T, 3, 3> Ei = Z.inverse() * Xi.inverse() * Xj;
        error = TransToPose(Ei);
        error = sqrt_info_matrix.template cast<T>() * error;

        residual[0] = error(0);
        residual[1] = error(1);
        residual[2] = error(2);

        return true;
    }

    static ceres::CostFunction *create(const Edge &edge)
    {
        return (new ceres::AutoDiffCostFunction<AutoDiffFunctor, 3, 3, 3>(new AutoDiffFunctor(edge)));
    }

private:
    Eigen::Vector3d measurement;
    Eigen::Matrix3d sqrt_info_matrix;
};