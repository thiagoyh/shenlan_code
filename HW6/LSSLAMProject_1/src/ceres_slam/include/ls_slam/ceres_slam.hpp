#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <iostream>
#include <gaussian_newton.h>

class optimization
{
public:
    optimization(const Edge &edge) : measurement(edge.measurement), infoMatrix(edge.infoMatrix) {}

    template <typename T>
    bool operator()(const T *const xi_, const T *const xj_, T *residual) const
    {
        Eigen::Matrix<T, 3, 1> xi{xi_[0], xi_[1], xi_[2]};
        Eigen::Matrix<T, 3, 1> xj{xj_[0], xj_[1], xj_[2]};
        //Eigen::Matrix<T, 3, 1> z= measurement.template cast<T>();
        Eigen::Matrix<T, 3, 1> error;
        Eigen::Matrix<T, 3, 1> z;
        z << T(measurement(0)), T(measurement(1)), T(measurement(2));

        Eigen::Matrix<T, 3, 3> infoMatrix_;
        infoMatrix_ << T(infoMatrix(0, 0)), T(infoMatrix(0, 1)), T(infoMatrix(0, 2)),
            T(infoMatrix(1, 0)), T(infoMatrix(1, 1)), T(infoMatrix(1, 2)),
            T(infoMatrix(2, 0)), T(infoMatrix(2, 1)), T(infoMatrix(2, 2));

        Eigen::Matrix<T, 3, 3> Xi = PoseToTrans(xi);
        Eigen::Matrix<T, 3, 3> Xj = PoseToTrans(xj);
        Eigen::Matrix<T, 3, 3> Z = PoseToTrans(z);

        Eigen::Matrix<T, 2, 2> Ri = Xi.block(0, 0, 2, 2);
        Eigen::Matrix<T, 2, 2> Rj = Xj.block(0, 0, 2, 2);
        Eigen::Matrix<T, 2, 2> Rij = Z.block(0, 0, 2, 2);

        Eigen::Matrix<T, 3, 3> Ei = Z.inverse() * Xi.inverse() * Xj;
        Eigen::Vector3d pose;

        Eigen::Matrix<T, 3, 1> ei；
            pose(0) = trans(0, 2);
        pose(1) = trans(1, 2);
        pose(2) = atan2(trans(1, 0), trans(0, 0));

        //Eigen::Matrix<T, 3, 1> error;
        //error << T(ei(0)), T(ei(1)), T(ei(2));
        error(0) = T(ei(0));
        error(1) = T(ei(1));
        error(2) = T(ei(2));

        error = infoMatrix_ * error;

        residual[0] = T(error(0));
        residual[1] = T(error(1));
        residual[2] = T(error(2));

        return true;
    }

    static ceres::CostFunction *create(const Edge &edge)
    {
        return (new ceres::AutoDiffCostFunction<optimization, 3, 3, 3>(new optimization(edge)));
    }

private:
    Eigen::Vector3d measurement;
    Eigen::Matrix3d infoMatrix;
};
