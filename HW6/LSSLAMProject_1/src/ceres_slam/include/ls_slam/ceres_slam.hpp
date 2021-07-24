#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <iostream>
#include <gaussian_newton.h>

class AnalyticDiffFunction : public ceres::SizedCostFunction<3, 3, 3>
{
public:
    virtual ~AnalyticDiffFunction() {}

    AnalyticDiffFunction(const Edge &edge) : measurement(edge.measurement), sqrt_info_matrix(edge.infoMatrix.array().sqrt()) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {

        Eigen::Vector3d xi{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Vector3d xj{parameters[1][0], parameters[1][1], parameters[1][2]};

        Eigen::Map<Eigen::Vector3d> error_ij{residuals};
        Eigen::Matrix3d Ai;
        Eigen::Matrix3d Bi;

        Eigen::Matrix3d Xi = PoseToTrans(xi);
        Eigen::Matrix2d Ri = Xi.block(0, 0, 2, 2);
        Eigen::Vector2d ti{xi(0), xi(1)};

        Eigen::Matrix3d Xj = PoseToTrans(xj);
        Eigen::Matrix2d Rj = Xj.block(0, 0, 2, 2);
        Eigen::Vector2d tj{xj(0), xj(1)};

        Eigen::Matrix3d Z = PoseToTrans(measurement);
        Eigen::Matrix2d Rij = Z.block(0, 0, 2, 2);
        Eigen::Vector2d tij{measurement(0), measurement(1)};

        Eigen::Matrix2d dRiT_dtheta;       //  derivative of Ri^T over theta
        dRiT_dtheta(0, 0) = -1 * Ri(1, 0); //  cosX -> -sinX
        dRiT_dtheta(0, 1) = 1 * Ri(0, 0);  //  sinX ->  cosX
        dRiT_dtheta(1, 0) = -1 * Ri(0, 0); // -sinX -> -cosX
        dRiT_dtheta(1, 1) = -1 * Ri(1, 0); //  cosX -> -sinX

        // calcuate error & normalize error on theta
        error_ij.segment<2>(0) = Rij.transpose() * (Ri.transpose() * (tj - ti) - tij);
        error_ij(2) = xj(2) - xi(2) - measurement(2);
        if (error_ij(2) > M_PI)
        {
            error_ij(2) -= 2 * M_PI;
        }
        else if (error_ij(2) < -1 * M_PI)
        {
            error_ij(2) += 2 * M_PI;
        }

        Ai.setZero();
        Ai.block(0, 0, 2, 2) = -Rij.transpose() * Ri.transpose();
        Ai.block(0, 2, 2, 1) = Rij.transpose() * dRiT_dtheta * (tj - ti);
        Ai(2, 2) = -1.0;

        Bi.setIdentity();
        Bi.block(0, 0, 2, 2) = Rij.transpose() * Ri.transpose();

        error_ij = sqrt_info_matrix * error_ij;

        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_xi(jacobians[0]);
                jacobian_xi = sqrt_info_matrix * Ai;
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_xj(jacobians[1]);
                jacobian_xj = sqrt_info_matrix * Bi;
            }
        }

        return true;
    }

private:
    Eigen::Vector3d measurement;
    Eigen::Matrix3d sqrt_info_matrix;
};