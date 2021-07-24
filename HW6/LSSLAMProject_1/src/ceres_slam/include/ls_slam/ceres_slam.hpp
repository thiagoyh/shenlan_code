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

        Eigen::Vector3d error_ij{residuals[0], residuals[1], residuals[2]};
        //Eigen::Map<Eigen::Vector3d> error_ij{residuals};
        Eigen::Matrix3d Ai, Bi;

        Eigen::Matrix3d Xi = PoseToTrans(xi);
        Eigen::Matrix3d Xj = PoseToTrans(xj);
        Eigen::Matrix3d Z = PoseToTrans(measurement);

        Eigen::Matrix3d Ei = Z.inverse() * Xi.inverse() * Xj;
        Eigen::Vector3d ei = TransToPose(Ei);
        error_ij = ei;

        Ai.setZero();
        Bi.setZero();

        Eigen::Matrix2d Ri = Xi.block(0, 0, 2, 2);
        Eigen::Matrix2d Rj = Xj.block(0, 0, 2, 2);
        Eigen::Matrix2d Rij = Z.block(0, 0, 2, 2);

        Eigen::Vector2d ti;
        ti << xi(0), xi(1);
        Eigen::Vector2d tj;
        tj << xj(0), xj(1);
        Eigen::Vector2d tij;
        tij << measurement(0), measurement(1);

        Eigen::Matrix2d translation;
        translation << -sin(xi(2)), cos(xi(2)), -cos(xi(2)), -sin(xi(2));
        Eigen::Vector2d tmpt = Rij.transpose() * translation * (tj - ti);

        Ai.block(0, 0, 2, 2) = -Rij.transpose() * Ri.transpose();
        Ai(2, 2) = -1;
        Ai(0, 2) = tmpt(0);
        Ai(1, 2) = tmpt(1);

        Bi.block(0, 0, 2, 2) = Rij.transpose() * Ri.transpose();
        Bi(2, 2) = 1;

        error_ij = sqrt_info_matrix * error_ij;

        residuals[0] = error_ij(0);
        residuals[1] = error_ij(1);
        residuals[2] = error_ij(2);

        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Matrix3d jacobian_xi = sqrt_info_matrix * Ai;
                for (int i = 0; i != 3; ++i)
                {
                    for (int j = 0; j != 3; ++j)
                    {
                        jacobians[0][i * 3 + j] = jacobian_xi(i, j);
                    }
                }
            }

            if (jacobians[1])
            {
                Eigen::Matrix3d jacobian_xj = sqrt_info_matrix * Bi;
                for (int i = 0; i != 3; ++i)
                {
                    for (int j = 0; j != 3; ++j)
                    {
                        jacobians[1][i * 3 + j] = jacobian_xj(i, j);
                    }
                }
            }
        }
        return true;
    }

private:
    Eigen::Vector3d measurement;
    Eigen::Matrix3d sqrt_info_matrix;
};
