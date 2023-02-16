#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Rewrite of the lesson exercise
// Starting from the current meand and covariance matrix and process model acceleration noises,
// it creates augmented sigma points.
void augmentedSigmaPoints(int n_x, int n_aug, double lambda,
                          double std_a, double std_yawdd,
                          VectorXd &x, MatrixXd &P,
                          MatrixXd &Xsig_aug)
{

    // create augmented mean vector
    VectorXd x_aug = VectorXd(7);

    // create augmented state covariance
    MatrixXd P_aug = MatrixXd(7, 7);

    // create augmented mean state
    x_aug.head(5) = x;
    x_aug(5) = 0;
    x_aug(6) = 0;

    // create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5, 5) = P;
    P_aug(5, 5) = std_a * std_a;
    P_aug(6, 6) = std_yawdd * std_yawdd;

    // create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    // create augmented sigma points
    Xsig_aug.col(0) = x_aug;

    for (int i = 0; i < n_aug; ++i)
    {
        Xsig_aug.col(i + 1) = x_aug + sqrt(lambda + n_aug) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug) = x_aug - sqrt(lambda + n_aug) * L.col(i);
    }
}

// Rewrite of the lesson exercise
// Starting from the augmented sigma points, it uses the process model to calculate the
// corresponding state space prediction for each of the sigma points
void sigmaPointPrediction(int n_x, int n_aug, double delta_t,
                          MatrixXd &Xsig_aug,
                          MatrixXd &Xsig_pred)
{

    // predict sigma points
    for (int i = 0; i < 2 * n_aug + 1; ++i)
    {
        // extract values for better readability
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        // predicted state values
        double px_p, py_p;

        // avoid division by zero
        if (fabs(yawd) > 0.001)
        {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        }
        else
        {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        // add noise
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p = v_p + nu_a * delta_t;

        yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawdd * delta_t;

        // write predicted sigma point into right column
        Xsig_pred(0, i) = px_p;
        Xsig_pred(1, i) = py_p;
        Xsig_pred(2, i) = v_p;
        Xsig_pred(3, i) = yaw_p;
        Xsig_pred(4, i) = yawd_p;
    }
}

// Rewrite of the lesson exercise
// Starting from a set of sigma points, it uses weights to calculate the mean state and covariance matrix
void predictMeanAndCovariance(int n_x, int n_aug, double lambda,
                              MatrixXd &Xsig_pred,
                              VectorXd &x, MatrixXd &P)
{

    // create vector for weights
    VectorXd weights = VectorXd(2 * n_aug + 1);

    // set weights
    double weight_0 = lambda / (lambda + n_aug);
    weights(0) = weight_0;
    for (int i = 1; i < 2 * n_aug + 1; ++i)
    { // 2n+1 weights
        double weight = 0.5 / (n_aug + lambda);
        weights(i) = weight;
    }

    // predicted state mean
    x.fill(0.0);
    for (int i = 0; i < 2 * n_aug + 1; ++i)
    { // iterate over sigma points
        x = x + weights(i) * Xsig_pred.col(i);
    }

    // predicted state covariance matrix
    P.fill(0.0);
    for (int i = 0; i < 2 * n_aug + 1; ++i)
    { // iterate over sigma points
        // state difference
        VectorXd x_diff = Xsig_pred.col(i) - x;
        // angle normalization
        while (x_diff(3) > M_PI)
            x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI)
            x_diff(3) += 2. * M_PI;

        P = P + weights(i) * x_diff * x_diff.transpose();
    }
}

// Rewrite of the lesson exercise
// Starting from the set of predicted sigma pointsin state space, it first uses the measurement model
// to calculate how the states would be seen by the radar, and then, it uses weights to calculate the
// mean state and covariance matrix of the transforemed sigma point in the radar measurement space.
void predictRadarMeasurement(int n_x, int n_aug, int n_z, double lambda,
                             double std_radr, double std_radphi, double std_radrd,
                             MatrixXd &Xsig_pred,
                             MatrixXd &Zsig, VectorXd &z_pred, MatrixXd &S)
{

    // set vector for weights
    VectorXd weights = VectorXd(2 * n_aug + 1);
    double weight_0 = lambda / (lambda + n_aug);
    double weight = 0.5 / (lambda + n_aug);
    weights(0) = weight_0;

    for (int i = 1; i < 2 * n_aug + 1; ++i)
    {
        weights(i) = weight;
    }

    // transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug + 1; ++i)
    { // 2n+1 simga points
        // extract values for better readability
        double p_x = Xsig_pred(0, i);
        double p_y = Xsig_pred(1, i);
        double v = Xsig_pred(2, i);
        double yaw = Xsig_pred(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // measurement model
        Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                         // r
        Zsig(1, i) = atan2(p_y, p_x);                                     // phi
        Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); // r_dot
    }

    // mean predicted measurement
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug + 1; ++i)
    {
        z_pred = z_pred + weights(i) * Zsig.col(i);
    }

    // innovation covariance matrix S
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug + 1; ++i)
    { // 2n+1 simga points
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // angle normalization
        while (z_diff(1) > M_PI)
            z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI)
            z_diff(1) += 2. * M_PI;

        S = S + weights(i) * z_diff * z_diff.transpose();
    }

    // add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_radr * std_radr, 0, 0,
        0, std_radphi * std_radphi, 0,
        0, 0, std_radrd * std_radrd;
    S = S + R;
}

// Rewrite of the lesson exercise and adapted to Lidar
// Starting from the set of predicted sigma pointsin state space, it first uses the measurement model
// to calculate how the states would be seen by the radar, and then, it uses weights to calculate the
// mean state and covariance matrix of the transforemed sigma point in the radar measurement space.
void predictLidarMeasurement(int n_x, int n_aug, int n_z, double lambda,
                             double std_lidx, double std_lidy,
                             MatrixXd &Xsig_pred,
                             MatrixXd &Zsig, VectorXd &z_pred, MatrixXd &S)
{

    // set vector for weights
    VectorXd weights = VectorXd(2 * n_aug + 1);
    double weight_0 = lambda / (lambda + n_aug);
    double weight = 0.5 / (lambda + n_aug);
    weights(0) = weight_0;

    for (int i = 1; i < 2 * n_aug + 1; ++i)
    {
        weights(i) = weight;
    }

    // transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug + 1; ++i)
    { // 2n+1 simga points
        // extract values for better readability
        double p_x = Xsig_pred(0, i);
        double p_y = Xsig_pred(1, i);
        double v = Xsig_pred(2, i);
        double yaw = Xsig_pred(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // measurement model
        Zsig(0, i) = p_x; // x
        Zsig(1, i) = p_y; // y
    }

    // mean predicted measurement
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug + 1; ++i)
    {
        z_pred = z_pred + weights(i) * Zsig.col(i);
    }

    // innovation covariance matrix S
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug + 1; ++i)
    { // 2n+1 simga points
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        S = S + weights(i) * z_diff * z_diff.transpose();
    }

    // add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_lidx * std_lidx, 0,
        0, std_lidy * std_lidy;
    S = S + R;
}

// Rewrite of the lesson exercise
// The final update step is implemented here. It uses the latest mesurements to update the predicted
// mean state and the covariance matrix to the one for this time step.
// The procedure is to calculate a cross correlation matric between the predicted sigma points in the state space
// and the predicted sigma point in the mesurement space. This cross correlation matrix, which will have
// dimensions of (n_x, n_z) will be used to calculate the Kalman gain, K = Tc * S.inverse(). The calculate residual
// i.e. the difference between the measurement data and the predicted mean in the measurement space, is used to
// update the mean state by x_pred = x_pred + K * z_diff.
// Equally, the predicted covariance matrix is updated by P_pred = P_pred - K * S * K.transpose();.

void updateState(int n_x, int n_aug, int n_z, double lambda,
                 MatrixXd &Zsig, VectorXd &z_pred, MatrixXd &S, VectorXd &z,
                 MatrixXd &Xsig_pred, VectorXd &x_pred, MatrixXd &P_pred)
{

    // set vector for weights
    VectorXd weights = VectorXd(2 * n_aug + 1);
    double weight_0 = lambda / (lambda + n_aug);
    double weight = 0.5 / (lambda + n_aug);
    weights(0) = weight_0;

    for (int i = 1; i < 2 * n_aug + 1; ++i)
    {
        weights(i) = weight;
    }

    MatrixXd Tc = MatrixXd(n_x, n_z);

    // calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug + 1; ++i)
    { // 2n+1 simga points
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // angle normalization
        while (z_diff(1) > M_PI)
            z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI)
            z_diff(1) += 2. * M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred.col(i) - x_pred;

        // angle normalization
        while (x_diff(3) > M_PI)
            x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI)
            x_diff(3) += 2. * M_PI;

        Tc = Tc + weights(i) * x_diff * z_diff.transpose();
    }

    // Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    // residual
    VectorXd z_diff = z - z_pred;

    // angle normalization
    while (z_diff(1) > M_PI)
        z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
        z_diff(1) += 2. * M_PI;

    // update state mean and covariance matrix
    x_pred = x_pred + K * z_diff;
    P_pred = P_pred - K * S * K.transpose();
}
