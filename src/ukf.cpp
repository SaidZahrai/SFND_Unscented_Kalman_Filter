#include "ukf.h"
#include "Eigen/Dense"

#include "coursefunctions.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */
  std_laspx_ = 0.15;  // Laser measurement noise standard deviation position1 in m
  std_laspy_ = 0.15;  // Laser measurement noise standard deviation position2 in m
  std_radr_ = 0.3;    // Radar measurement noise standard deviation radius in m
  std_radphi_ = 0.03; // Radar measurement noise standard deviation angle in rad
  std_radrd_ = 0.3;   // Radar measurement noise standard deviation radius change in m/s
  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  is_initialized_ = false; // initially set to false, set to true in first call of ProcessMeasurement

  time_us_ = 0; // time when the state is true, in us

  std_a_ = 0.2;     // Process noise standard deviation longitudinal acceleration in m/s^2 - Value from the course exercise
  std_yawdd_ = 0.2; // Process noise standard deviation yaw acceleration in rad/s^2 - Value from the course exercise

  n_x_ = 5;             // State dimension
  n_aug_ = 7;           // Augmented state dimension
  lambda_ = 3 - n_aug_; // Sigma point spreading parameter

  x_ = VectorXd(n_x_);       // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  P_ = MatrixXd(n_x_, n_x_); // state covariance matrix

  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1); // augmented sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);  // predicted sigma points matrix
  weights_ = VectorXd(2 * n_aug_ + 1);          // Weights of sigma points
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_)
  {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      x_.fill(0.0);
      x_(0) = meas_package.raw_measurements_[0];
      x_(1) = meas_package.raw_measurements_[1];

      P_.fill(0.0);
      P_(0, 0) = std_laspx_ * std_laspx_;
      P_(1, 1) = std_laspy_ * std_laspy_;
      P_(2, 2) = 0.1;
      P_(3, 3) = 0.1;
      P_(4, 4) = 0.1;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      x_.fill(0.0);

      x_(0) = meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]);
      x_(1) = meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]);
      x_(2) = meas_package.raw_measurements_[2];
      x_(3) = meas_package.raw_measurements_[3];

      P_.fill(0.0);
      P_(0, 0) = std_laspx_ * std_laspx_;
      P_(1, 1) = std_laspy_ * std_laspy_;
      P_(2, 2) = 0.1;
      P_(3, 3) = 0.1;
      P_(4, 4) = 0.1;
    }
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }
  // Predict
  double delta_t = (meas_package.timestamp_ - time_us_) / 1e6;
  time_us_ = meas_package.timestamp_;
  Prediction(delta_t);
  // Update
  if (meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_package);
  }
}

void UKF::Prediction(double delta_t)
{
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */
  augmentedSigmaPoints(n_x_, n_aug_, lambda_,
                       std_a_, std_yawdd_,
                       x_, P_,
                       Xsig_aug_);

  sigmaPointPrediction(n_x_, n_aug_,
                       delta_t, Xsig_aug_, Xsig_pred_);

  predictMeanAndCovariance(n_x_, n_aug_, lambda_,
                           Xsig_pred_, x_, P_);
}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1); // sigma points transfered to the measurement space
  VectorXd z_pred = VectorXd(n_z);               // mean predicted measurement
  MatrixXd S_pred = MatrixXd(n_z, n_z);          // predicted measurement covariance

  predictRadarMeasurement(n_x_, n_aug_, n_z, lambda_,
                          std_radr_, std_radphi_, std_radrd_,
                          Xsig_pred_,
                          Zsig, z_pred, S_pred);

  updateState(n_x_, n_aug_, n_z, lambda_,
              Zsig, z_pred, S_pred, meas_package.raw_measurements_,
              Xsig_pred_, x_, P_);
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  int n_z = 2;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1); // sigma points transfered to the measurement space
  VectorXd z_pred = VectorXd(n_z);               // mean predicted measurement
  MatrixXd S_pred = MatrixXd(n_z, n_z);          // predicted measurement covariance

  predictLidarMeasurement(n_x_, n_aug_, n_z, lambda_,
                          std_laspx_, std_laspy_,
                          Xsig_pred_,
                          Zsig, z_pred, S_pred);

  updateState(n_x_, n_aug_, n_z, lambda_,
              Zsig, z_pred, S_pred, meas_package.raw_measurements_,
              Xsig_pred_, x_, P_);
}