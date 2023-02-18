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
   * Initialization of other member properties. 
   * The two below lines are deleted: 
   *   std_a_ = 30;
   *   std_yawdd_ = 30;
   * and both values are set to 1.0. 
   */

  is_initialized_ = false; // initially set to false, set to true in first call of ProcessMeasurement

  time_us_ = 0; // time when the state is true, in us

  std_a_ = 1.0;     // Process noise standard deviation longitudinal acceleration in m/s^2
  std_yawdd_ = 1.0; // Process noise standard deviation yaw acceleration in rad/s^2

  n_x_ = 5;             // State dimension
  n_aug_ = 7;           // Augmented state dimension
  lambda_ = 3 - n_aug_; // Sigma point spreading parameter

  x_ = VectorXd(n_x_);       // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  P_ = MatrixXd(n_x_, n_x_); // state covariance matrix

  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1); // augmented sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);  // predicted sigma points matrix
  weights_ = VectorXd(2 * n_aug_ + 1);          // Weights of sigma points
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; ++i)
  {
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }

  // Variables used in the algorithm. They can be generated once and used at each step
  // Radar
  n_z_radar_ = 3;                                     // Size of the measurement vector
  Zsig_radar_ = MatrixXd(n_z_radar_, 2 * n_aug_ + 1); // sigma points transfered to the measurement space
  z_pred_radar_ = VectorXd(n_z_radar_);               // mean predicted measurement
  S_pred_radar_ = MatrixXd(n_z_radar_, n_z_radar_);   // predicted measurement covariance
  // Lidar
  n_z_lidar_ = 2;                                     // Size of the measurement vector
  Zsig_lidar_ = MatrixXd(n_z_lidar_, 2 * n_aug_ + 1); // sigma points transfered to the measurement space
  z_pred_lidar_ = VectorXd(n_z_lidar_);               // mean predicted measurement
  S_pred_lidar_ = MatrixXd(n_z_lidar_, n_z_lidar_);   // predicted measurement covariance
}

UKF::~UKF() {}

// The main function to update the states after each measurement.
// The first time it is called, it sets the state variable to the measured values
// and the covariance matrix to the values that are available.
// In the subsequent steps, every time a measurement data arrives, the state variable
// is updated according to the unscented Kalman filter algorithm by calling other
// member functions.
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
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
  // Prediction independent from the type of incoming data
  double delta_t = (meas_package.timestamp_ - time_us_) / 1e6;
  time_us_ = meas_package.timestamp_;
  Prediction(delta_t);
  // Update by calling other member functions.
  if (meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_package);
  }
}

// Prediction based on unscented Kalman filter algorithm.
void UKF::Prediction(double delta_t)
{
  // Step 1) Use the current state vector (x_) and covariance matrix (P_) and define the augmented sigma points (Xsig_aug_)
  augmentedSigmaPoints(n_x_, n_aug_, lambda_,
                       std_a_, std_yawdd_,
                       x_, P_,
                       Xsig_aug_); // Output

  // Step 2) For each augmented sigma point (Xsig_aug_) calculate a predicted next point in the state space (Xsig_pred_)
  sigmaPointPrediction(n_x_, n_aug_,
                       delta_t, Xsig_aug_,
                       Xsig_pred_); // Output

  // Step 3) Use the predicted sigma points (Xsig_pred_) and precalculated weights (weights_)
  // to estimate the mean state (x_) and covariance matrix (P_) assumong that a normal noise distrbution
  predictMeanAndCovariance(n_x_, n_aug_,
                           weights_, Xsig_pred_,
                           x_, P_); // Output
}

// Update the state using a new radar measurement point
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  // Step 1) Transfer the predicted sigma points (Xsig_pred_) to the measurement space,
  // i.e. calculate how they will be viewd by the sensor (Zsig) and then, use them to calculate
  // the mean predicted measuremnt (z_pred) and the related covariance matrix (S_pred).
  predictRadarMeasurement(n_x_, n_aug_, n_z_radar_, weights_,
                          std_radr_, std_radphi_, std_radrd_,
                          Xsig_pred_,
                          Zsig_radar_, z_pred_radar_, S_pred_radar_); // Output

  // Step 2) Use the predicted values, sigma points both in the state space (Xsig_pred_) and in the measurement space (Zsig),
  // mean vector (z_pred) and the corresponding predicted covariance matrix (S_pred) and
  // update the state vector (x_) and the covariance matrix (P_)
  updateState(n_x_, n_aug_, n_z_radar_, weights_,
              Zsig_radar_, z_pred_radar_, S_pred_radar_, meas_package.raw_measurements_,
              Xsig_pred_,
              x_, P_); // Input/Output - Will be updated
}

// Update the state using a new lidar measurement point
void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  // Step 1) Transfer the predicted sigma points (Xsig_pred_) to the measurement space,
  // i.e. calculate how they will be viewd by the sensor (Zsig) and then, use them to calculate
  // the mean predicted measuremnt (z_pred) and the related covariance matrix (S_pred).
  predictLidarMeasurement(n_x_, n_aug_, n_z_lidar_, weights_,
                          std_laspx_, std_laspy_,
                          Xsig_pred_,
                          Zsig_lidar_, z_pred_lidar_, S_pred_lidar_); // Output

  // Step 2) Use the predicted values, sigma points both in the state space (Xsig_pred_) and in the measurement space (Zsig),
  // mean vector (z_pred) and the corresponding predicted covariance matrix (S_pred) and
  // update the state vector (x_) and the covariance matrix (P_)
  updateState(n_x_, n_aug_, n_z_lidar_, weights_,
              Zsig_lidar_, z_pred_lidar_, S_pred_lidar_, meas_package.raw_measurements_,
              Xsig_pred_,
              x_, P_); // Input/Output - Will be updated
}