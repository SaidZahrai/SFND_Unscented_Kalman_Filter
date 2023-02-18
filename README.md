# Udacity Sensor Fusion Nanodegree - Project 4: Unscented Kalman Filter

This is the final project in the above mentioned nano degree concerning implementation of Unscented Kalman Filter for fusing Radar and Lidar measurement data. The repository is a fork from [Udacity's repository for the project](https://github.com/udacity/SFND_Unscented_Kalman_Filter) completed according to the instructions, fulfilling the requirements.

<img src="media/ukf_highway_tracked.gif" width="700" height="400" />

## Project rubric
### The submission must compile.
The code presented in this repository compiles and runs well on the environment provided by Udacity, using `cmake` and `make`. I tried to set up the environment on different versions of Ubuntu, both as host OS as well as Docker images, but unfortunately, I could not succeed to build PCL 1.2. If you succeed to have the dependencies, especially PCL 1.2, you should also be able to build this project with the standard steps of

1. cloning this repository and changing the directory to that
2. mkdir build && cd build
3. cmake ..
4. make
5. ./ukf_highway

The complete list of dependencies is
* cmake >= 3.5
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
* gcc/g++ >= 5.4
* PCL 1.2

See [Udacity's repository for the project](https://github.com/udacity/SFND_Unscented_Kalman_Filter).

### The methods in the code should avoid unnecessary calculations.
There is no unnecessary calculations in the code. In addition, to avoid repeated memory allocation and initialization in each step, vectors and matrices that are needed in the algorithm are allocated and initialized in the base class `UKF`, in the header file, `ukf.h` and with the implemetation in `ukf.cpp`.

`ukf.h`:
```C++
  // Variables used in the algorithm. They can be generated once and used at each step
  // Radar
  // Size of the measurement vector
  int n_z_radar_; 

  // sigma points transfered to the measurement space
  MatrixXd Zsig_radar_; 

  // mean predicted measurement
  VectorXd z_pred_radar_;      

  // predicted measurement covariance
  MatrixXd S_pred_radar_;          
  // Lidar
  // Size of the measurement vector
  int n_z_lidar_;

  // sigma points transfered to the measurement space
  MatrixXd Zsig_lidar_; 

  // mean predicted measurement
  VectorXd z_pred_lidar_;               

  // predicted measurement covariance
  MatrixXd S_pred_lidar_;  
  ```

`ukf.cpp`:
```C++
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
  ```

### px, py, vx, vy output coordinates must have an RMSE <= [0.30, 0.16, 0.95, 0.70] after running for longer than 1 second.
This is checked by the program and in case of violation of the requirement, the values are shown on the screen in red text.

### Your Sensor Fusion algorithm follows the general processing flow as taught in the preceding lessons.
The code is implementing the algorithm exactly the way it was presented in the lessons in 6 methods implemented in a header file `coursefunctions.h` as below. 

```C++
// Rewrite of the lesson exercise
// Starting from the current meand and covariance matrix and process model acceleration noises,
// it creates augmented sigma points.
void augmentedSigmaPoints(int n_x, int n_aug, double lambda,
                          double std_a, double std_yawdd,
                          VectorXd &x, MatrixXd &P,
                          MatrixXd &Xsig_aug)
{
  /...
}

// Rewrite of the lesson exercise
// Starting from the augmented sigma points, it uses the process model to calculate the
// corresponding state space prediction for each of the sigma points
void sigmaPointPrediction(int n_x, int n_aug, double delta_t,
                          MatrixXd &Xsig_aug,
                          MatrixXd &Xsig_pred)
{
  /...
}

// Rewrite of the lesson exercise
// Starting from a set of sigma points, it uses weights to calculate the mean state and covariance matrix
void predictMeanAndCovariance(int n_x, int n_aug, VectorXd weights,
                              MatrixXd &Xsig_pred,
                              VectorXd &x, MatrixXd &P)
{
  /...
}

// Rewrite of the lesson exercise
// Starting from the set of predicted sigma points in state space, it first uses the measurement model
// to calculate how the states would be seen by the radar, and then, it uses weights to calculate the
// mean state and covariance matrix of the transforemed sigma point in the radar measurement space.
void predictRadarMeasurement(int n_x, int n_aug, int n_z, VectorXd weights,
                             double std_radr, double std_radphi, double std_radrd,
                             MatrixXd &Xsig_pred,
                             MatrixXd &Zsig, VectorXd &z_pred, MatrixXd &S)
{
  /...
}

// Rewrite of the lesson exercise and adapted to Lidar
// Starting from the set of predicted sigma pointsin state space, it first uses the measurement model
// to calculate how the states would be seen by the radar, and then, it uses weights to calculate the
// mean state and covariance matrix of the transforemed sigma point in the radar measurement space.
void predictLidarMeasurement(int n_x, int n_aug, int n_z, VectorXd weights,
                             double std_lidx, double std_lidy,
                             MatrixXd &Xsig_pred,
                             MatrixXd &Zsig, VectorXd &z_pred, MatrixXd &S)
{
  /...
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

void updateState(int n_x, int n_aug, int n_z, VectorXd weights,
                 MatrixXd &Zsig, VectorXd &z_pred, MatrixXd &S, VectorXd &z,
                 MatrixXd &Xsig_pred, VectorXd &x_pred, MatrixXd &P_pred)
{
  /...
}
```

These functions are called by `UKF::Prediction`, `UKF::UpdateRadar` and `UKF::UpdateLidar`, in `ukf.cpp` as shown below.

```C++
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
  predictRadarMeasurement(n_x_, n_aug_, n_z_lidar_, weights_,
                          std_radr_, std_radphi_, std_radrd_,
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
```