cmake_minimum_required(VERSION 2.8.3)
project(ceres_slam)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

catkin_package(

)


include_directories(
  include
  include/ls_slam
  ${catkin_INCLUDE_DIRS}  ${EIGEN_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS}
  "/usr/include/eigen3"
)


add_executable(ls_slam_ceres src/main.cpp src/readfile.cpp src/gaussian_newton.cpp)
target_link_libraries(ls_slam_ceres ${catkin_LIBRARIES} ${CERES_LIBRARIES})






