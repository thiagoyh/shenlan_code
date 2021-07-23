cmake_minimum_required(VERSION 2.8.3)
project(ls_slam)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

find_package(Eigen3 REQUIRED)
find_package(G2O REQUIRED)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ls_slam
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)


include_directories(
  include
  include/ls_slam
  ${catkin_INCLUDE_DIRS}  ${EIGEN_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIR}
)



add_executable(ls_slam src/main.cpp src/readfile.cpp)
target_link_libraries(ls_slam ${catkin_LIBRARIES} ${CSPARSE_LIBRARY} )

