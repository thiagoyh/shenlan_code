cmake_minimum_required(VERSION 2.8.3)
project(occupany_mapping)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
)



find_package(Eigen3 REQUIRED)



catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES occupany_mapping
  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)


include_directories(
  include
  include/occupany_mapping/
  ${catkin_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
)


add_executable(occupany_mapping

src/occupany_mapping_3.cpp
src/readfile.cpp)


 target_link_libraries(occupany_mapping
   ${catkin_LIBRARIES}
   ${EIGEN_LIBRARIES}
 )

