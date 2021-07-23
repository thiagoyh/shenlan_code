#include <gaussian_newton.h>
#include <readfile.h>

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <g2o_slam.h>

//for visual
void PublishGraphForVisulization(ros::Publisher *pub,
                                 std::vector<Eigen::Vector3d> &Vertexs,
                                 std::vector<Edge> &Edges,
                                 int color = 0)
{
    visualization_msgs::MarkerArray marray;

    //point--red
    visualization_msgs::Marker m;
    m.header.frame_id = "map";
    m.header.stamp = ros::Time::now();
    m.id = 0;
    m.ns = "ls-slam";
    m.type = visualization_msgs::Marker::SPHERE;
    m.pose.position.x = 0.0;
    m.pose.position.y = 0.0;
    m.pose.position.z = 0.0;
    m.scale.x = 0.1;
    m.scale.y = 0.1;
    m.scale.z = 0.1;

    if (color == 0)
    {
        m.color.r = 1.0;
        m.color.g = 0.0;
        m.color.b = 0.0;
    }
    else
    {
        m.color.r = 0.0;
        m.color.g = 1.0;
        m.color.b = 0.0;
    }

    m.color.a = 1.0;
    m.lifetime = ros::Duration(0);

    //linear--blue
    visualization_msgs::Marker edge;
    edge.header.frame_id = "map";
    edge.header.stamp = ros::Time::now();
    edge.action = visualization_msgs::Marker::ADD;
    edge.ns = "karto";
    edge.id = 0;
    edge.type = visualization_msgs::Marker::LINE_STRIP;
    edge.scale.x = 0.1;
    edge.scale.y = 0.1;
    edge.scale.z = 0.1;

    if (color == 0)
    {
        edge.color.r = 0.0;
        edge.color.g = 0.0;
        edge.color.b = 1.0;
    }
    else
    {
        edge.color.r = 1.0;
        edge.color.g = 0.0;
        edge.color.b = 1.0;
    }
    edge.color.a = 1.0;

    m.action = visualization_msgs::Marker::ADD;
    uint id = 0;

    //加入节点
    for (uint i = 0; i < Vertexs.size(); i++)
    {
        m.id = id;
        m.pose.position.x = Vertexs[i](0);
        m.pose.position.y = Vertexs[i](1);
        marray.markers.push_back(visualization_msgs::Marker(m));
        id++;
    }

    //加入边
    for (int i = 0; i < Edges.size(); i++)
    {
        Edge tmpEdge = Edges[i];
        edge.points.clear();

        geometry_msgs::Point p;
        p.x = Vertexs[tmpEdge.xi](0);
        p.y = Vertexs[tmpEdge.xi](1);
        edge.points.push_back(p);

        p.x = Vertexs[tmpEdge.xj](0);
        p.y = Vertexs[tmpEdge.xj](1);
        edge.points.push_back(p);
        edge.id = id;

        marray.markers.push_back(visualization_msgs::Marker(edge));
        id++;
    }

    pub->publish(marray);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ls_slam");

    ros::NodeHandle nodeHandle;

    // beforeGraph
    ros::Publisher beforeGraphPub, afterGraphPub;
    beforeGraphPub = nodeHandle.advertise<visualization_msgs::MarkerArray>("beforePoseGraph", 1, true);
    afterGraphPub = nodeHandle.advertise<visualization_msgs::MarkerArray>("afterPoseGraph", 1, true);

    std::string VertexPath = "/home/eventec/LSSLAMProject/src/ls_slam/data/test_quadrat-v.dat";
    std::string EdgePath = "/home/eventec/LSSLAMProject/src/ls_slam/data/test_quadrat-e.dat";

    //    std::string VertexPath = "/home/eventec/LSSLAMProject/src/ls_slam/data/intel-v.dat";
    //    std::string EdgePath = "/home/eventec/LSSLAMProject/src/ls_slam/data/intel-e.dat";

    std::vector<Eigen::Vector3d> Vertexs;
    std::vector<Edge> Edges;

    ReadVertexInformation(VertexPath, Vertexs);
    ReadEdgesInformation(EdgePath, Edges);

    PublishGraphForVisulization(&beforeGraphPub,
                                Vertexs,
                                Edges);

    double initError = ComputeError(Vertexs, Edges);
    std::cout << "initError:" << initError << std::endl;

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 3>> Block;

    Block::LinearSolverType *linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block *solver_ptr = new Block(std::unique_ptr<Block::LinearSolverType>(linearSolver));

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(solver_ptr));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    //向图中添加顶点
    for (int i = 0; i != Vertexs.size(); ++i)
    {
        Slam_Vertex *v = new Slam_Vertex();
        v->setEstimate(Vertexs[i]);
        v->setId(i);
        if (!i)
            v->setFixed(true);
        optimizer.addVertex(v);
    }

    //向图中增加边
    for (int i = 0; i != Edges.size(); ++i)
    {
        Slam_Edge *edge = new Slam_Edge();
        Edge tmpt = Edges[i];
        edge->setId(i);
        edge->setVertex(0, optimizer.vertices()[tmpt.xi]);
        edge->setVertex(1, optimizer.vertices()[tmpt.xj]);
        edge->setMeasurement(tmpt.measurement);
        edge->setInformation(tmpt.infoMatrix);
        optimizer.addEdge(edge);
    }

    std::cout << "start optimization!" << std::endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "this optimization process takes " << time_used.count() << " seconds!!" << std::endl;

    for (int i = 0; i != Vertexs.size(); ++i)
    {
        Slam_Vertex *v_estimate = static_cast<Slam_Vertex *>(optimizer.vertices[i]);
        Vertexs[i] = v_estimate->estimate();
    }

    PublishGraphForVisulization(&afterGraphPub,
                                Vertexs,
                                Edges, 1);

    ros::spin();

    return 0;
}
