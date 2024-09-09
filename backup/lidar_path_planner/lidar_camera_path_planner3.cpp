#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
#include <thread>
#include <fstream>
#include <queue>
#include <unordered_map>
#include "sl_lidar.h"
#include "sl_lidar_driver.h"

#ifndef _countof
#define _countof(_Array) (int)(sizeof(_Array) / sizeof(_Array[0]))
#endif

#include <unistd.h>
static inline void delay(sl_word_size_t ms) {
    while (ms >= 1000) {
        usleep(1000 * 1000);
        ms -= 1000;
    };
    if (ms != 0)
        usleep(ms * 1000);
}

void clearScreen() {
    int ret = system("clear");
    if (ret != 0) {
        fprintf(stderr, "Warning: Failed to clear the screen.\n");
    }
}

using namespace sl;
using namespace cv;

// Global variables for control
bool ctrl_c_pressed;
void ctrlc(int) {
    ctrl_c_pressed = true;
}

// Structure for detection results
struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

// Custom Node struct to avoid naming conflict with OpenCV
struct PathNode {
    Point position;
    float g_cost;
    float f_cost;
    PathNode* parent;

    PathNode(const Point& pos, float g, float f, PathNode* par)
        : position(pos), g_cost(g), f_cost(f), parent(par) {}
};

// Comparator for A* algorithm's priority queue
struct ComparePathNode {
    bool operator()(PathNode* a, PathNode* b) {
        return a->f_cost > b->f_cost; // Higher f_cost means lower priority
    }
};

// Function to check SLAMTEC LIDAR health
bool checkSLAMTECLIDARHealth(ILidarDriver* drv) {
    sl_result op_result;
    sl_lidar_response_device_health_t healthinfo;

    op_result = drv->getHealth(healthinfo);
    if (SL_IS_OK(op_result)) {
        printf("SLAMTEC Lidar health status : %d\n", healthinfo.status);
        if (healthinfo.status == SL_LIDAR_STATUS_ERROR) {
            fprintf(stderr, "Error, slamtec lidar internal error detected. Please reboot the device to retry.\n");
            return false;
        } else {
            return true;
        }
    } else {
        fprintf(stderr, "Error, cannot retrieve the lidar health code: %x\n", op_result);
        return false;
    }
}

// YOLOv5 and visualization parameters
const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;
const float AREA_THRESHOLD = 15000.0;

// Function to format YOLOv5 input
cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

// Function to load class list
std::vector<std::string> load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("/home/vegaai/sensorfusion/backup/object_detection/classes.txt");
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

// Function to load YOLO model
void load_net(cv::dnn::Net& net) {
    auto result = cv::dnn::readNet("/home/vegaai/sensorfusion/backup/object_detection/yolov5s.onnx");
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    } else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

// Function to detect objects in frames
void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& className) {
    cv::Mat blob;
    auto input_image = format_yolov5(image);
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float* data = (float*)outputs[0].data;
    const int dimensions = 85;
    const int rows = 25200;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (size_t i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

// Function to handle camera operations in a thread
void cameraThread() {
    std::vector<std::string> class_list = load_class_list();
    cv::Mat frame;
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        std::cerr << "Error opening camera\n";
        return;
    }
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cv::dnn::Net net;
    load_net(net);

    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;

    while (!ctrl_c_pressed) {
        capture.read(frame);
        if (frame.empty()) {
            std::cout << "No frame captured from camera\n";
            break;
        }

        std::vector<Detection> output;
        detect(frame, net, output, class_list);

        for (const auto& detection : output) {
            if (detection.box.area() > AREA_THRESHOLD) {
                const auto color = colors[detection.class_id % colors.size()];
                cv::rectangle(frame, detection.box, color, 3);
                cv::rectangle(frame, cv::Point(detection.box.x, detection.box.y - 20), cv::Point(detection.box.x + detection.box.width, detection.box.y), color, cv::FILLED);
                cv::putText(frame, class_list[detection.class_id], cv::Point(detection.box.x, detection.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

        if (++frame_count >= 30) {
            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        if (fps > 0) {
            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2) << "FPS: " << fps;
            cv::putText(frame, fps_label.str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Camera Detection", frame);

        if (cv::waitKey(1) != -1) {
            capture.release();
            std::cout << "Detection Terminated\n";
            break;
        }
    }
}

// Functions for LiDAR processing
std::vector<Point> calculate_corners(const Point& center, int side_length) {
    int half_side = side_length / 2;
    return {
        Point(center.x - half_side, center.y), // Top-left corner
        Point(center.x + half_side, center.y), // Top-right corner
        Point(center.x + half_side, center.y + side_length), // Bottom-right corner
        Point(center.x - half_side, center.y + side_length)  // Bottom-left corner
    };
}

bool isPointInRedZone(const Point& point, const std::vector<std::vector<Point>>& bounding_boxes) {
    for (const auto& box_corners : bounding_boxes) {
        double sum = 0.0;
        for (int i = 0; i < 4; ++i) {
            Point p1 = box_corners[i];
            Point p2 = box_corners[(i + 1) % 4];
            sum += (point.x - p1.x) * (p2.y - p1.y) - (point.y - p1.y) * (p2.x - p1.x);
        }
        if (fabs(sum) < 1e-6) {
            return true;
        }
    }
    return false;
}

bool line_intersects_box(const Point& line_start, const Point& line_end, const std::vector<Point>& box_corners) {
    for (int i = 0; i < 4; ++i) {
        Point box_start = box_corners[i];
        Point box_end = box_corners[(i + 1) % 4];
        
        double det1 = (line_end.x - line_start.x) * (box_start.y - line_start.y) - (line_end.y - line_start.y) * (box_start.x - line_start.x);
        double det2 = (line_end.x - line_start.x) * (box_end.y - line_start.y) - (line_end.y - line_start.y) * (box_end.x - line_start.x);
        double det3 = (box_end.x - box_start.x) * (line_start.y - box_start.y) - (box_end.y - box_start.y) * (line_start.x - box_start.x);
        double det4 = (box_end.x - box_start.x) * (line_end.y - box_start.y) - (box_end.y - box_start.y) * (line_end.x - box_start.x);
        
        if ((det1 * det2 < 0) && (det3 * det4 < 0)) {
            return true;
        }
    }
    return false;
}

std::vector<Point> a_star(const Point& start, const Point& goal, const std::vector<Point>& corners, const std::vector<std::vector<Point>>& bounding_boxes) {
    std::priority_queue<PathNode*, std::vector<PathNode*>, ComparePathNode> open_list;
    std::unordered_map<int, PathNode*> all_nodes;

    PathNode* start_node = new PathNode(start, 0.0, norm(goal - start), nullptr);
    open_list.push(start_node);
    all_nodes[start.x + start.y * 10000] = start_node;

    while (!open_list.empty()) {
        PathNode* current = open_list.top();
        open_list.pop();

        if (current->position == goal) {
            std::vector<Point> path;
            while (current) {
                path.push_back(current->position);
                current = current->parent;
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        for (const auto& neighbor : corners) {
            if (isPointInRedZone(neighbor, bounding_boxes)) continue;

            bool intersects = false;
            for (const auto& box : bounding_boxes) {
                if (line_intersects_box(current->position, neighbor, box)) {
                    intersects = true;
                    break;
                }
            }
            if (intersects) continue;

            float tentative_g_cost = current->g_cost + norm(neighbor - current->position);
            float h_cost = norm(goal - neighbor);
            float f_cost = tentative_g_cost + h_cost;

            int neighbor_key = neighbor.x + neighbor.y * 10000;
            if (all_nodes.find(neighbor_key) == all_nodes.end() || tentative_g_cost < all_nodes[neighbor_key]->g_cost) {
                PathNode* neighbor_node = new PathNode(neighbor, tentative_g_cost, f_cost, current);
                open_list.push(neighbor_node);
                all_nodes[neighbor_key] = neighbor_node;
            }
        }
    }

    return {};
}

void drawPathPlanning(Mat& display, const Point& lidarPosition, const Point& destination, const std::vector<std::vector<Point>>& bounding_boxes) {
    // Check if destination is within any bounding box
    if (isPointInRedZone(destination, bounding_boxes)) {
        std::cout << "Destination is in the red zone. No connecting lines will be drawn." << std::endl;
        return;
    }

    // Check if the direct path from LIDAR position to destination is clear
    bool direct_path_clear = true;
    for (const auto& box : bounding_boxes) {
        if (line_intersects_box(lidarPosition, destination, box)) {
            direct_path_clear = false;  // An obstacle is blocking the direct path
            break;
        }
    }

    // If the direct path is clear, draw it directly
    if (direct_path_clear) {
        line(display, lidarPosition, destination, Scalar(255, 255, 255), 4);
        return;
    }

    // Otherwise, use A* to find the shortest path around the obstacles
    std::vector<Point> graph_points;
    graph_points.push_back(lidarPosition);  // Start point
    graph_points.push_back(destination);  // End point

    // Add all the corners of the bounding boxes as potential points for pathfinding
    for (const auto& box : bounding_boxes) {
        for (const auto& corner : box) {
            bool valid = true;
            // Ensure that this corner is not inside any other bounding box
            for (const auto& other_box : bounding_boxes) {
                if (isPointInRedZone(corner, {other_box})) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                graph_points.push_back(corner);  // Add valid corner to graph points
                line(display, lidarPosition, corner, Scalar(128, 0, 128), 2);  // Visualize connections
            }
        }
    }

    // Compute the shortest path using the A* algorithm
    std::vector<Point> shortest_path = a_star(lidarPosition, destination, graph_points, bounding_boxes);

    // If a valid shortest path is found, draw it
    if (!shortest_path.empty()) {
        for (size_t i = 1; i < shortest_path.size(); ++i) {
            line(display, shortest_path[i - 1], shortest_path[i], Scalar(0, 255, 255), 4);
        }
    } else {
        std::cout << "No valid path found." << std::endl;
    }
}

// Draw LiDAR Points and Bounding Boxes
void drawLidarPointsAndBoundingBoxes(Mat& display, const Point& lidarPosition, const Point& destination, const std::vector<std::pair<float, float>>& points, float fixed_length_extension, float fixed_width_extension) {
    display = Mat(display.size(), CV_8UC3, Scalar(0, 0, 255));

    std::vector<Point> obstaclePoints; 

    for (const auto& point : points) {
        float angle = point.first;
        float distance = point.second;

        int x = static_cast<int>(lidarPosition.x + distance * sin(angle * CV_PI / 180.0) / 10); 
        int y = static_cast<int>(lidarPosition.y - distance * cos(angle * CV_PI / 180.0) / 10); 

        line(display, lidarPosition, Point(x, y), Scalar(0, 255, 0), 2); 

        if (distance <= 2000) {
            obstaclePoints.push_back(Point(x, y)); 
        }
    }

    // Draw two-meter radius line
    int two_meter_radius_px = static_cast<int>(2000 / 10); // Assuming 1 pixel = 10 mm
    circle(display, lidarPosition, two_meter_radius_px, Scalar(0, 255, 255), 1);

    std::vector<std::vector<Point>> clusters;
    const float cluster_gap_threshold_px = 50.0; 

    for (const auto& p : obstaclePoints) {
        bool added_to_cluster = false;

        for (auto& cluster : clusters) {
            for (const auto& cluster_point : cluster) {
                float distance_to_point = sqrt(pow(p.x - cluster_point.x, 2) + pow(p.y - cluster_point.y, 2));
                if (distance_to_point < cluster_gap_threshold_px) {
                    cluster.push_back(p);
                    added_to_cluster = true;
                    break;
                }
            }
            if (added_to_cluster) break;
        }

        if (!added_to_cluster) {
            clusters.push_back({ p });
        }
    }

    std::vector<std::vector<Point>> bounding_boxes;
    for (const auto& cluster : clusters) {
        if (cluster.size() < 2) continue;

        Point2f center(0, 0);
        for (const auto& p : cluster) {
            center.x += p.x;
            center.y += p.y;
        }
        center.x /= cluster.size();
        center.y /= cluster.size();

        float max_dist_x = 0.0, max_dist_y = 0.0;
        for (const auto& p : cluster) {
            max_dist_x = std::max(max_dist_x, fabs(p.x - center.x));
            max_dist_y = std::max(max_dist_y, fabs(p.y - center.y));
        }

        // Align the center to the top line of the bounding box
        Rect boundingBox(center.x - max_dist_x - fixed_width_extension / 2,
                         center.y - max_dist_y - fixed_length_extension,
                         2 * max_dist_x + fixed_width_extension,
                         2 * max_dist_y + fixed_length_extension);

        rectangle(display, boundingBox, Scalar(255, 0, 0), 2);

        // Calculate and display distance from LIDAR to bounding box center
        float distance_to_cluster = sqrt(pow(center.x - lidarPosition.x, 2) + pow(center.y - lidarPosition.y, 2)) * 10; // Convert pixels to mm
        std::ostringstream distance_text;
        distance_text << std::fixed << std::setprecision(2) << distance_to_cluster / 1000.0 << " m"; // Convert to meters
        putText(display, distance_text.str(), center, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

        bounding_boxes.push_back(calculate_corners(Point(boundingBox.x + boundingBox.width / 2, boundingBox.y), boundingBox.width));
    }

    drawPathPlanning(display, lidarPosition, destination, bounding_boxes);
    imshow("Lidar Data Visualization", display);
    waitKey(1);
}

int main(int argc, const char* argv[]) {
    const char* opt_is_channel = NULL;
    const char* opt_channel = NULL;
    const char* opt_channel_param_first = NULL;
    sl_u32 opt_channel_param_second = 0;
    sl_u32 baudrateArray[2] = {115200, 256000};
    sl_result op_result;
    int opt_channel_type = CHANNEL_TYPE_SERIALPORT;

    bool useArgcBaudrate = false;
    IChannel* _channel = nullptr;

    printf("Ultra simple LIDAR data grabber for SLAMTEC LIDAR.\n"
           "Version: %s\n", SL_LIDAR_SDK_VERSION);

    if (argc > 1) {
        opt_is_channel = argv[1];
    } else {
        printf("Usage instructions...\n");
        return -1;
    }

    if (strcmp(opt_is_channel, "--channel") == 0) {
        opt_channel = argv[2];
        if (strcmp(opt_channel, "-s") == 0 || strcmp(opt_channel, "--serial") == 0) {
            opt_channel_param_first = argv[3];
            if (argc > 4) opt_channel_param_second = strtoul(argv[4], NULL, 10);
            useArgcBaudrate = true;
        } else if (strcmp(opt_channel, "-u") == 0 || strcmp(opt_channel, "--udp") == 0) {
            opt_channel_param_first = argv[3];
            if (argc > 4) opt_channel_param_second = strtoul(argv[4], NULL, 10);
            opt_channel_type = CHANNEL_TYPE_UDP;
        } else {
            printf("Usage instructions...\n");
            return -1;
        }
    } else {
        printf("Usage instructions...\n");
        return -1;
    }

    if (opt_channel_type == CHANNEL_TYPE_SERIALPORT) {
        if (!opt_channel_param_first) {
            opt_channel_param_first = "/dev/ttyUSB0";
        }
    }

    ILidarDriver* drv = *createLidarDriver();

    if (!drv) {
        fprintf(stderr, "Insufficient memory, exit\n");
        return -2;
    }

    sl_lidar_response_device_info_t devinfo;
    bool connectSuccess = false;

    do {
        if (opt_channel_type == CHANNEL_TYPE_SERIALPORT) {
            if (useArgcBaudrate) {
                _channel = (*createSerialPortChannel(opt_channel_param_first, opt_channel_param_second));
                if (SL_IS_OK((drv)->connect(_channel))) {
                    op_result = drv->getDeviceInfo(devinfo);
                    if (SL_IS_OK(op_result)) {
                        connectSuccess = true;
                    } else {
                        delete drv;
                        drv = NULL;
                    }
                }
            } else {
                size_t baudRateArraySize = (sizeof(baudrateArray)) / (sizeof(baudrateArray[0]));
                for (size_t i = 0; i < baudRateArraySize; ++i) {
                    _channel = (*createSerialPortChannel(opt_channel_param_first, baudrateArray[i]));
                    if (SL_IS_OK((drv)->connect(_channel))) {
                        op_result = drv->getDeviceInfo(devinfo);
                        if (SL_IS_OK(op_result)) {
                            connectSuccess = true;
                            break;
                        } else {
                            delete drv;
                            drv = NULL;
                        }
                    }
                }
            }
        } else if (opt_channel_type == CHANNEL_TYPE_UDP) {
            _channel = *createUdpChannel(opt_channel_param_first, opt_channel_param_second);
            if (SL_IS_OK((drv)->connect(_channel))) {
                op_result = drv->getDeviceInfo(devinfo);
                if (SL_IS_OK(op_result)) {
                    connectSuccess = true;
                } else {
                    delete drv;
                    drv = NULL;
                }
            }
        }

        if (!connectSuccess) {
            fprintf(stderr, "Error, cannot bind to the specified channel %s.\n", opt_channel_param_first);
            break;
        }

        printf("SLAMTEC LIDAR S/N: ");
        for (int pos = 0; pos < 16; ++pos) {
            printf("%02X", devinfo.serialnum[pos]);
        }
        printf("\nFirmware Ver: %d.%02d\nHardware Rev: %d\n",
               devinfo.firmware_version >> 8,
               devinfo.firmware_version & 0xFF,
               (int)devinfo.hardware_version);

        if (!checkSLAMTECLIDARHealth(drv)) {
            break;
        }

        signal(SIGINT, ctrlc);

        if (opt_channel_type == CHANNEL_TYPE_SERIALPORT)
            drv->setMotorSpeed();

        drv->startScan(0, 1);

        std::thread lidar_thread([&]() {
            std::vector<std::pair<float, float>> lidarPoints;
            const int imgWidth = 800;
            const int imgHeight = 600;
            Point lidarPosition = Point(imgWidth / 2, imgHeight);
            Point destination = Point(imgWidth / 2, 0);
            Mat display(imgHeight, imgWidth, CV_8UC3, Scalar(0, 0, 255));
            float fixed_length_extension = 100.0;
            float fixed_width_extension = 50.0;

            while (!ctrl_c_pressed) {
                sl_lidar_response_measurement_node_hq_t nodes[8192];
                size_t count = _countof(nodes);
                op_result = drv->grabScanDataHq(nodes, count);
                clearScreen();
                printf("Set_Start.\n");

                if (SL_IS_OK(op_result)) {
                    drv->ascendScanData(nodes, count);
                    lidarPoints.clear();

                    for (int pos = 0; pos < (int)count; ++pos) {
                        float raw_angle_degrees = (nodes[pos].angle_z_q14 * 90.f) / 16384.f;
                        float distance = nodes[pos].dist_mm_q2 / 4.0f;
                        float quality = nodes[pos].quality;

                        if ((raw_angle_degrees >= 0.0f && raw_angle_degrees <= 60.0f) ||
                            (raw_angle_degrees >= 300.0f && raw_angle_degrees <= 360.0f)) {

                            if (quality > 0) {
                                lidarPoints.emplace_back(raw_angle_degrees, distance);
                                printf("%03.2f %08.2f\n", raw_angle_degrees, distance);
                            }
                        }
                    }

                    drawLidarPointsAndBoundingBoxes(display, lidarPosition, destination, lidarPoints, fixed_length_extension, fixed_width_extension);
                }

                if (ctrl_c_pressed) {
                    break;
                }
            }

            drv->stop();
            delay(200);
            if (opt_channel_type == CHANNEL_TYPE_SERIALPORT)
                drv->setMotorSpeed(0);
        });

        std::thread camera_thread(cameraThread);

        lidar_thread.join();
        camera_thread.join();

    } while (0);

    if (drv) {
        delete drv;
        drv = NULL;
    }
    return 0;
}