#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "sl_lidar.h" 
#include "sl_lidar_driver.h"

#ifndef _countof
#define _countof(_Array) (int)(sizeof(_Array) / sizeof(_Array[0]))
#endif

#ifdef _WIN32
#include <Windows.h>
#define delay(x)   ::Sleep(x)
void clearScreen() {
    system("cls"); // Clear console for Windows
}
#else
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
    system("clear"); // Clear console for Linux/Unix
}
#endif

using namespace sl;
using namespace cv;

bool ctrl_c_pressed;
void ctrlc(int)
{
    ctrl_c_pressed = true;
}

bool checkSLAMTECLIDARHealth(ILidarDriver * drv)
{
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

void drawLidarPointsAndBoundingBoxes(Mat& display, const Point& lidarPosition, const std::vector<std::pair<float, float>>& points, float fixed_length_extension, float fixed_width_extension) {
    // Initialize the visualization area as filled
    display = Mat(display.size(), CV_8UC3, Scalar(0, 0, 255)); // Filled with red color indicating occupied space

    // Draw a 2-meter radius circle around the LiDAR position
    int radius_2m_px = static_cast<int>(2000 / 10.0); // Convert 2 meters to pixels (scale factor of 10)
    circle(display, lidarPosition, radius_2m_px, Scalar(255, 255, 255), 1); // Draw the circle in white color

    std::vector<Point> obstaclePoints; // To store obstacle points

    // Clear the area based on LiDAR data
    for (const auto& point : points) {
        float angle = point.first;
        float distance = point.second;

        // Convert polar coordinates to Cartesian for visualization
        int x = static_cast<int>(lidarPosition.x + distance * sin(angle * CV_PI / 180.0) / 10); // Scaling factor for x
        int y = static_cast<int>(lidarPosition.y - distance * cos(angle * CV_PI / 180.0) / 10); // Scaling factor for y

        // Draw a line from the LiDAR position to each point to clear the space
        line(display, lidarPosition, Point(x, y), Scalar(0, 255, 0), 2); // Draw in green color to indicate free space

        // Check if the point is within the 2-meter range
        if (distance <= 2000) {
            obstaclePoints.push_back(Point(x, y)); // Store points within 2 meters as part of an obstacle
        }
    }

    // Enhanced distance-based clustering
    std::vector<std::vector<Point>> clusters;
    const float min_object_width_px = 10.0; // Minimum object width in pixels (approx. 10 cm)
    const float cluster_gap_threshold_px = 50.0; // Gap threshold in pixels to distinguish separate objects

    for (const auto& p : obstaclePoints) {
        bool added_to_cluster = false;

        // Try to add the point to an existing cluster
        for (auto& cluster : clusters) {
            bool close_to_cluster = false;

            // Check if the point is close to any point in the cluster
            for (const auto& cluster_point : cluster) {
                float distance_to_point = sqrt(pow(p.x - cluster_point.x, 2) + pow(p.y - cluster_point.y, 2));
                if (distance_to_point < cluster_gap_threshold_px) {
                    close_to_cluster = true;
                    break;
                }
            }

            if (close_to_cluster) {
                cluster.push_back(p);
                added_to_cluster = true;
                break;
            }
        }

        // If the point was not added to any cluster, start a new cluster
        if (!added_to_cluster) {
            clusters.push_back({ p });
        }
    }

    // Draw bounding boxes for each detected object and display their distances
    for (const auto& cluster : clusters) {
        if (cluster.size() < 2) continue; // Skip small clusters that cannot form a bounding box

        // Calculate the center point of all obstacle points in the cluster
        Point2f center(0, 0);
        for (const auto& p : cluster) {
            center.x += p.x;
            center.y += p.y;
        }
        center.x /= cluster.size();
        center.y /= cluster.size();

        // Calculate maximum distance from the center to define the bounding box size
        float max_dist_x = 0.0, max_dist_y = 0.0;
        for (const auto& p : cluster) {
            max_dist_x = std::max(max_dist_x, fabs(p.x - center.x));
            max_dist_y = std::max(max_dist_y, fabs(p.y - center.y));
        }

        // Define the bounding box centered at the computed center with the calculated size
        Rect boundingBox(center.x - max_dist_x - fixed_width_extension / 2,
                         center.y - max_dist_y - fixed_length_extension / 2,
                         2 * max_dist_x + fixed_width_extension,
                         2 * max_dist_y + fixed_length_extension);

        // Draw the bounding box around the detected obstacle
        rectangle(display, boundingBox, Scalar(255, 0, 0), 2); // Draw in blue color to indicate the obstacle

        // Calculate the distance from the LiDAR to the center of the bounding box
        float distance_to_center = sqrt(pow(center.x - lidarPosition.x, 2) + pow(center.y - lidarPosition.y, 2)) * 10 / 1000.0; // Convert pixels to meters
        char distance_text[50];
        sprintf(distance_text, "Distance: %.2f m", distance_to_center);
        putText(display, distance_text, center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1); // Display distance text
    }

    imshow("Lidar Data Visualization", display);
    waitKey(1);  // Refresh display continuously
}

int main(int argc, const char * argv[]) {
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
#ifdef _WIN32
            opt_channel_param_first = "\\\\.\\com3";
#elif __APPLE__
            opt_channel_param_first = "/dev/tty.SLAB_USBtoUART";
#else
            opt_channel_param_first = "/dev/ttyUSB0";
#endif
        }
    }

    ILidarDriver* drv = *createLidarDriver();

    if (!drv) {
        fprintf(stderr, "Insufficient memory, exit\n");
        return -2;
    }

    sl_lidar_response_device_info_t devinfo;
    bool connectSuccess = false;

    // Start of connection handling
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

        std::vector<std::pair<float, float>> lidarPoints; // To store LiDAR points
        const int imgWidth = 800;   // Width of the display window
        const int imgHeight = 600;  // Height of the display window
        Point lidarPosition = Point(imgWidth / 2, imgHeight); // Bottom center position
        Mat display(imgHeight, imgWidth, CV_8UC3, Scalar(0, 0, 255)); // Initialize as filled

        // Fixed values for obstacle size extension
        float fixed_length_extension = 100.0;  // Example value for length extension in mm
        float fixed_width_extension = 50.0;    // Example value for width extension in mm

        while (true) {
            sl_lidar_response_measurement_node_hq_t nodes[8192];
            size_t count = _countof(nodes);

            op_result = drv->grabScanDataHq(nodes, count);
            clearScreen();
            printf("Set_Start.\n");

            if (SL_IS_OK(op_result)) {
                drv->ascendScanData(nodes, count);
                lidarPoints.clear(); // Clear previous points

                for (int pos = 0; pos < (int)count; ++pos) {
                    float raw_angle_degrees = (nodes[pos].angle_z_q14 * 90.f) / 16384.f;
                    float distance = nodes[pos].dist_mm_q2 / 4.0f;
                    float quality = nodes[pos].quality;

                    // Filter points within the specified angle range of +60 to -60 degrees
                    if ((raw_angle_degrees >= 0.0f && raw_angle_degrees <= 60.0f) || 
                        (raw_angle_degrees >= 300.0f && raw_angle_degrees <= 360.0f)) {
                        
                        if (quality > 0) {
                            lidarPoints.emplace_back(raw_angle_degrees, distance); // Store the point
                            printf("%03.2f %08.2f\n", raw_angle_degrees, distance);
                        }
                    }
                }

                // Draw Lidar points and Bounding Boxes around detected obstacles
                drawLidarPointsAndBoundingBoxes(display, lidarPosition, lidarPoints, fixed_length_extension, fixed_width_extension);  
            }

            if (ctrl_c_pressed) {
                break;
            }
        }

        drv->stop();
        delay(200);
        if (opt_channel_type == CHANNEL_TYPE_SERIALPORT)
            drv->setMotorSpeed(0);

    } while (0); // End of do-while block

    if (drv) {
        delete drv;
        drv = NULL;
    }
    return 0;
}

//sudo ./ultra_simple --channel --serial /dev/ttyUSB0 1000000