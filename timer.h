#pragma once

#include <chrono>
#include <vector>


class Timer {
public:
    Timer();

    // Start the timer
    void start();

    // Update the timer and record frame time
    void update();

    // Getters for time-related data
    float getTimeElapsed() const;
    const std::vector<float>& getPreviousTimes() const;

private:
    std::vector<float> previous_times;
    std::chrono::high_resolution_clock::time_point newt;
    std::chrono::high_resolution_clock::time_point oldt;
    std::chrono::high_resolution_clock::time_point start_time;
    float time_elapsed;
    static constexpr size_t ESTIMATE_FROM_LAST_FRAMES = 60;
};
